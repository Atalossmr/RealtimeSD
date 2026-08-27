#!/usr/bin/env python3
"""实时管线一键运行编排脚本。

启动顺序（ASR 模型加载慢，先就绪再开跑，避免管线空跑或段积压）：

1. ASR 跟随进程（python -m asr.app --follow）先启动，轮询等待其就绪哨兵
   （--ready_file，模型加载完成后 touch）；ASR 进程提前退出或超时则整体失败；
2. ASR 就绪后启动 viewer 服务器，并在本进程内跑 diarization 管线
   （`run_diarization` = extract + cluster + 分段音频导出）；
3. 管线结束（无论成败）→ touch done 哨兵 → 等 ASR 收尾落盘 transcript →
   打印"音频已处理完成"并挂起等待（viewer 保持可访问）；用户 Ctrl+C
   （或 server 因其他原因退出，如 POST /api/shutdown）后脚本才退出并
   关闭 viewer。

环境变量作为对应命令行参数的缺省值：

    WITH_ASR=0      → --no-asr        CONFIG_PATH  → --config
    MODEL_PATH      → --model_path    HF_TOKEN     → --hf_token
    HF_CACHE_DIR    → --hf_cache_dir  DEBUG=1      → --debug
    SHOW_RTTM=1     → --show_rttm     OUTPUT_ROOT  → --output_root
    RUN_NAME        → --run_name      VIEWER_PORT  → --viewer_port
    ASR_CONFIG_PATH → --asr_config    WAIT_VIEWER=0 → --no-wait

用法：

    python3 run.py <音频文件> [--config config/config.yaml] [--run_name default]
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from speakerlab.utils.fileio import load_audio

from common.log import log_structured, setup_logger
from common.pipeline_config import (
    ChunkPipelineConfig,
    config_from_args,
    merge_args_with_config,
)
from common.run_helpers import add_common_args, append_run_header, env_flag
from common.schema import ChunkArtifacts, ChunkDebugInfo, ChunkObservation
from diarization.cluster.backends import build_clusterer
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter
from diarization.cluster.runner import run_clustering
from diarization.extract.extractor import ChunkExtractor

BASE_DIR = Path(__file__).resolve().parent

# ASR 模型加载（含可能的下载）等待上限。
_DEFAULT_READY_TIMEOUT = 3600.0

_TEST_NAME = "common"  # 输出固定在 {output_root}/common/{run_name}

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实时管线一键运行（ASR 先就绪，再启管线与 viewer）"
    )
    parser.add_argument(
        "audio",
        help="输入音频文件（wav/flac/mp3；仅单个文件，不接受目录）",
    )
    add_common_args(parser)
    parser.add_argument(
        "--device", default="auto", help="运行设备，如 auto/cpu/cuda:0"
    )
    parser.add_argument(
        "--model_type",
        default="eres2netv2",
        choices=["eres2netv2"],
        help="speaker encoder 类型，当前仅支持 eres2netv2",
    )
    parser.add_argument(
        "--segmentation_model",
        default="pyannote/segmentation-3.0",
        help="pyannote segmentation 模型名",
    )
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    parser.add_argument(
        "--asr_config",
        default=os.environ.get("ASR_CONFIG_PATH", "./config/asr.yaml"),
        help="ASR 转写的 YAML 配置（传给 python -m asr.app）",
    )
    parser.add_argument(
        "--debug", action="store_true", default=env_flag("DEBUG", False)
    )
    parser.add_argument(
        "--show_rttm", action="store_true", default=env_flag("SHOW_RTTM", False)
    )
    parser.add_argument(
        "--no-asr",
        dest="asr",
        action="store_false",
        default=env_flag("WITH_ASR", True),
        help="不启动 ASR 跟随进程",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        default=env_flag("WITH_VIEWER", True),
        help="不启动 viewer 服务器",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        default=env_flag("WAIT_VIEWER", True),
        help="管线结束后不挂起等待 Ctrl+C，直接收尾退出（无人值守/批处理场景）",
    )
    parser.add_argument(
        "--viewer_port", type=int, default=int(os.environ.get("VIEWER_PORT", "8000"))
    )
    parser.add_argument(
        "--asr_ready_timeout",
        type=float,
        default=_DEFAULT_READY_TIMEOUT,
        help="等待 ASR 模型就绪的超时时间（秒）",
    )
    return parser


def _log_debug_chunk(
    *,
    chunk_index: int,
    chunk_start: float,
    commit_start: float,
    commit_end: float,
    seg_scores: np.ndarray,
    observations: list[ChunkObservation],
    local_to_global: dict[int, int],
    debug_info: ChunkDebugInfo,
    emitted_frames: int,
) -> None:
    """输出 chunk 级调试信息"""

    debug_summary = {
        "chunk_index": int(chunk_index),
        "chunk": {
            "start": round(float(chunk_start), 3),
            "commit_start": round(float(commit_start), 3),
            "end": round(float(commit_end), 3),
        },
        "segmentation_summary": {
            "shape": [int(dim) for dim in seg_scores.shape],
            "min": round(float(np.min(seg_scores)), 6),
            "max": round(float(np.max(seg_scores)), 6),
            "mean": round(float(np.mean(seg_scores)), 6),
        },
        "window_state": {
            "observations": int(len(observations)),
            "embedded": int(len(observations)),
            "emitted": int(emitted_frames),
        },
        "assignment": {
            "local_to_global": {
                str(local_idx): int(global_id)
                for local_idx, global_id in sorted(local_to_global.items())
            },
            "local_assignments": debug_info["local_assignments"],
        },
        "centroids": {
            "before": int(debug_info["num_centroids_before"]),
            "after": int(debug_info["num_centroids_after"]),
        },
    }
    log_structured(logger, logging.DEBUG, "[debug]", "window_summary", debug_summary)

    for key in ("new_speakers", "updated_speakers", "skipped_updates"):
        if debug_info[key]:
            log_structured(logger, logging.DEBUG, "[debug]", key, debug_info[key])
    if debug_info["global_speakers"]:
        log_structured(
            logger,
            logging.INFO,
            "[runtime]",
            "current_global_speakers",
            debug_info["global_speakers"],
        )


def run_diarization(
    config: ChunkPipelineConfig,
    audio_path: str,
    exp_dir: Path,
    model_path: Optional[str] = None,
) -> str:
    """单条音频的 diarization 管线：extract + cluster + RTTM/segments 输出。

    供 run.py main 与 test_der.py 复用；返回 raw RTTM 路径。
    """

    config.output_dir = str(exp_dir)

    extractor = ChunkExtractor(config, model_path)
    clusterer = build_clusterer(config)

    waveform = load_audio(audio_path, obj_fs=config.sample_rate)
    waveform = extractor.prepare_waveform(waveform, config.sample_rate)

    uri = Path(audio_path).stem
    raw_rttm_path = str(exp_dir / "rttm" / f"{uri}.{clusterer.output_tag}.rttm")
    writer = AppendOnlyRTTMWriter(
        raw_rttm_path,
        uri,
        config.min_segment_duration,
        config.segment_merge_gap,
        config.show_rttm,
    )

    # 音频片段导出：仅 streaming 后端下逐 commit 区生效。
    # 默认 WavSegmentSink 落盘 wav + manifest，转写由 ASR 跟随进程
    # 读取输出目录独立完成；TIGER 仍是按需惰性加载。
    exporter = None
    if config.separation_enabled:
        if clusterer.deferred:
            logger.warning(
                "[separation] separation 仅支持 streaming 后端，当前后端 %s 已跳过",
                type(clusterer).__name__,
            )
        else:
            from separation import StreamingSegmentExporter

            exporter = StreamingSegmentExporter(
                config,
                waveform,
                extractor.embedder,
                uri=uri,
                output_dir=str(exp_dir / "segments"),
            )

    def chunk_log_hook(
        chunk: ChunkArtifacts,
        local_to_global: Optional[dict[int, int]],
        debug_info: ChunkDebugInfo,
        emitted_frames: int,
    ) -> None:
        """# 每 chunk 分配完成后的回调，负责日志与分段导出"""

        # 输出本身由 runner/writer 完成。deferred 后端下 local_to_global
        # 为 None、emitted_frames 为 0（重放阶段不再回调）。
        # 分段音频导出：commit 区音频段输出（重叠区经 TIGER 分离）。
        if exporter is not None and local_to_global:
            exporter.handle_chunk(chunk, local_to_global)
        log_structured(
            logger,
            logging.INFO,
            "[runtime]",
            "frame_decision",
            {
                "chunk_index": int(chunk.chunk_index),
                "chunk_start": round(float(chunk.chunk_start), 3),
                "commit": [
                    round(float(chunk.commit_start), 3),
                    round(float(chunk.commit_end), 3),
                ],
                "local_to_global": {
                    str(local_idx): int(global_id)
                    for local_idx, global_id in sorted(
                        (local_to_global or {}).items()
                    )
                },
            },
        )
        if config.debug:
            _log_debug_chunk(
                chunk_index=chunk.chunk_index,
                chunk_start=chunk.chunk_start,
                commit_start=chunk.commit_start,
                commit_end=chunk.commit_end,
                seg_scores=chunk.seg_scores,
                observations=chunk.observations,
                local_to_global=local_to_global or {},
                debug_info=debug_info,
                emitted_frames=emitted_frames,
            )

    # refined 级（两后端通用）：streaming 逐 chunk + merge 事件动态重生成，
    # ahc（deferred）在 EOF 重放完成后最终刷新；均在 EOF 叠加小样本合并。
    from diarization.cluster.post_process import RefinedRTTMWriter

    base = raw_rttm_path[: -len(f".{clusterer.output_tag}.rttm")]
    refiner = RefinedRTTMWriter(
        raw_rttm_path,
        f"{base}.refined.rttm",
        clusterer,
        min_count=config.post_merge_min_count,
        min_similarity=config.post_merge_min_similarity,
    )

    # 生成器惰性生产 chunk：streaming 后端下输出延迟 ≈ 一个 hop，
    # 不会因为组合成统一循环而引入额外的缓冲。
    try:
        run_clustering(
            extractor.iter_chunk_artifacts(waveform),
            clusterer,
            writer,
            chunk_hook=chunk_log_hook,
            refiner=refiner,
        )
    finally:
        # 闭合 exporter 残余的 open segment（音频结束，与 writer.finalize 同步）；
        # 中途异常也要收尾，否则已累积的段音频不落盘。
        if exporter is not None:
            exporter.finalize()
    return raw_rttm_path


def _check_asr_config(config_path: Path) -> None:
    """WITH_ASR 需要 config 里开启分段导出且使用 streaming 后端。

    ahc 后端不产出 segments manifest，ASR follow 进程会空转到超时。
    """

    with open(config_path, encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not data.get("separation_enabled"):
        raise SystemExit(
            f"ERROR: ASR 跟随需要 {config_path} 中 separation_enabled 为 true"
        )
    backend = data.get("clustering_backend", "streaming")
    if backend != "streaming":
        raise SystemExit(
            f"ERROR: ASR 跟随需要 {config_path} 中 clustering_backend 为 streaming"
            f"（当前为 {backend!r}，该后端不产出分段 manifest）"
        )


def _check_input_paths(audio: Path, config_path: Path) -> None:
    """启动前 fail-fast：音频/配置不存在时尽早报错，避免模型加载完才发现。"""

    if not audio.exists():
        raise SystemExit(f"ERROR: 音频输入不存在: {audio}")
    if audio.is_dir():
        raise SystemExit(f"ERROR: 音频输入只接受单个文件，不支持目录: {audio}")
    if not config_path.exists():
        raise SystemExit(f"ERROR: 配置文件不存在: {config_path}")


def _wait_asr_ready(
    asr_proc: subprocess.Popen, ready_file: Path, timeout: float
) -> None:
    """轮询等 ASR 模型就绪；进程早退或超时则报错退出。"""

    deadline = time.monotonic() + timeout
    while not ready_file.exists():
        if asr_proc.poll() is not None:
            raise RuntimeError(
                f"ASR 进程在就绪前退出（rc={asr_proc.returncode}），见 transcribe.log"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"等待 ASR 就绪超时（{timeout}s）")
        time.sleep(1.0)


def _finish_asr(asr_proc: subprocess.Popen, done_file: Path) -> None:
    """落 done 哨兵放行 ASR 收尾，等其落盘；超时先 SIGTERM，仍不退出再 SIGKILL。"""

    done_file.touch()
    try:
        asr_proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        # 收尾超时：先 SIGTERM，仍不退出再 SIGKILL，避免 run.py 永久挂起。
        asr_proc.terminate()
        try:
            asr_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            asr_proc.kill()
            asr_proc.wait()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    audio = Path(args.audio)
    config_path = Path(args.config)
    _check_input_paths(audio, config_path)
    basic_dir = Path(args.output_root) / _TEST_NAME
    exp_dir = basic_dir / args.run_name
    results_file = basic_dir / "results.txt"
    done_file = exp_dir / ".diarization_done"
    ready_file = exp_dir / ".asr_ready"

    if args.asr:
        _check_asr_config(config_path)

    # 合并 CLI 与 YAML（argparse 默认值 < YAML < 显式 CLI），构建管线配置。
    merged_args = merge_args_with_config(parser, args, sys.argv[1:])
    # 沿用原 build_pipeline_cmd 的传参语义：--debug/--show_rttm 为 true
    # （含环境变量缺省）时作为显式 CLI 覆盖 YAML；为 false 时落回 YAML。
    if args.debug:
        merged_args.debug = True
    if args.show_rttm:
        merged_args.show_rttm = True
    config = config_from_args(merged_args)

    # 只清理本次 run 自己的输出目录，同 output_root 下的其他历史 run 保留。
    shutil.rmtree(exp_dir, ignore_errors=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Running with YAML config")
    print(f"audio_input={audio}")
    print(f"config_path={config_path}")
    print(f"output_dir={exp_dir}")
    for name in ("model_path", "hf_cache_dir"):
        if getattr(args, name):
            print(f"{name} override enabled")
    if args.hf_token:
        print("hf_token override enabled")
    print(f"with_asr={args.asr} with_viewer={args.viewer}")
    print("==========================================")

    # results.txt 按 run 追加：同一 output_root 下历史 run 的汇总记录保留。
    append_run_header(
        results_file,
        args.run_name,
        "Online Speaker Diarization pipeline",
        {
            "audio_input": audio,
            "config_path": config_path,
            "model_path_override": args.model_path,
            "hf_cache_dir_override": args.hf_cache_dir,
            "with_asr": args.asr,
        },
    )

    # ---- 组装子进程命令（ASR / viewer；diarization 已改为本进程内调用） ----
    asr_cmd = [
        sys.executable,
        "-m",
        "asr.app",
        "--segments_dir",
        str(exp_dir / "segments"),
        "--output_dir",
        str(exp_dir / "transcripts"),
        "--log_file",
        str(exp_dir / "logs" / "transcribe.log"),
        "--config",
        str(args.asr_config),
        "--follow",
        "--done_file",
        str(done_file),
        "--ready_file",
        str(ready_file),
    ]
    if args.debug:
        asr_cmd.append("--verbose")

    uri = audio.stem
    viewer_cmd = [
        sys.executable,
        "viewer/server.py",
        "--exp_root",
        str(basic_dir),
        "--audio_root",
        "datasets",
        "--audio",
        f"{uri}={audio}",
        "--port",
        str(args.viewer_port),
    ]

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "command.log", "w", encoding="utf-8") as log:
        log.write(f"Audio: {audio}\n")
        log.write(f"Config: {config_path}\n")
        log.write(
            "Pipeline: in-process run_diarization"
            f" (device={config.device}, model_type={config.model_type},"
            f" segmentation_model={config.segmentation_model},"
            f" backend={config.clustering_backend})\n"
        )
        if args.asr:
            log.write(f"ASR Command: {shlex.join(asr_cmd)}\n")
        if args.viewer:
            log.write(f"Viewer Command: {shlex.join(viewer_cmd)}\n")

    asr_proc: subprocess.Popen | None = None
    viewer_proc: subprocess.Popen | None = None
    pipeline_rc = 0
    try:
        # 1. ASR 先启动并等模型就绪。
        if args.asr:
            print(f"Starting ASR follow process: {shlex.join(asr_cmd)}")
            asr_proc = subprocess.Popen(asr_cmd)
            print("Waiting for ASR model ready ...")
            _wait_asr_ready(asr_proc, ready_file, args.asr_ready_timeout)
            print("ASR model ready.")

        # 2. ASR 就绪后启动 viewer 与 diarization 管线（本进程内）。
        if args.viewer:
            viewer_log = open(logs_dir / "viewer.log", "w", encoding="utf-8")
            try:
                viewer_proc = subprocess.Popen(
                    viewer_cmd,
                    stdout=viewer_log,
                    stderr=subprocess.STDOUT,
                )
            finally:
                # Popen 已复制 fd，父进程持有的句柄可以立即关闭。
                viewer_log.close()
            time.sleep(0.5)
            if viewer_proc.poll() is not None:
                print(
                    f"WARNING: viewer 启动失败，见 {logs_dir}/viewer.log",
                    file=sys.stderr,
                )
                viewer_proc = None
            else:
                print(f"Viewer: http://127.0.0.1:{args.viewer_port}")

        # 管线日志写入 logs/run.log（控制台只保留脚本自身输出）；
        # --debug 时同步提升日志级别（沿用原 build_pipeline_cmd 的约定）。
        setup_logger(
            bool(merged_args.verbose or args.debug), str(logs_dir / "run.log")
        )
        print(f"Starting diarization pipeline (in-process): {audio}")
        try:
            run_diarization(config, str(audio), exp_dir, model_path=merged_args.model_path)
        except Exception:
            # 管线失败不跳过收尾：记录后继续走 ASR/viewer 的 finally 清理。
            logger.exception("Diarization pipeline failed")
            print(f"ERROR: 管线执行失败，见 {logs_dir}/run.log", file=sys.stderr)
            pipeline_rc = 1

        # 管线结束：先落哨兵放行 ASR 收尾并等其落盘，再进入交互等待，
        # 保证打印"音频已处理完成"时 transcript 已是最终版。
        if asr_proc is not None:
            _finish_asr(asr_proc, done_file)
            asr_proc = None

        # 管线跑完后默认不立即退出：viewer 保持可访问，等用户 Ctrl+C
        # （或 server 因其他原因退出，poll 检测到后同样放行）；
        # --no-wait / WAIT_VIEWER=0 时跳过等待，直接收尾（无人值守场景）。
        if viewer_proc is not None and args.wait:
            if pipeline_rc == 0:
                print("音频已处理完成。")
            else:
                print(f"管线已退出（rc={pipeline_rc}）。")
            print(
                f"Viewer 仍在运行: http://127.0.0.1:{args.viewer_port}"
                "（按 Ctrl+C 结束并关闭 viewer）"
            )
            try:
                while viewer_proc.poll() is None:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                # 管线已正常结束，这次 Ctrl+C 只用于关闭 viewer，不向外套层抛。
                print("\nShutting down viewer ...", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down ...", file=sys.stderr)
        pipeline_rc = 130
    finally:
        # 中断/异常路径的统一收尾：ASR 正常路径已在交互等待前完成（asr_proc
        # 已置 None），这里只兜住管线运行中被 Ctrl+C 或出错的情况。
        if asr_proc is not None:
            _finish_asr(asr_proc, done_file)
        # viewer 在脚本退出前关闭（正常路径下它在交互等待结束时才走到这里）。
        if viewer_proc is not None:
            viewer_proc.terminate()
            try:
                viewer_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                viewer_proc.kill()
                viewer_proc.wait()
            viewer_proc = None

    # ---- 汇总 ----
    rttm_count = len(list((exp_dir / "rttm").glob("*.refined.rttm")))
    with open(results_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(
            f"{args.run_name} -> refined_rttm_files={rttm_count} | config={config_path}\n"
        )
        if args.asr:
            transcript_count = len(
                list((exp_dir / "transcripts").glob("*.transcript.jsonl"))
            )
            file_obj.write(
                f"{args.run_name} -> transcript_files={transcript_count} | config={config_path}\n"
            )
    print(f"Result: {args.run_name} -> refined_rttm_files={rttm_count}")
    print(f"Pipeline log: {exp_dir}/logs/run.log")
    if args.asr:
        print(f"ASR log: {exp_dir}/logs/transcribe.log")

    return pipeline_rc


if __name__ == "__main__":
    sys.exit(main())
