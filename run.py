#!/usr/bin/env python3
"""实时管线一键运行编排脚本。

启动顺序（ASR 模型加载慢，先就绪再开跑，避免管线空跑或段积压）：

1. ASR 跟随进程（python -m asr.app --follow）先启动，轮询等待其就绪哨兵
   （--ready_file，模型加载完成后 touch）；ASR 进程提前退出或超时则整体失败；
2. ASR 就绪后启动 diarization 管线（python -m diarization.app）和 viewer 服务器；
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

    python3 run.py [audio_input] [--config config/config.yaml] [--run_name default]
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common.run_helpers import (
    add_common_args,
    append_run_header,
    build_pipeline_cmd,
    env_flag,
)

BASE_DIR = Path(__file__).resolve().parent

# ASR 模型加载（含可能的下载）等待上限。
_DEFAULT_READY_TIMEOUT = 3600.0

_TEST_NAME = "common"  # 输出固定在 {output_root}/common/{run_name}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实时管线一键运行（ASR 先就绪，再启管线与 viewer）"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="./datasets",
        help="输入音频（wav）",
    )
    add_common_args(parser)
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
        "--viewer_port", type=int, default=int(os.environ.get("VIEWER_PORT", "9331"))
    )
    parser.add_argument(
        "--asr_ready_timeout",
        type=float,
        default=_DEFAULT_READY_TIMEOUT,
        help="等待 ASR 模型就绪的超时时间（秒）",
    )
    return parser


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
    args = build_parser().parse_args()

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

    # ---- 组装子进程命令 ----
    pipeline_cmd = build_pipeline_cmd(args, audio, exp_dir, config_path)

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
        log.write(f"Command: {shlex.join(pipeline_cmd)}\n")
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

        # 2. ASR 就绪后启动 viewer 与 diarization 管线。
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

        print(f"Starting pipeline: {shlex.join(pipeline_cmd)}")
        pipeline_rc = subprocess.run(pipeline_cmd).returncode

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
