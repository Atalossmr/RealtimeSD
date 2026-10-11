"""编排层：把 diarization 装配成完整管线（音频 → RTTM）。

公开入口 `process(audio=...)` 统一支持单文件与批量，run.py 与
test_der.py 均为其薄 CLI 封装；`DiarizationRunner` 为组合执行核心
（模型只加载一次，可连续处理多条音频）；`run_der_evaluation` 对产物
计算 DER；`add_common_args` 等为入口脚本共用的 CLI 辅助。

oracle 模式（--oracle --ref）：用 ref RTTM 驱动的 OracleFrontend 替代
segmentation 前端（不加载 segmentation 模型），与
`diarization.extract.oracle.oracle_chunk_artifacts` 组合后走同一
聚类/输出路径，用于上限实验。

归属判断标准：需要知道 ≥2 个模块协作语义的代码归这里；纯通用机制或
被其他模块直接 import 的归本包底座子包（config / contracts / utils /
models）。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np

from .cluster.backends import build_clusterer
from .cluster.post_process import RefinedRTTMWriter
from .cluster.rttm_writer import AppendOnlyRTTMWriter
from .cluster.runner import run_clustering
from .config import ChunkPipelineConfig, load_config_and_runtime
from .constants import BASE_DIR
from .contracts import ChunkArtifacts, ChunkDebugInfo, ChunkObservation
from .extract.extractor import ChunkExtractor
from .extract.oracle import oracle_chunk_artifacts
from .frontend import ChunkTrackBuilder, OracleFrontend
from .models.embedding_infer import build_region_embedder
from .speakerlab.utils.fileio import load_audio
from .utils import log_structured, setup_logger
from .utils.audio import prepare_mono_waveform

logger = logging.getLogger(__name__)


# ======================================================================
# 入口脚本（run.py / test_der.py）共用的 CLI 辅助（只依赖标准库）
# ======================================================================


def env_flag(name: str, default: bool) -> bool:
    """环境变量作为布尔缺省值：仅 "1" 为真。

    Args:
        name: 环境变量名。
        default: 环境变量未设置时的缺省值。

    Returns:
        环境变量未设置时返回 default，否则返回其是否为 "1"。
    """

    raw = os.environ.get(name)
    return default if raw is None else raw == "1"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """run.py 与 test_der.py 共有的 CLI 参数（缺省值取对应环境变量）。

    Args:
        parser: 待添加参数的 ArgumentParser。
    """

    parser.add_argument(
        "--config", default=os.environ.get("CONFIG_PATH", "./config/config.yaml")
    )
    parser.add_argument("--output_root", default=os.environ.get("OUTPUT_ROOT", "./exp"))
    parser.add_argument("--run_name", default=os.environ.get("RUN_NAME", "default"))
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH") or None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN") or None)
    parser.add_argument(
        "--hf_cache_dir", default=os.environ.get("HF_CACHE_DIR") or None
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        default=env_flag("ORACLE", False),
        help="oracle 模式：用 --ref 的参考 RTTM 替代 segmentation 前端"
        "（不加载 segmentation 模型），用于上限实验",
    )
    parser.add_argument(
        "--ref",
        default=os.environ.get("REF") or None,
        help="oracle 模式的参考 RTTM：单文件给 RTTM 路径，批量给含 "
        "{uri}.rttm 的目录；--oracle 时必填",
    )


def check_input_paths(audio: Path, config_path: Path) -> None:
    """启动前 fail-fast：音频/配置不存在时尽早报错，避免模型加载完才发现。

    audio 接受单个音频文件、目录或路径清单文本（与 collect_audio_paths
    的三种输入形式一致）。

    Args:
        audio: 音频输入路径（文件、目录或清单文本）。
        config_path: YAML 配置路径。

    Raises:
        SystemExit: audio 或 config_path 不存在。
    """

    if not audio.exists():
        raise SystemExit(f"ERROR: 音频输入不存在: {audio}")
    if not config_path.exists():
        raise SystemExit(f"ERROR: 配置文件不存在: {config_path}")


# ======================================================================
# 一次完整运行的公开 API：process(audio=...)（单文件与批量统一）
# ======================================================================


@dataclass
class RunDirs:
    """一次运行的全部路径约定（{output_root}/{test_name}/{run_name}）。

    Attributes:
        basic_dir: {output_root}/{test_name} 目录。
        exp_dir: 本次 run 的输出目录 {output_root}/{test_name}/{run_name}。
        results_file: basic_dir 下按 run 追加的汇总记录。
        logs_dir: exp_dir 下的日志目录。
    """

    basic_dir: Path
    exp_dir: Path
    results_file: Path
    logs_dir: Path


def resolve_run_dirs(
    output_root: str | Path, test_name: str, run_name: str
) -> RunDirs:
    """纯路径计算（无副作用）；目录清理/创建由 process 负责。

    Args:
        output_root: 输出根目录。
        test_name: 测试/批次名。
        run_name: 本次运行名。

    Returns:
        本次运行的全部路径约定。
    """

    basic_dir = Path(output_root) / test_name
    exp_dir = basic_dir / run_name
    return RunDirs(
        basic_dir=basic_dir,
        exp_dir=exp_dir,
        results_file=basic_dir / "results.txt",
        logs_dir=exp_dir / "logs",
    )


def _prepare_run_dirs(dirs: RunDirs) -> None:
    """只清理本次 run 自己的输出目录，同 output_root 下的其他历史 run 保留。

    Args:
        dirs: 本次运行的路径约定。
    """

    shutil.rmtree(dirs.exp_dir, ignore_errors=True)
    dirs.logs_dir.mkdir(parents=True, exist_ok=True)


def _append_run_header(
    results_file: Path,
    run_name: str,
    title: str,
    fields: dict[str, object],
) -> None:
    """向 results.txt 追加一个 run 的记录头（按 run 追加，历史汇总保留）。

    Args:
        results_file: 汇总记录文件。
        run_name: 本次运行名。
        title: 记录头标题。
        fields: 写出的键值字段；值为 None 的键不写出。
    """

    with open(results_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"run: {run_name} | {title}\n")
        file_obj.write("-" * 40 + "\n")
        for key, value in fields.items():
            if value is not None:
                file_obj.write(f"{key}: {value}\n")
        file_obj.write("\n")


@dataclass
class ProcessResult:
    """`process` 的运行结果。相对路径字段均相对于 `exp_dir`。

    Attributes:
        returncode: 退出码（0 成功；1 有文件失败；130 被 Ctrl+C 中断）。
        exp_dir: 本次 run 的输出目录。
        results_file: 汇总记录文件。
        logs_dir: 日志目录。
        failed: 处理失败的音频路径。
        raw_rttms: 成功文件的 raw RTTM 路径。
        rttm_count: refined RTTM 文件数。
    """

    returncode: int
    exp_dir: Path
    results_file: Path
    logs_dir: Path
    failed: list[str] = field(default_factory=list)  # 处理失败的音频路径
    raw_rttms: list[str] = field(default_factory=list)
    rttm_count: int = 0


def process(
    audio: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    config: Optional[ChunkPipelineConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, object]] = None,
    output_root: Union[str, Path] = "./exp",
    run_name: str = "default",
    test_name: str = "common",
    model_path: Optional[str] = None,
    oracle: bool = False,
    ref: Optional[str] = None,
    verbose: bool = False,
    header_fields: Optional[Mapping[str, object]] = None,
    header_title: str = "Online Speaker Diarization pipeline",
) -> ProcessResult:
    """完整管线：音频 → RTTM（diarization）。

    批量时全部音频的产物写入同一输出目录（uri 区分文件名），模型只加载
    一次，单文件失败记入 `failed` 并继续，有失败时 returncode=1；Ctrl+C
    置 rc=130。

    Args:
        audio: 单个音频路径或路径列表。
        config: 管线配置；缺省时经 `load_config_and_runtime(config_path,
            overrides)` 无 argparse 构建（config_path 缺省为仓库自带
            config/config.yaml）。CLI 入口应自行合并后传入（此时
            config_path 仅用于运行记录）。
        config_path: YAML 配置路径；仅 config 缺省时用于加载，否则只作记录。
        overrides: 加载配置时的覆盖项。
        output_root: 输出根目录。
        run_name: 本次运行名（输出目录为 {output_root}/{test_name}/{run_name}）。
        test_name: 测试/批次名。
        model_path: 显式模型路径；优先级高于 YAML 的 model_path 运行时键。
        oracle: True 时用 ref RTTM 驱动的 oracle 前端替代 segmentation
            前端（不加载 segmentation 模型），用于上限实验。
        ref: oracle 模式的参考 RTTM：单文件给 RTTM 路径，批量给含
            {uri}.rttm 的目录；oracle=True 时必填。
        verbose: 提升日志级别。
        header_fields: results.txt 记录头字段；缺省为内置字段集，值为
            None 的键不写出。
        header_title: results.txt 记录头标题。

    Returns:
        运行结果（退出码、输出目录、失败列表、产物计数等）。

    Raises:
        ValueError: audio 为空列表。
    """

    if isinstance(audio, (str, Path)):
        audio_paths = [str(audio)]
    else:
        audio_paths = [str(p) for p in audio]
    if not audio_paths:
        raise ValueError("audio must contain at least one path")
    if config is None:
        config, runtime = load_config_and_runtime(config_path, dict(overrides or {}))
        if model_path is None:
            # YAML 的 model_path 是运行时键（CLI 下为 argparse 参数）：
            # 显式 model_path 参数优先，YAML 其次。
            model_path = runtime.get("model_path")

    dirs = resolve_run_dirs(output_root, test_name, run_name)
    _prepare_run_dirs(dirs)

    # results.txt 按 run 追加：同一 output_root 下历史 run 的汇总记录保留。
    _append_run_header(
        dirs.results_file,
        run_name,
        header_title,
        dict(
            header_fields
            or {
                "audio_input": audio_paths[0]
                if len(audio_paths) == 1
                else f"{len(audio_paths)} file(s)",
                "config_path": config_path,
                "model_path_override": model_path,
            }
        ),
    )

    with open(dirs.logs_dir / "command.log", "w", encoding="utf-8") as log:
        log.write(f"Audio: {len(audio_paths)} file(s)\n")
        for path in audio_paths:
            log.write(f"  - {path}\n")
        log.write(f"Config: {config_path}\n")
        log.write(
            "Pipeline: in-process diarization.pipeline.DiarizationRunner"
            f" (device={config.device}, model_type={config.model_type},"
            f" segmentation_model={config.segmentation_model},"
            f" backend={config.clustering_backend})\n"
        )

    # 管线日志写入 logs/run.log（控制台只保留编排输出）；
    # verbose/debug 时同步提升日志级别。
    setup_logger(verbose or config.debug, str(dirs.logs_dir / "run.log"))

    rc = 0
    failed: list[str] = []
    raw_rttms: list[str] = []
    try:
        # 本进程内逐文件跑 diarization 管线。
        # 已知风险（暂不修，留待后续去重/fail-fast）：uri 取
        # Path(audio_path).stem，同批两个不同目录的同名文件（如
        # a/01.wav 与 b/01.wav）会共用 uri——raw RTTM 相互覆盖。
        # 调用方应避免同批传入 stem 相同的音频。
        runner = DiarizationRunner(config, model_path, oracle=oracle)
        for audio_path in audio_paths:
            logger.info("Processing %s", audio_path)
            print(f"Processing {audio_path} ...")
            try:
                raw_rttms.append(runner.process_file(audio_path, dirs.exp_dir, ref))
            except Exception:
                failed.append(audio_path)
                rc = 1
                logger.exception("Diarization pipeline failed for %s", audio_path)
                print(
                    f"ERROR: 管线执行失败: {audio_path}，见 {dirs.logs_dir}/run.log",
                    file=sys.stderr,
                )
                continue
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down ...", file=sys.stderr)
        rc = 130

    # ---- 汇总 ----
    rttm_count = len(list((dirs.exp_dir / "rttm").glob("*.refined.rttm")))
    with open(dirs.results_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(
            f"{run_name} -> refined_rttm_files={rttm_count} | config={config_path}\n"
        )
    print(
        f"Result: {run_name} -> refined_rttm_files={rttm_count}"
        f" (failed={len(failed)}/{len(audio_paths)})"
    )
    print(f"Pipeline log: {dirs.logs_dir}/run.log")

    return ProcessResult(
        returncode=rc,
        exp_dir=dirs.exp_dir,
        results_file=dirs.results_file,
        logs_dir=dirs.logs_dir,
        failed=failed,
        raw_rttms=raw_rttms,
        rttm_count=rttm_count,
    )


# ======================================================================
# 组合执行：DiarizationRunner（extract + cluster + RTTM）
# ======================================================================


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
    """输出 chunk 级调试信息。"""

    debug_summary = {
        "chunk_index": int(chunk_index),
        "chunk": {
            "start": round(float(chunk_start), 3),
            "commit_start": round(float(commit_start), 3),
            "end": round(float(commit_end), 3),
        },
        "segmentation_summary": {
            "shape": [int(dim) for dim in seg_scores.shape],
            # oracle 前端在无活跃 speaker 的 chunk 产 0 列矩阵，min/max 无定义。
            "min": round(float(np.min(seg_scores)), 6) if seg_scores.size else None,
            "max": round(float(np.max(seg_scores)), 6) if seg_scores.size else None,
            "mean": round(float(np.mean(seg_scores)), 6) if seg_scores.size else None,
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


class DiarizationRunner:
    """可复用的 diarization 执行器：模型加载一次，可连续处理多条音频。

    每条音频独立构建 clusterer / writer（状态不跨文件），仅提取侧持有的
    模型跨文件复用（正常模式：segmentation + embedding；oracle 模式：仅
    embedding）。

    Attributes:
        config: 管线配置。
        oracle: 是否 oracle 模式（ref RTTM 替代 segmentation 前端）。
        extractor: 正常模式的 chunk 提取器（oracle 模式为 None）。
    """

    def __init__(
        self,
        config: ChunkPipelineConfig,
        model_path: Optional[str] = None,
        oracle: bool = False,
    ):
        """加载模型并初始化执行器。

        Args:
            config: 管线配置。
            model_path: 显式模型路径覆盖。
            oracle: True 时用 OracleFrontend（无 segmentation 模型）+
                RegionEmbedder 组合替代 ChunkExtractor。
        """

        self.config = config
        self.oracle = oracle
        if oracle:
            self.extractor = None
            self.oracle_frontend = OracleFrontend(config)
            self.region_embedder = build_region_embedder(config, model_path)
            self.track_builder = ChunkTrackBuilder(config)
        else:
            self.extractor = ChunkExtractor(config, model_path)

    @staticmethod
    def _resolve_ref_rttm(ref: Optional[str], audio_path: str) -> str:
        """解析当前音频对应的参考 RTTM 路径。

        Args:
            ref: --ref 传入的路径（RTTM 文件或含 {uri}.rttm 的目录）。
            audio_path: 当前音频路径（uri 取其文件名主干）。

        Returns:
            参考 RTTM 路径。

        Raises:
            ValueError: oracle 模式未提供 --ref，或按目录解析的 RTTM 不存在。
        """

        if ref is None:
            raise ValueError("oracle 模式需要 --ref 指定参考 RTTM（文件或目录）")
        ref_path = Path(ref)
        if ref_path.is_dir():
            candidate = ref_path / f"{Path(audio_path).stem}.rttm"
            if not candidate.is_file():
                raise ValueError(f"参考 RTTM 不存在: {candidate}")
            return str(candidate)
        return str(ref_path)

    def process_file(
        self, audio_path: str, exp_dir: Path, ref: Optional[str] = None
    ) -> str:
        """处理单条音频，输出 RTTM。

        Args:
            audio_path: 音频文件路径；uri 取其文件名主干。
            exp_dir: 本次 run 的输出目录。
            ref: oracle 模式的参考 RTTM（文件或目录）；正常模式忽略。

        Returns:
            raw RTTM 路径。
        """

        config = self.config
        config.output_dir = str(exp_dir)

        clusterer = build_clusterer(config)

        waveform = load_audio(audio_path, obj_fs=config.sample_rate)
        if self.oracle:
            ref_rttm = self._resolve_ref_rttm(ref, audio_path)
            waveform = prepare_mono_waveform(
                waveform, config.sample_rate, config.sample_rate
            )
            artifacts = oracle_chunk_artifacts(
                self.oracle_frontend.iter_prepared_chunks(waveform, ref_rttm),
                waveform,
                self.region_embedder,
                self.track_builder,
            )
        else:
            waveform = self.extractor.prepare_waveform(waveform, config.sample_rate)
            artifacts = self.extractor.iter_chunk_artifacts(waveform)

        uri = Path(audio_path).stem
        raw_rttm_path = str(exp_dir / "rttm" / f"{uri}.{clusterer.output_tag}.rttm")
        writer = AppendOnlyRTTMWriter(
            raw_rttm_path,
            uri,
            config.min_segment_duration,
            config.segment_merge_gap,
            config.show_rttm,
        )

        def chunk_log_hook(
            chunk: ChunkArtifacts,
            local_to_global: Optional[dict[int, int]],
            debug_info: ChunkDebugInfo,
            emitted_frames: int,
        ) -> None:
            """每 chunk 分配完成后的回调：日志输出。

            Args:
                chunk: 本 chunk 的产物。
                local_to_global: 局部到全局的说话人映射；deferred 后端下为
                    None（重放阶段不再回调）。
                debug_info: chunk 级调试信息。
                emitted_frames: 本 chunk 已提交帧数；deferred 后端下为 0。
            """

            # 输出本身由 runner/writer 完成。deferred 后端下 local_to_global
            # 为 None、emitted_frames 为 0（重放阶段不再回调）。
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
        run_clustering(
            artifacts,
            clusterer,
            writer,
            chunk_hook=chunk_log_hook,
            refiner=refiner,
        )
        return raw_rttm_path


def run_diarization(
    config: ChunkPipelineConfig,
    audio_path: str,
    exp_dir: Path,
    model_path: Optional[str] = None,
) -> str:
    """单条音频的 diarization 管线：extract + cluster + RTTM 输出。

    供 diarization.app（独立 CLI）复用；批量场景请直接用
    DiarizationRunner 以复用已加载的模型。

    Args:
        config: 管线配置。
        audio_path: 音频文件路径。
        exp_dir: 本次 run 的输出目录。
        model_path: 显式模型路径覆盖。

    Returns:
        raw RTTM 路径。
    """

    return DiarizationRunner(config, model_path).process_file(audio_path, exp_dir)


# ======================================================================
# DER 评估：refined RTTM 目录 + 参考 RTTM → 逐文件 + 全局 DER
# ======================================================================


def _read_summary_values(path: Path) -> dict[str, str]:
    """读取 summary 文件的 key=value 行（不含 '=' 的行忽略）。"""
    values = {}
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                values[key] = value
    return values


def run_der_evaluation(
    exp_dir: Path,
    ref_rttm: str,
    run_name: str,
    der_verbose: bool,
    results_file: Path,
    collar: float = 0.0,
) -> int:
    """对 exp_dir/rttm 下的 refined RTTM 计算 DER（逐文件 + 拼接全局）。

    子进程调用 tools/compute_der.py，结果回写 der.log / der_summary.txt /
    results.txt；无 refined RTTM 时记 SKIPPED 并直接返回 0。

    Args:
        exp_dir: 本次 run 的输出目录（取 exp_dir/rttm 下的 *.refined.rttm）。
        ref_rttm: 参考 RTTM 目录。
        run_name: 本次运行名（用于日志与汇总记录）。
        der_verbose: 是否透传 --verbose 给 compute_der。
        results_file: 追加写入 DER 汇总的文件。
        collar: md-eval 的宽容边界（秒），默认 0.0（最严格口径）。

    Returns:
        退出码：0 成功或无可评文件；非 0 为 compute_der 子进程失败。
    """

    rttm_count = len(list((exp_dir / "rttm").glob("*.refined.rttm")))
    der_log = exp_dir / "logs" / "der.log"
    der_summary = exp_dir / "der_summary.txt"
    with open(der_log, "w", encoding="utf-8") as log:
        if rttm_count == 0:
            message = f"{run_name} -> DER: SKIPPED | reason: no refined RTTM found"
            print(message)
            log.write(message + "\n")
            with open(results_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(message + "\n")
            return 0
        header = f"========== Computing DER for {run_name} (collar={collar}s) =========="
        print(header)
        log.write(header + "\n")
        der_cmd = [
            sys.executable,
            # 锚定仓库根：库式调用时 cwd 不一定是仓库根。
            str(BASE_DIR / "tools" / "compute_der.py"),
            "--sys",
            str(exp_dir / "rttm"),
            "--ref",
            ref_rttm,
            "--summary-file",
            str(der_summary),
            "--collar",
            str(collar),
            "--sys-suffix",
            ".refined.rttm",
            "--ref-suffix",
            ".rttm",
        ]
        if der_verbose:
            der_cmd.append("--verbose")
        proc = subprocess.run(der_cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        log.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            return proc.returncode
        if der_summary.is_file():
            values = _read_summary_values(der_summary)
            der = values.get("global_der", "NA")
            files = values.get("global_files", "0")
            print(f"{run_name} -> DER(global): {der}% over {files} file(s) (collar={collar}s)")
            with open(results_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(
                    f"{run_name} -> DER(global): {der}% | files={files} | collar={collar}\n"
                )
    return 0


__all__ = [
    "env_flag",
    "add_common_args",
    "check_input_paths",
    "ProcessResult",
    "RunDirs",
    "process",
    "resolve_run_dirs",
    "DiarizationRunner",
    "run_diarization",
    "run_der_evaluation",
]
