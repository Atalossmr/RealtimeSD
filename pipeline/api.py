"""一次完整运行的公开 API：`process`（单文件 stage1-3）与 `process_batch`（批量）。

把 run.py / test_der.py 的编排下沉为库式调用：

- `process`：单条音频的完整管线——输出目录准备、配置构建（可无 argparse）、
  ASR follow 子进程（stage>=3）与哨兵协调、diarization 执行、结果汇总。
  viewer 不在此封装：调用方经 `on_asr_ready` 钩子在 ASR 就绪后自行启动；
- `process_batch`：多条音频共享已加载模型的批量运行（DER 评估场景），
  单文件失败不中断整批；不启动 ASR。

CLI 入口（run.py / test_der.py）只负责 argparse 与环境变量缺省，合并出
`ChunkPipelineConfig` 后调用本模块。
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from common.config import ChunkPipelineConfig, load_config_and_runtime
from common.utils import setup_logger

from .follow import build_asr_follow_cmd, validate_follow_requirements
from .run_helpers import (
    append_run_header,
    finish_follow_process,
    wait_process_ready,
)
from .runner import DiarizationRunner, run_diarization

logger = logging.getLogger(__name__)

# ASR 模型加载（含可能的下载）等待上限。
_DEFAULT_READY_TIMEOUT = 3600.0


@dataclass
class RunDirs:
    """一次运行的全部路径约定（{output_root}/{test_name}/{run_name}）。"""

    basic_dir: Path
    exp_dir: Path
    results_file: Path
    logs_dir: Path
    done_file: Path
    ready_file: Path


def resolve_run_dirs(
    output_root: str | Path, test_name: str, run_name: str
) -> RunDirs:
    """纯路径计算（无副作用）；目录清理/创建由 process / process_batch 负责。"""

    basic_dir = Path(output_root) / test_name
    exp_dir = basic_dir / run_name
    return RunDirs(
        basic_dir=basic_dir,
        exp_dir=exp_dir,
        results_file=basic_dir / "results.txt",
        logs_dir=exp_dir / "logs",
        done_file=exp_dir / ".diarization_done",
        ready_file=exp_dir / ".asr_ready",
    )


def _prepare_run_dirs(dirs: RunDirs) -> None:
    """只清理本次 run 自己的输出目录，同 output_root 下的其他历史 run 保留。"""

    shutil.rmtree(dirs.exp_dir, ignore_errors=True)
    dirs.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessResult:
    """`process` 的运行结果。"""

    returncode: int
    exp_dir: Path
    raw_rttm: Optional[str]
    rttm_count: int
    transcript_count: int  # stage<3 时为 0


@dataclass
class BatchResult:
    """`process_batch` 的运行结果。"""

    exp_dir: Path
    results_file: Path
    logs_dir: Path
    failed: list[str] = field(default_factory=list)
    rttm_count: int = 0


def process(
    audio: str | Path,
    *,
    stage: int = 3,
    config: Optional[ChunkPipelineConfig] = None,
    config_path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, object]] = None,
    output_root: str | Path = "./exp",
    run_name: str = "default",
    test_name: str = "common",
    model_path: Optional[str] = None,
    asr_config: str = "./config/asr.yaml",
    asr_ready_timeout: float = _DEFAULT_READY_TIMEOUT,
    verbose: bool = False,
    on_asr_ready: Optional[Callable[[], None]] = None,
    header_fields: Optional[Mapping[str, object]] = None,
) -> ProcessResult:
    """单条音频的完整管线：stage1 diarization → +stage2 separation → +stage3 ASR。

    `stage` 为累积式：1=只 diarization（RTTM）；2=+separation 分段音频；
    3=+ASR 转写（follow 子进程，先就绪再开跑）。

    config 缺省时经 `load_config(config_path, overrides)` 无 argparse 构建
    （config_path 缺省为仓库自带 config/config.yaml）；CLI 入口应自行合并后
    传入 config（config_path 此时仅用于运行记录）。

    `on_asr_ready` 在 ASR 就绪后、管线启动前调用（stage<3 时同样在管线启动前
    调用一次），供调用方启动 viewer 等伴随服务；挂起等待与 viewer 收尾由
    调用方负责。

    管线异常不跳过收尾：记录后照常放行 ASR 收尾落盘，rc=1；Ctrl+C 置 rc=130。
    """

    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1/2/3, got {stage!r}")
    audio = str(audio)
    if config is None:
        config, runtime = load_config_and_runtime(config_path, dict(overrides or {}))
        if model_path is None:
            # YAML 的 model_path 是运行时键（CLI 下为 argparse 参数）：
            # 显式 model_path 参数优先，YAML 其次。
            model_path = runtime.get("model_path")
    if stage >= 3:
        validate_follow_requirements(config)

    dirs = resolve_run_dirs(output_root, test_name, run_name)
    _prepare_run_dirs(dirs)

    # results.txt 按 run 追加：同一 output_root 下历史 run 的汇总记录保留。
    append_run_header(
        dirs.results_file,
        run_name,
        "Online Speaker Diarization pipeline",
        dict(
            header_fields
            or {
                "audio_input": audio,
                "config_path": config_path,
                "model_path_override": model_path,
                "with_asr": stage >= 3,
                "stage": stage,
            }
        ),
    )

    asr_cmd: list[str] = []
    if stage >= 3:
        asr_cmd = build_asr_follow_cmd(
            dirs.exp_dir,
            str(asr_config),
            dirs.done_file,
            dirs.ready_file,
            verbose=verbose or config.debug,
        )
    with open(dirs.logs_dir / "command.log", "w", encoding="utf-8") as log:
        log.write(f"Audio: {audio}\n")
        log.write(f"Config: {config_path}\n")
        log.write(
            "Pipeline: in-process pipeline.run_diarization"
            f" (device={config.device}, model_type={config.model_type},"
            f" segmentation_model={config.segmentation_model},"
            f" backend={config.clustering_backend})\n"
        )
        if asr_cmd:
            log.write(f"ASR Command: {shlex.join(asr_cmd)}\n")

    # 管线日志写入 logs/run.log（控制台只保留编排输出）；
    # verbose/debug 时同步提升日志级别。
    setup_logger(verbose or config.debug, str(dirs.logs_dir / "run.log"))

    asr_proc: Optional[subprocess.Popen] = None
    rc = 0
    raw_rttm: Optional[str] = None
    try:
        # 1. ASR 先启动并等模型就绪（仅 stage >= 3）。
        if stage >= 3:
            print(f"Starting ASR follow process: {shlex.join(asr_cmd)}")
            asr_proc = subprocess.Popen(asr_cmd)
            print("Waiting for ASR model ready ...")
            wait_process_ready(asr_proc, dirs.ready_file, asr_ready_timeout)
            print("ASR model ready.")

        # 2. ASR 就绪（或 stage<3 直接）后放行调用方的伴随服务（如 viewer）。
        if on_asr_ready is not None:
            on_asr_ready()

        # 3. 本进程内跑 diarization 管线（stage>=2 启用 separation）。
        print(f"Starting diarization pipeline (in-process): {audio}")
        try:
            raw_rttm = run_diarization(
                config,
                audio,
                dirs.exp_dir,
                model_path=model_path,
                with_separation=stage >= 2,
            )
        except Exception:
            # 管线失败不跳过收尾：记录后继续走 ASR 的 finally 清理。
            logger.exception("Diarization pipeline failed")
            print(f"ERROR: 管线执行失败，见 {dirs.logs_dir}/run.log", file=sys.stderr)
            rc = 1
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down ...", file=sys.stderr)
        rc = 130
    finally:
        # 落 done 哨兵放行 ASR 收尾并等其落盘（正常/异常/中断路径统一）。
        if asr_proc is not None:
            finish_follow_process(asr_proc, dirs.done_file)

    # ---- 汇总 ----
    rttm_count = len(list((dirs.exp_dir / "rttm").glob("*.refined.rttm")))
    transcript_count = (
        len(list((dirs.exp_dir / "transcripts").glob("*.transcript.jsonl")))
        if stage >= 3
        else 0
    )
    with open(dirs.results_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(
            f"{run_name} -> refined_rttm_files={rttm_count} | config={config_path}\n"
        )
        if stage >= 3:
            file_obj.write(
                f"{run_name} -> transcript_files={transcript_count}"
                f" | config={config_path}\n"
            )
    print(f"Result: {run_name} -> refined_rttm_files={rttm_count}")
    print(f"Pipeline log: {dirs.logs_dir}/run.log")
    if stage >= 3:
        print(f"ASR log: {dirs.logs_dir}/transcribe.log")

    return ProcessResult(
        returncode=rc,
        exp_dir=dirs.exp_dir,
        raw_rttm=raw_rttm,
        rttm_count=rttm_count,
        transcript_count=transcript_count,
    )


def process_batch(
    audio_paths: list[str],
    *,
    config: ChunkPipelineConfig,
    config_path: Optional[str | Path] = None,
    output_root: str | Path = "./exp",
    run_name: str = "default",
    test_name: str = "der_test",
    model_path: Optional[str] = None,
    with_separation: bool = False,
    verbose: bool = False,
    header_fields: Optional[Mapping[str, object]] = None,
) -> BatchResult:
    """批量运行 diarization 管线：模型只加载一次，单文件失败不中断整批。

    全部音频的 RTTM 写入同一输出目录；不启动 ASR / viewer（评估场景）。
    """

    dirs = resolve_run_dirs(output_root, test_name, run_name)
    _prepare_run_dirs(dirs)

    append_run_header(
        dirs.results_file,
        run_name,
        "DER test",
        dict(
            header_fields
            or {
                "num_audio_files": len(audio_paths),
                "with_separation": with_separation,
                "model_path_override": model_path,
            }
        ),
    )

    with open(dirs.logs_dir / "command.log", "w", encoding="utf-8") as log:
        log.write(f"Audio: {len(audio_paths)} file(s)\n")
        for path in audio_paths:
            log.write(f"  - {path}\n")
        log.write(f"Config: {config_path} (with_separation={with_separation})\n")
        log.write(
            "Pipeline: in-process pipeline.DiarizationRunner"
            f" (device={config.device}, model_type={config.model_type},"
            f" segmentation_model={config.segmentation_model},"
            f" backend={config.clustering_backend})\n"
        )

    setup_logger(verbose or config.debug, str(dirs.logs_dir / "run.log"))

    runner = DiarizationRunner(config, model_path, with_separation=with_separation)
    failed: list[str] = []
    for audio_path in audio_paths:
        logger.info("Processing %s", audio_path)
        print(f"Processing {audio_path} ...")
        try:
            runner.process_file(audio_path, dirs.exp_dir)
        except Exception:
            failed.append(audio_path)
            logger.exception("Diarization pipeline failed for %s", audio_path)
            print(
                f"ERROR: 管线执行失败: {audio_path}，见 {dirs.logs_dir}/run.log",
                file=sys.stderr,
            )
            continue

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

    return BatchResult(
        exp_dir=dirs.exp_dir,
        results_file=dirs.results_file,
        logs_dir=dirs.logs_dir,
        failed=failed,
        rttm_count=rttm_count,
    )


__all__ = [
    "BatchResult",
    "ProcessResult",
    "RunDirs",
    "process",
    "process_batch",
    "resolve_run_dirs",
]
