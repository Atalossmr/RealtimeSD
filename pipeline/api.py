"""一次完整运行的公开 API：`process(audio=...)` 统一支持单文件与批量。

`audio` 传单个路径时处理一条音频；传路径列表时按顺序在同一进程内
逐个处理（模型只加载一次），单个文件失败不中断整体，失败路径记入
`ProcessResult.failed`。stage>=3 时启动一个常驻 ASR follow 子进程，
与 diarization 时间重叠。Ctrl+C 尽快停止并返回 `returncode=130`
（CLI `--wait` 语义依赖该值）。

CLI 入口（run.py / test_der.py）只负责 argparse 与环境变量缺省，合并出
`ChunkPipelineConfig` 后调用本模块。viewer 不在此封装：调用方经
`on_asr_ready` 钩子在 ASR 就绪后自行启动。
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

from common.config import ChunkPipelineConfig, load_config_and_runtime
from common.constants import BASE_DIR
from common.utils import setup_logger

from .follow import (
    build_asr_follow_cmd,
    finish_follow_process,
    validate_follow_requirements,
    wait_process_ready,
)
from .runner import DiarizationRunner

logger = logging.getLogger(__name__)


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
    """纯路径计算（无副作用）；目录清理/创建由 process 负责。"""

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


def _append_run_header(
    results_file: Path,
    run_name: str,
    title: str,
    fields: dict[str, object],
) -> None:
    """向 results.txt 追加一个 run 的记录头（按 run 追加，历史汇总保留）。

    fields 中值为 None 的键不写出。
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
    """`process` 的运行结果。相对路径字段均相对于 `exp_dir`。"""

    returncode: int
    exp_dir: Path
    results_file: Path
    logs_dir: Path
    failed: list[str] = field(default_factory=list)  # 处理失败的音频路径
    raw_rttms: list[str] = field(default_factory=list)
    rttm_count: int = 0
    transcript_count: int = 0  # stage<3 时为 0


def process(
    audio: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    stage: int = 3,
    config: Optional[ChunkPipelineConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, object]] = None,
    output_root: Union[str, Path] = "./exp",
    run_name: str = "default",
    test_name: str = "common",
    model_path: Optional[str] = None,
    asr_config: str = "./config/asr.yaml",
    verbose: bool = False,
    on_asr_ready: Optional[Callable[[], None]] = None,
    header_fields: Optional[Mapping[str, object]] = None,
    header_title: str = "Online Speaker Diarization pipeline",
) -> ProcessResult:
    """完整管线：stage1 diarization → +stage2 separation → +stage3 ASR。

    `stage` 为累积式：1=只 diarization（RTTM）；2=+separation 分段音频；
    3=+ASR 转写（follow 子进程，先就绪再开跑）。

    `audio` 为单个路径或路径列表。批量时全部音频的产物写入同一输出目录
    （uri 区分文件名），模型只加载一次，单文件失败记入 `failed` 并继续，
    有失败时 returncode=1。

    config 缺省时经 `load_config_and_runtime(config_path, overrides)` 无
    argparse 构建（config_path 缺省为仓库自带 config/config.yaml）；CLI
    入口应自行合并后传入 config（config_path 此时仅用于运行记录）。

    `on_asr_ready` 在 ASR 就绪后、管线启动前调用（stage<3 时同样在管线
    启动前调用一次），供调用方启动 viewer 等伴随服务；挂起等待与 viewer
    收尾由调用方负责。

    ASR 协调：就绪等待无超时上限（模型加载耗时不可预估，早退会立即
    报错，卡住时由用户 Ctrl+C 中断）；收尾排空前有兜底上限（默认
    300s，中断路径 30s），超时按 terminate→kill 升级并打 WARNING，
    见 follow.finish_follow_process。

    管线异常不跳过收尾：记录后照常放行 ASR 收尾落盘；Ctrl+C 置 rc=130。
    """

    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1/2/3, got {stage!r}")
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
    if stage >= 3:
        validate_follow_requirements(config)

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
                "with_asr": stage >= 3,
                "stage": stage,
            }
        ),
    )

    asr_cmd: list[str] = []
    if stage >= 3:
        # 相对路径锚定仓库根（与 eval.py 的 BASE_DIR 口径一致）：
        # follow 子进程 cwd 固定为仓库根，调用方 cwd 不影响解析。
        asr_config_path = Path(asr_config)
        if not asr_config_path.is_absolute():
            asr_config_path = BASE_DIR / asr_config_path
        asr_cmd = build_asr_follow_cmd(
            dirs.exp_dir,
            str(asr_config_path),
            dirs.done_file,
            dirs.ready_file,
            verbose=verbose or config.debug,
        )
    with open(dirs.logs_dir / "command.log", "w", encoding="utf-8") as log:
        log.write(f"Audio: {len(audio_paths)} file(s)\n")
        for path in audio_paths:
            log.write(f"  - {path}\n")
        log.write(f"Config: {config_path} (stage={stage})\n")
        log.write(
            "Pipeline: in-process pipeline.DiarizationRunner"
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
    failed: list[str] = []
    raw_rttms: list[str] = []
    try:
        # 1. ASR 先启动并等模型就绪（仅 stage >= 3）：加载耗时与首文件的
        #    diarization 重叠，且避免段积压到最后才转写。
        if stage >= 3:
            print(f"Starting ASR follow process: {shlex.join(asr_cmd)}")
            # cwd 锚定仓库根：python -m asr.app 依赖仓库根在 sys.path，
            # 调用方 cwd 不是仓库根时子进程会立即 "No module named asr"。
            asr_proc = subprocess.Popen(asr_cmd, cwd=BASE_DIR)
            print("Waiting for ASR model ready ...")
            wait_process_ready(asr_proc, dirs.ready_file)
            print("ASR model ready.")

        # 2. ASR 就绪（或 stage<3 直接）后放行调用方的伴随服务（如 viewer）。
        if on_asr_ready is not None:
            on_asr_ready()

        # 3. 本进程内逐文件跑 diarization 管线（stage>=2 启用 separation）。
        # 已知风险（暂不修，留待后续去重/fail-fast）：uri 取
        # Path(audio_path).stem，同批两个不同目录的同名文件（如
        # a/01.wav 与 b/01.wav）会共用 uri——raw RTTM 相互覆盖、
        # segments manifest 与 transcript 混合。调用方应避免同批
        # 传入 stem 相同的音频。
        runner = DiarizationRunner(config, model_path, with_separation=stage >= 2)
        for audio_path in audio_paths:
            logger.info("Processing %s", audio_path)
            print(f"Processing {audio_path} ...")
            try:
                raw_rttms.append(runner.process_file(audio_path, dirs.exp_dir))
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
    finally:
        # 落 done 哨兵放行 ASR 收尾并等其落盘（stage>=3）；stage<3 无 ASR，
        # 管线结束即 touch 哨兵，供 viewer 判定 LIVE 状态。
        if asr_proc is not None:
            # 中断路径（rc=130）传更短的排空超时：保证 Ctrl+C 能尽快返回，
            # 且等待期间的二次 Ctrl+C 不会把 ASR 子进程遗弃为孤儿。
            asr_rc = finish_follow_process(
                asr_proc, dirs.done_file, timeout=30.0 if rc == 130 else 300.0
            )
            if asr_rc != 0:
                logger.error(
                    "ASR follow 进程异常退出（rc=%s），transcript 可能不全，见 %s",
                    asr_rc,
                    dirs.logs_dir / "transcribe.log",
                )
                print(
                    f"WARNING: ASR follow 进程异常退出（rc={asr_rc}），"
                    f"transcript 可能不全，见 {dirs.logs_dir}/transcribe.log",
                    file=sys.stderr,
                )
                if rc == 0:
                    rc = 1
        else:
            dirs.done_file.touch()

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
    print(
        f"Result: {run_name} -> refined_rttm_files={rttm_count}"
        f" (failed={len(failed)}/{len(audio_paths)})"
    )
    print(f"Pipeline log: {dirs.logs_dir}/run.log")
    if stage >= 3:
        print(f"ASR log: {dirs.logs_dir}/transcribe.log")

    return ProcessResult(
        returncode=rc,
        exp_dir=dirs.exp_dir,
        results_file=dirs.results_file,
        logs_dir=dirs.logs_dir,
        failed=failed,
        raw_rttms=raw_rttms,
        rttm_count=rttm_count,
        transcript_count=transcript_count,
    )


__all__ = [
    "ProcessResult",
    "RunDirs",
    "process",
    "resolve_run_dirs",
]
