"""一次完整运行的公开 API：`process(audio=...)` 统一支持单文件与批量。

`audio` 传单个路径时处理一条音频；传路径列表时按顺序在同一进程内
逐个处理（模型只加载一次），单个文件失败不中断整体，失败路径记入
`ProcessResult.failed`。管线就是「音频 → RTTM」。Ctrl+C 尽快停止并返回
`returncode=130`。

CLI 入口（run.py / test_der.py）只负责 argparse 与环境变量缺省，合并出
`ChunkPipelineConfig` 后调用本模块。
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

from common.config import ChunkPipelineConfig, load_config_and_runtime
from common.utils import setup_logger

from .runner import DiarizationRunner

logger = logging.getLogger(__name__)


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
            "Pipeline: in-process pipeline.DiarizationRunner"
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


__all__ = [
    "ProcessResult",
    "RunDirs",
    "process",
    "resolve_run_dirs",
]
