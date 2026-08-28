"""ASR 跟随进程的协作编排：前置契约校验 + 子进程命令组装。

stage3（ASR 转写）以独立 follow 进程消费 stage2 产出的 segments manifest，
这里集中放这对跨进程协作关系本身的逻辑；进程哨兵等待/收尾等通用机制在
common.utils.run_helpers。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def validate_follow_requirements(config_path: Path) -> None:
    """校验 ASR 跟随（python -m asr.app --follow）对管线配置的前置要求。

    需要 config 里开启分段导出且使用 streaming 后端；
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


def build_asr_follow_cmd(
    exp_dir: Path,
    asr_config: str,
    done_file: Path,
    ready_file: Path,
    verbose: bool = False,
) -> list[str]:
    """组装 ASR follow 子进程命令（消费 exp_dir/segments，产出 transcripts）。"""

    cmd = [
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
        str(asr_config),
        "--follow",
        "--done_file",
        str(done_file),
        "--ready_file",
        str(ready_file),
    ]
    if verbose:
        cmd.append("--verbose")
    return cmd


__all__ = ["validate_follow_requirements", "build_asr_follow_cmd"]
