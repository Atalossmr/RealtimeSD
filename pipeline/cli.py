"""入口脚本（run.py / test_der.py）共用的 CLI 辅助。

环境变量布尔缺省、公共 CLI 参数、启动前输入检查。只依赖标准库，
不含任何业务 import；管线本体的编排逻辑在 api.py / follow.py。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def env_flag(name: str, default: bool) -> bool:
    """环境变量作为布尔缺省值：仅 "1" 为真。"""

    raw = os.environ.get(name)
    return default if raw is None else raw == "1"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """run.py 与 test_der.py 共有的 CLI 参数（缺省值取对应环境变量）。"""

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


def check_input_paths(audio: Path, config_path: Path) -> None:
    """启动前 fail-fast：音频/配置不存在时尽早报错，避免模型加载完才发现。"""

    if not audio.exists():
        raise SystemExit(f"ERROR: 音频输入不存在: {audio}")
    if audio.is_dir():
        raise SystemExit(f"ERROR: 音频输入只接受单个文件，不支持目录: {audio}")
    if not config_path.exists():
        raise SystemExit(f"ERROR: 配置文件不存在: {config_path}")


__all__ = [
    "env_flag",
    "add_common_args",
    "check_input_paths",
]
