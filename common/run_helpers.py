"""run.py / test_der.py 共用的编排辅助。

两脚本共有的部分统一收敛到这里：环境变量布尔缺省、公共 CLI 参数、
diarization 管线子进程命令组装、results.txt 按 run 追加的记录头。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def env_flag(name: str, default: bool) -> bool:
    """环境变量作为布尔缺省值：仅 "1" 为真。"""

    raw = os.environ.get(name)
    return default if raw is None else raw == "1"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """run.py 与 test_der.py 共有的 CLI 参数（缺省值取同名环境变量）。"""

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


def build_pipeline_cmd(
    args: argparse.Namespace, audio: object, exp_dir: Path, config: object
) -> list[str]:
    """组装 diarization 管线子进程命令（两脚本共用）。"""

    cmd = [
        sys.executable,
        "-m",
        "diarization.app",
        "--wav",
        str(audio),
        "--output_dir",
        str(exp_dir),
        "--config",
        str(config),
    ]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    if args.hf_cache_dir:
        cmd += ["--hf_cache_dir", args.hf_cache_dir]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    if args.debug:
        cmd += ["--debug", "--verbose"]
    if args.show_rttm:
        cmd += ["--show_rttm"]
    return cmd


def append_run_header(
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


__all__ = [
    "env_flag",
    "add_common_args",
    "build_pipeline_cmd",
    "append_run_header",
]
