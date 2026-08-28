"""pipeline 层编排辅助：run.py / test_der.py 与本包 api 共用。

两脚本共有的部分统一收敛到这里：环境变量布尔缺省、公共 CLI 参数、
results.txt 按 run 追加的记录头、启动前输入检查、跟随子进程的就绪等待
与收尾。本模块只依赖标准库，不含任何业务 import。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
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


def check_input_paths(audio: Path, config_path: Path) -> None:
    """启动前 fail-fast：音频/配置不存在时尽早报错，避免模型加载完才发现。"""

    if not audio.exists():
        raise SystemExit(f"ERROR: 音频输入不存在: {audio}")
    if audio.is_dir():
        raise SystemExit(f"ERROR: 音频输入只接受单个文件，不支持目录: {audio}")
    if not config_path.exists():
        raise SystemExit(f"ERROR: 配置文件不存在: {config_path}")


def wait_process_ready(
    proc: subprocess.Popen, ready_file: Path, timeout: float
) -> None:
    """轮询等子进程的就绪哨兵；进程早退或超时则报错退出。"""

    deadline = time.monotonic() + timeout
    while not ready_file.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"子进程在就绪前退出（rc={proc.returncode}）")
        if time.monotonic() > deadline:
            raise TimeoutError(f"等待子进程就绪超时（{timeout}s）")
        time.sleep(1.0)


def finish_follow_process(proc: subprocess.Popen, done_file: Path) -> None:
    """落 done 哨兵放行跟随进程收尾，等其落盘；超时先 SIGTERM，仍不退出再 SIGKILL。"""

    done_file.touch()
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        # 收尾超时：先 SIGTERM，仍不退出再 SIGKILL，避免主进程永久挂起。
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


__all__ = [
    "env_flag",
    "add_common_args",
    "append_run_header",
    "check_input_paths",
    "wait_process_ready",
    "finish_follow_process",
]
