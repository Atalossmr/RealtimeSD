"""日志工具。"""

from __future__ import annotations

import logging
from typing import Optional

from .paths import ensure_parent_dir


def setup_logger(verbose: bool, run_log_path: Optional[str] = None) -> None:
    """初始化日志系统。

    当前 CLI 约定会把运行日志强制写到 `output_dir/run.log`，
    默认不再把常规日志输出到控制台；控制台只保留脚本自身输出，以及可选的 RTTM 流式输出。
    """

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    if run_log_path:
        ensure_parent_dir(run_log_path)
        file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)


__all__ = ["setup_logger"]
