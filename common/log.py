"""日志初始化（asr / diarization 共用）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(verbose: bool, run_log_path: Optional[str] = None) -> None:
    """初始化日志系统。

    当前 CLI 约定会把运行日志写到输出目录的 `logs/` 子目录（如 `logs/run.log`），
    默认不再把常规日志输出到控制台；控制台只保留脚本自身输出，以及可选的 RTTM 流式输出。
    """

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    if run_log_path:
        Path(run_log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)


__all__ = ["setup_logger"]
