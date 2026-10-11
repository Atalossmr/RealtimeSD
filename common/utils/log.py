"""日志初始化与结构化日志（diarization / separation 共用）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional


def setup_logger(verbose: bool, run_log_path: Optional[str] = None) -> None:
    """初始化日志系统。

    清空 root logger 的已有 handler，按 verbose 设定级别；给定 run_log_path 时
    以覆盖模式写文件日志，不向控制台输出常规日志。当前 CLI 约定把运行日志写
    到输出目录的 `logs/` 子目录（如 `logs/run.log`），控制台只保留脚本自身输
    出，以及可选的 RTTM 流式输出。

    Args:
        verbose: 为 True 时使用 DEBUG 级别，否则 INFO。
        run_log_path: 运行日志文件路径；为 None 时不写文件日志。
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


def log_structured(
    logger: logging.Logger, level: int, prefix: str, title: str, payload: object
) -> None:
    """输出结构化日志事件：`<prefix> <title>:` 后跟 indent=2 的 JSON payload。

    日志行格式与 tools/ 下的分析脚本（如 analyze_log.py）约定一致，
    改动格式需同步消费者。

    Args:
        logger: 目标 logger。
        level: 日志级别。
        prefix: 事件前缀。
        title: 事件标题。
        payload: 可被 json.dumps 序列化的事件内容。
    """

    logger.log(
        level,
        "%s %s:\n%s",
        prefix,
        title,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


__all__ = ["setup_logger", "log_structured"]
