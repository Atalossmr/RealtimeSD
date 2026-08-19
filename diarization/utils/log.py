"""日志工具。setup_logger 实现已收敛到 common.log，此处 re-export 保持旧导入路径。"""

from __future__ import annotations

import json
import logging

from common.log import setup_logger


def log_structured(
    logger: logging.Logger, level: int, prefix: str, title: str, payload: object
) -> None:
    """输出结构化日志事件：`<prefix> <title>:` 后跟 indent=2 的 JSON payload。

    日志行格式与 tools/ 下的分析脚本（_read_json_block 解析器）约定一致，
    改动格式需同步消费者。
    """

    logger.log(
        level,
        "%s %s:\n%s",
        prefix,
        title,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


__all__ = ["setup_logger", "log_structured"]
