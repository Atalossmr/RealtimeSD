"""ASR 模块的日志初始化（实现已收敛到 common.log，此处 re-export 保持旧导入路径）。"""

from __future__ import annotations

from common.log import setup_logger

__all__ = ["setup_logger"]
