"""通用工具：日志、路径、数值。

轻量子模块在包级别 re-export；audio（torchaudio）与 device（torch）
为重依赖，按需 `from common.utils.device import ...` 直接引用子模块。
"""

from .log import log_structured, setup_logger
from .numeric import l2_normalize
from .paths import collect_audio_paths, ensure_parent_dir

__all__ = [
    "log_structured",
    "setup_logger",
    "l2_normalize",
    "collect_audio_paths",
    "ensure_parent_dir",
]
