"""通用工具 + diarization 中间产物存取。

轻量子模块在包级别 re-export；audio（torchaudio）与 device（torch）
为重依赖，按需 `from diarization.utils.device import ...` 直接引用子模块。
RTTM 解析（rttm）只依赖标准库，chunk_io 为 diarization 的 chunk
artifacts npz 中间文件格式。
"""

from .chunk_io import load_chunks, save_chunks
from .log import log_structured, setup_logger
from .numeric import l2_normalize
from .paths import collect_audio_paths, ensure_parent_dir

__all__ = [
    "log_structured",
    "setup_logger",
    "l2_normalize",
    "collect_audio_paths",
    "ensure_parent_dir",
    "load_chunks",
    "save_chunks",
]
