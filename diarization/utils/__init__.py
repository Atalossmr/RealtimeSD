"""通用工具子包：日志、设备、数值、音频、路径与 chunk 中间文件存取。

所有成员在包级别 re-export，外部保持 `from ...utils import xxx` 的用法不变。
"""

from .audio import resample_waveform_if_needed
from .chunk_io import load_chunks, save_chunks
from .device import resolve_device
from .log import log_structured, setup_logger
from .numeric import l2_normalize
from .paths import collect_audio_paths, ensure_parent_dir

__all__ = [
    "resample_waveform_if_needed",
    "load_chunks",
    "save_chunks",
    "resolve_device",
    "setup_logger",
    "log_structured",
    "l2_normalize",
    "collect_audio_paths",
    "ensure_parent_dir",
]
