"""diarization 中间产物存取：chunk artifacts npz。

通用工具（音频、设备、数值、路径）已上提至 common.utils；
本子包只保留 diarization 自己的中间文件格式（chunk_io）。
"""

from .chunk_io import load_chunks, save_chunks

__all__ = [
    "load_chunks",
    "save_chunks",
]
