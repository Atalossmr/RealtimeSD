"""聚类与输出子模块：chunk artifacts -> assigner 分配 -> RTTM。

- `base.py`：后端接口 BaseChunkAssigner；
- `backends/`：内置后端（streaming / ahc）与工厂 build_assigner；
- `runner.py`：聚类消费循环 run_clustering；
- `rttm_writer.py`：零重写 RTTM 写出；
- `app.py`：聚类阶段 CLI。
"""

from .backends import AHCChunkAssigner, ChunkSpeakerClusterer, build_assigner
from .base import BaseChunkAssigner
from .rttm_writer import AppendOnlyRTTMWriter
from .runner import run_clustering

__all__ = [
    "BaseChunkAssigner",
    "build_assigner",
    "ChunkSpeakerClusterer",
    "AHCChunkAssigner",
    "AppendOnlyRTTMWriter",
    "run_clustering",
]
