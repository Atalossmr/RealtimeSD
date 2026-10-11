"""聚类与输出子模块：chunk artifacts -> clusterer 分配 -> RTTM。

- `base.py`：后端接口 BaseChunkClusterer；
- `backends/`：内置后端（streaming / ahc / fastdiar）与工厂 build_clusterer；
- `runner.py`：聚类消费循环 run_clustering；
- `rttm_writer.py`：零重写 RTTM 写出；
- `post_process.py`：refined 级输出与小样本簇强制合并后处理；
- `app.py`：聚类阶段 CLI。
"""

from .backends import (
    AHCChunkClusterer,
    FastdiarChunkClusterer,
    StreamingChunkClusterer,
    build_clusterer,
)
from .base import BaseChunkClusterer
from .rttm_writer import AppendOnlyRTTMWriter
from .runner import run_clustering

__all__ = [
    "BaseChunkClusterer",
    "build_clusterer",
    "StreamingChunkClusterer",
    "AHCChunkClusterer",
    "FastdiarChunkClusterer",
    "AppendOnlyRTTMWriter",
    "run_clustering",
]
