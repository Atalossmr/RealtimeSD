"""聚类后端子包：内置后端与工厂。

新增聚类方法：在本目录新建模块实现 `BaseChunkAssigner`
（见 ../base.py），并在 `build_assigner` 注册一个分支即可。
"""

from __future__ import annotations

from ...config import ChunkPipelineConfig
from ..base import BaseChunkAssigner
from .ahc import AHCChunkAssigner
from .streaming import ChunkSpeakerClusterer


def build_assigner(config: ChunkPipelineConfig) -> BaseChunkAssigner:
    """按 YAML 的 clustering_backend 构造聚类后端。"""

    backend = str(config.clustering_backend)
    if backend == "streaming":
        return ChunkSpeakerClusterer(config)
    if backend == "ahc":
        return AHCChunkAssigner(config)
    raise ValueError(
        f"Unknown clustering_backend: {backend!r} (expected 'streaming' or 'ahc')"
    )


__all__ = [
    "build_assigner",
    "ChunkSpeakerClusterer",
    "AHCChunkAssigner",
]
