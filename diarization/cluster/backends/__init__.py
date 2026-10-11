"""聚类后端子包：内置后端与工厂。

新增聚类方法：在本目录新建模块实现 `BaseChunkClusterer`
（见 ../base.py），并在 `build_clusterer` 注册一个分支即可。
"""

from __future__ import annotations

from diarization.config import ChunkPipelineConfig

from ..base import BaseChunkClusterer
from .ahc import AHCChunkClusterer
from .fastdiar import FastdiarChunkClusterer
from .streaming import StreamingChunkClusterer


def build_clusterer(config: ChunkPipelineConfig) -> BaseChunkClusterer:
    """按 YAML 的 clustering_backend 构造聚类后端。

    Args:
        config: 管线配置，读取其 clustering_backend 字段选择后端。

    Returns:
        对应后端的 BaseChunkClusterer 实例。

    Raises:
        ValueError: clustering_backend 不是 'streaming'、'ahc' 或 'fastdiar'。
    """

    backend = str(config.clustering_backend)
    if backend == "streaming":
        return StreamingChunkClusterer(config)
    if backend == "ahc":
        return AHCChunkClusterer(config)
    if backend == "fastdiar":
        return FastdiarChunkClusterer(config)
    raise ValueError(
        f"Unknown clustering_backend: {backend!r} "
        "(expected 'streaming', 'ahc' or 'fastdiar')"
    )


__all__ = [
    "build_clusterer",
    "StreamingChunkClusterer",
    "AHCChunkClusterer",
    "FastdiarChunkClusterer",
]
