"""聚类后端（assigner）接口与实现。

设计要点：

- embedding 提取（extract/）与聚类分配（assigner）解耦，
  后端通过 `build_assigner(config)` 按 YAML 配置插拔；
- 流式后端（deferred=False）：`assign_chunk` 立即返回最终 local->global 映射，
  调用方逐 chunk 写出 RTTM；
- 离线后端（deferred=True）：`assign_chunk` 只缓冲 observations，
  `finalize()` 统一聚类并返回逐 chunk 的映射，调用方在音频结束后
  用同一 writer 逻辑重放帧级输出。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import ChunkPipelineConfig
from ..schema import ChunkDebugInfo, ChunkObservation


logger = logging.getLogger(__name__)


def _empty_debug_info() -> ChunkDebugInfo:
    return {
        "num_centroids_before": 0,
        "num_centroids_after": 0,
        "local_assignments": [],
        "new_speakers": [],
        "updated_speakers": [],
        "skipped_updates": [],
        "global_speakers": [],
    }


class BaseChunkAssigner(ABC):
    """chunk 级 local->global 分配后端接口。"""

    # False：assign_chunk 立即返回最终 id；True：缓冲到 finalize 统一分配。
    deferred: bool = False
    # 输出 RTTM 文件名后缀：<stem>.<output_tag>.rttm。
    output_tag: str = "streaming"

    @abstractmethod
    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[Optional[dict[int, int]], ChunkDebugInfo]:
        """处理一个 chunk 的 observations。

        流式后端返回 (local_to_global, debug_info)；
        离线后端返回 (None, debug_info)，映射由 finalize() 统一给出。
        """

    def finalize(self) -> list[dict[int, int]]:
        """离线后端在音频结束后统一分配，返回逐 chunk 的 local->global 列表。"""

        raise NotImplementedError(
            f"{type(self).__name__} 不是 deferred 后端，无需 finalize"
        )


class AHCChunkAssigner(BaseChunkAssigner):
    """离线 AHC 后端：缓冲全部 embedding，音频结束后一次层次聚类。"""

    deferred = True
    output_tag = "ahc"

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config
        # 逐 chunk 缓冲带 embedding 的 observations。
        self._buffered: list[list[ChunkObservation]] = []

    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[Optional[dict[int, int]], ChunkDebugInfo]:
        embedded = [obs for obs in observations if obs.embedding is not None]
        self._buffered.append(embedded)
        return None, _empty_debug_info()

    def finalize(self) -> list[dict[int, int]]:
        from sklearn.cluster import AgglomerativeClustering

        observations = [obs for chunk in self._buffered for obs in chunk]
        if not observations:
            return [{} for _ in self._buffered]

        embeddings = np.stack([obs.embedding for obs in observations])
        model = AgglomerativeClustering(
            metric="cosine",
            linkage=self.config.ahc_linkage,
            distance_threshold=1.0 - self.config.ahc_similarity_threshold,
            n_clusters=None,
        )
        labels = model.fit_predict(embeddings)
        num_clusters = int(len(set(labels.tolist())))
        logger.info(
            "[ahc] observations=%d clusters=%d (similarity_threshold=%.3f, linkage=%s)",
            len(observations),
            num_clusters,
            self.config.ahc_similarity_threshold,
            self.config.ahc_linkage,
        )

        assignments: list[dict[int, int]] = []
        cursor = 0
        for chunk in self._buffered:
            local_to_global: dict[int, int] = {}
            for obs in chunk:
                local_to_global[int(obs.local_idx)] = int(labels[cursor])
                cursor += 1
            assignments.append(local_to_global)
        return assignments


def build_assigner(config: ChunkPipelineConfig) -> BaseChunkAssigner:
    """按配置构造聚类后端。"""

    backend = str(config.clustering_backend)
    if backend == "streaming":
        # lazy import：clusterer 继承本模块的基类，模块级导入会循环依赖。
        from .clusterer import ChunkSpeakerClusterer

        return ChunkSpeakerClusterer(config)
    if backend == "ahc":
        return AHCChunkAssigner(config)
    raise ValueError(
        f"Unknown clustering_backend: {backend!r} (expected 'streaming' or 'ahc')"
    )


__all__ = ["BaseChunkAssigner", "AHCChunkAssigner", "build_assigner"]
