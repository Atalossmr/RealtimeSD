"""离线 AHC 聚类后端：缓冲全部 embedding，音频结束后一次层次聚类。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...config import ChunkPipelineConfig
from ...schema import ChunkDebugInfo, ChunkObservation
from ..base import BaseChunkAssigner


logger = logging.getLogger(__name__)


def _empty_debug_info() -> ChunkDebugInfo:
    """离线后端在 assign_chunk 阶段不做判定，返回空的 debug 结构。"""

    return {
        "num_centroids_before": 0,
        "num_centroids_after": 0,
        "local_assignments": [],
        "new_speakers": [],
        "updated_speakers": [],
        "skipped_updates": [],
        "global_speakers": [],
    }


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
        # 离线后端此刻不做任何判定：只按 chunk 顺序缓冲带 embedding 的
        # observations，标签在 finalize 统一给出，因此这里返回 None。
        embedded = [obs for obs in observations if obs.embedding is not None]
        self._buffered.append(embedded)
        return None, _empty_debug_info()

    def finalize(self) -> list[dict[int, int]]:
        # sklearn 延迟导入：streaming 后端运行时不需要它。
        from sklearn.cluster import AgglomerativeClustering

        # 展平为全局 observation 序列，cursor 顺序与下方重建映射时严格一致。
        observations = [obs for chunk in self._buffered for obs in chunk]
        if not observations:
            return [{} for _ in self._buffered]

        embeddings = np.stack([obs.embedding for obs in observations])
        # embedding 已 L2 归一化：cosine 距离 = 1 - 相似度，
        # distance_threshold = 1 - t 即"相似度 ≥ t 才允许并入同一簇"。
        # n_clusters=None + distance_threshold：簇数完全由阈值决定。
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

        # 把全局 label 序列按 chunk 切回逐 chunk 的 local->global 映射，
        # 供 runner 按原始顺序重放帧级输出。
        assignments: list[dict[int, int]] = []
        cursor = 0
        for chunk in self._buffered:
            local_to_global: dict[int, int] = {}
            for obs in chunk:
                local_to_global[int(obs.local_idx)] = int(labels[cursor])
                cursor += 1
            assignments.append(local_to_global)
        return assignments


__all__ = ["AHCChunkAssigner"]
