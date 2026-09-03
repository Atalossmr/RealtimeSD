"""离线 AHC 聚类后端：缓冲全部 embedding，音频结束后一次层次聚类。

输出与 streaming 后端同为两级：帧级结果写出 raw 级（`*.raw.rttm`），
小样本强制合并等修正在 refined 级（`post_process.RefinedRTTMWriter`）
于 EOF 最终刷新时叠加，生成 `*.refined.rttm` 与 speakers.json sidecar。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from common.config import ChunkPipelineConfig
from common.contracts import ChunkDebugInfo, ChunkObservation

from ..base import BaseChunkClusterer
from ..post_process import cluster_stats


logger = logging.getLogger(__name__)


def _empty_debug_info() -> ChunkDebugInfo:
    """离线后端在 assign_chunk 阶段不做判定，返回空的 debug 结构。

    Returns:
        各字段均为空值（计数为 0、列表为空）的 ChunkDebugInfo。
    """

    return {
        "num_centroids_before": 0,
        "num_centroids_after": 0,
        "local_assignments": [],
        "new_speakers": [],
        "updated_speakers": [],
        "skipped_updates": [],
        "global_speakers": [],
    }


class AHCChunkClusterer(BaseChunkClusterer):
    """离线 AHC 后端：缓冲全部 embedding，音频结束后一次层次聚类。

    assign_chunk 只按 chunk 顺序缓冲带 embedding 的 observations 并返回
    (None, 空 debug)；finalize 用 AgglomerativeClustering（cosine 距离 +
    distance_threshold，簇数完全由阈值决定）统一聚类，返回逐 chunk 的
    local->global 映射供调用方重放写出。

    Attributes:
        centroids: finalize 后填充的簇质心（均值重归一化）。
        counts: finalize 后填充的各簇 observation 数。
        merged_into: 流式 merge 链；ahc 无流式 merge，恒为空。
    """

    deferred = True
    # 与 streaming 后端统一：raw 级一律为 <stem>.raw.rttm。
    output_tag = "raw"

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config
        # 逐 chunk 缓冲带 embedding 的 observations。
        self._buffered: list[list[ChunkObservation]] = []
        # finalize 后填充，供 refined 级（post_process.RefinedRTTMWriter）
        # 读取：簇质心 / 簇内 observation 数 / 流式 merge 链（ahc 恒为空）。
        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        self.merged_into: dict[int, int] = {}

    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[Optional[dict[int, int]], ChunkDebugInfo]:
        """按 chunk 顺序缓冲带 embedding 的 observations，不做任何判定。

        Args:
            observations: 本 chunk 的一组 local track 观测。

        Returns:
            (None, 空 debug_info)；标签在 finalize() 统一给出。
        """
        # 离线后端此刻不做任何判定：只按 chunk 顺序缓冲带 embedding 的
        # observations，标签在 finalize 统一给出，因此这里返回 None。
        embedded = [obs for obs in observations if obs.embedding is not None]
        self._buffered.append(embedded)
        return None, _empty_debug_info()

    def finalize(self) -> list[dict[int, int]]:
        """对缓冲的全部 embedding 做一次层次聚类，返回逐 chunk 的分配。

        用 AgglomerativeClustering（cosine 距离 + distance_threshold，簇数
        完全由阈值决定）统一聚类，同时填充 centroids / counts 供 refined 级
        EOF 刷新时做小样本强制合并；小样本合并不在这里做，标签按原样写出
        为 raw 级。

        Returns:
            与缓冲 chunk 一一对应的 local->global 映射列表（无 observation
            时各项为空 dict）。
        """
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

        # 逐簇统计质心与样本计数（簇内 observation 数，与 streaming 侧
        # counts 口径对应），挂在实例上供 refined 级 EOF 刷新时做小样本
        # 强制合并；时长仅用于日志与调参参考。
        centroids, durations = cluster_stats(labels, observations, embeddings)
        self.centroids = centroids
        self.counts = {
            int(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        }
        logger.info(
            "[ahc] cluster durations: %s",
            {label: round(durations[label], 2) for label in sorted(durations)},
        )

        # 小样本强制合并不在这里做：标签按原样写出为 raw 级（*.raw.rttm），
        # 由 refined 级（post_process.RefinedRTTMWriter）在 EOF 最终刷新时
        # 叠加合并映射生成 *.refined.rttm，与 streaming 后端同一流程。

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


__all__ = ["AHCChunkClusterer"]
