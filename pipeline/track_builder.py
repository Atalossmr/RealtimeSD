"""chunk 内 local track 构造。

把 segmentation 的逐帧多标签分数矩阵转成每个 local slot 的聚合 track：
- 优先拼接"非重叠纯净区"用于提 embedding；
- 纯净区不足时回退到全活跃区（overlap_fallback）， embedding 仍可提取但不允许更新 centroid；
- 活跃时长过短的 local slot 直接跳过（不建 track、帧也不输出）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ChunkPipelineConfig
from .schema import ChunkObservation


@dataclass
class LocalTrack:
    """一个 local slot 在 chunk 内的聚合结果。"""

    local_idx: int
    regions: list[tuple[float, float]] = field(default_factory=list)
    active_start: float = 0.0
    active_end: float = 0.0
    active_duration: float = 0.0
    mean_activity: float = 0.0
    allow_centroid_update: bool = True
    selection_mode: str = "non_overlap"

    @property
    def embed_duration(self) -> float:
        return float(sum(end - start for start, end in self.regions))


class ChunkTrackBuilder:
    """从单个 chunk 的 segmentation 结果构造 local track observation。"""

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config

    @staticmethod
    def _connected_regions(active_mask: np.ndarray) -> list[tuple[int, int]]:
        """把连续为真的帧区间合并成若干个连续 region（半开区间）。"""

        regions: list[tuple[int, int]] = []
        start = None
        for idx, value in enumerate(active_mask.tolist()):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                regions.append((start, idx))
                start = None
        if start is not None:
            regions.append((start, len(active_mask)))
        return regions

    def _select_regions(
        self,
        mask: np.ndarray,
        local_scores: np.ndarray,
        frame_step: float,
        chunk_start_time: float,
    ) -> list[tuple[float, float]]:
        """从掩码中按平均活跃度降序挑选连通区，拼接时长封顶。"""

        regions = self._connected_regions(mask)
        if not regions:
            return []

        scored = []
        for start_idx, end_idx in regions:
            scores_slice = local_scores[start_idx:end_idx]
            mean_score = float(np.mean(scores_slice)) if scores_slice.size else 0.0
            scored.append((mean_score, start_idx, end_idx))
        scored.sort(reverse=True)

        max_duration = float(self.config.max_segment_duration_for_embedding)
        selected: list[tuple[float, float]] = []
        total = 0.0
        for _, start_idx, end_idx in scored:
            if total >= max_duration:
                break
            start_time = chunk_start_time + start_idx * frame_step
            end_time = chunk_start_time + end_idx * frame_step
            remaining = max_duration - total
            if end_time - start_time > remaining:
                end_time = start_time + remaining
            selected.append((float(start_time), float(end_time)))
            total += end_time - start_time

        selected.sort()
        return selected

    def build_tracks(
        self,
        seg_scores: np.ndarray,
        frame_step: float,
        chunk_start_time: float,
    ) -> list[LocalTrack]:
        """为一个 chunk 的全部 local slot 构造 track。"""

        tracks: list[LocalTrack] = []
        if seg_scores.size == 0:
            return tracks

        num_frames, num_locals = seg_scores.shape
        if num_frames == 0 or num_locals == 0:
            return tracks

        config = self.config
        frame_step = max(1e-6, float(frame_step))
        all_active = seg_scores > 0.0
        overlap_frames = np.sum(all_active, axis=1) >= 2

        for local_idx in range(num_locals):
            local_active = all_active[:, local_idx]
            active_duration = float(np.sum(local_active)) * frame_step
            if active_duration < config.min_local_activity_duration:
                continue

            local_scores = seg_scores[:, local_idx]
            active_indices = np.flatnonzero(local_active)
            active_start = chunk_start_time + int(active_indices[0]) * frame_step
            active_end = (
                chunk_start_time + (int(active_indices[-1]) + 1) * frame_step
            )
            mean_activity = float(np.mean(local_scores[local_active]))

            # 优先使用非重叠纯净区。
            pure_mask = np.logical_and(local_active, np.logical_not(overlap_frames))
            regions = self._select_regions(
                pure_mask, local_scores, frame_step, chunk_start_time
            )
            pure_duration = float(sum(end - start for start, end in regions))
            allow_update = True
            selection_mode = "non_overlap"

            # 纯净区不足时回退到全活跃区。
            if pure_duration < config.min_segment_duration_for_embedding:
                regions = self._select_regions(
                    local_active, local_scores, frame_step, chunk_start_time
                )
                fallback_duration = float(
                    sum(end - start for start, end in regions)
                )
                if fallback_duration < config.min_segment_duration_for_embedding:
                    continue
                allow_update = False
                selection_mode = "overlap_fallback"

            tracks.append(
                LocalTrack(
                    local_idx=int(local_idx),
                    regions=regions,
                    active_start=float(active_start),
                    active_end=float(active_end),
                    active_duration=float(active_duration),
                    mean_activity=mean_activity,
                    allow_centroid_update=allow_update,
                    selection_mode=selection_mode,
                )
            )

        return tracks

    def to_observation(
        self,
        track: LocalTrack,
        embedding: Optional[np.ndarray],
    ) -> ChunkObservation:
        """把 track 与 embedding 组装成分配器使用的 observation。"""

        return ChunkObservation(
            local_idx=int(track.local_idx),
            start=float(track.active_start),
            end=float(track.active_end),
            duration=float(track.embed_duration),
            embedding=embedding,
            mean_activity=float(track.mean_activity),
            allow_centroid_update=bool(track.allow_centroid_update),
            selection_mode=str(track.selection_mode),
        )


__all__ = ["LocalTrack", "ChunkTrackBuilder"]
