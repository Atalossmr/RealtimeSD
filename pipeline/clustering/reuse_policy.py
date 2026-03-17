"""clustering 复用策略函数。"""

from __future__ import annotations

from typing import Optional, Mapping

import numpy as np

from ..schema import ReusedObservationDecision, SegmentCandidate


def segment_overlap_ratio(
    *, left_start: float, left_end: float, right_start: float, right_end: float
) -> float:
    """计算两个片段的归一化时间重合比。"""

    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    min_duration = max(1e-6, min(left_end - left_start, right_end - right_start))
    return float(overlap / min_duration)


def prune_recent_assignments(
    *, recent_assignments: list, current_target_time: float, reuse_time_horizon: float
) -> list:
    """按时间窗清理过期复用缓存。"""

    if not recent_assignments:
        return []
    return [
        item
        for item in recent_assignments
        if (current_target_time - float(item.target_time)) <= float(reuse_time_horizon)
    ]


def try_reuse_assignment(
    *,
    candidate: SegmentCandidate,
    target_time: float,
    enable_observation_reuse: bool,
    recent_assignments: list,
    centroids: Mapping[int, np.ndarray],
    reuse_overlap_threshold: float,
    reuse_time_horizon: float,
) -> tuple[Optional[ReusedObservationDecision], list]:
    """尝试为候选片段复用近期分配结果。"""

    if not enable_observation_reuse:
        return None, recent_assignments

    pruned = prune_recent_assignments(
        recent_assignments=recent_assignments,
        current_target_time=float(target_time),
        reuse_time_horizon=float(reuse_time_horizon),
    )

    best = None
    best_overlap = -1.0
    for item in reversed(pruned):
        if int(item.global_id) not in centroids:
            continue

        overlap_ratio = segment_overlap_ratio(
            left_start=float(item.start),
            left_end=float(item.end),
            right_start=float(candidate.start),
            right_end=float(candidate.end),
        )
        if overlap_ratio < float(reuse_overlap_threshold):
            continue
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best = item

    if best is None:
        return None, pruned

    return (
        ReusedObservationDecision(
            local_idx=int(candidate.local_idx),
            global_id=int(best.global_id),
            start=float(candidate.start),
            end=float(candidate.end),
            score_at_target=float(candidate.score_at_target),
            mean_activity=float(candidate.mean_activity),
            speech_ratio=float(candidate.speech_ratio),
            selection_mode=str(candidate.selection_mode),
            overlap_ratio=float(best_overlap),
            source_target_time=float(best.target_time),
        ),
        pruned,
    )


def record_recent_assignment(
    *,
    enable_observation_reuse: bool,
    recent_assignments: list,
    local_idx: int,
    global_id: int,
    start: float,
    end: float,
    target_time: float,
    decision: str,
    reuse_time_horizon: float,
    reuse_max_recent_records: int,
    record_factory,
) -> list:
    """记录可复用的近期分配，并做容量裁剪。"""

    if not enable_observation_reuse:
        return recent_assignments
    if decision != "matched":
        return recent_assignments

    updated = list(recent_assignments)
    updated.append(
        record_factory(
            local_idx=int(local_idx),
            global_id=int(global_id),
            start=float(start),
            end=float(end),
            target_time=float(target_time),
            decision=str(decision),
        )
    )

    updated = prune_recent_assignments(
        recent_assignments=updated,
        current_target_time=float(target_time),
        reuse_time_horizon=float(reuse_time_horizon),
    )

    updated.sort(key=lambda it: float(it.target_time))
    max_records = max(1, int(reuse_max_recent_records))
    return updated[-max_records:]
