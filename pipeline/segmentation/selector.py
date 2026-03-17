"""segmentation 目标帧活动统计逻辑。"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..schema import PipelineConfig


def window_frame_mask(
    *,
    absolute_centers: np.ndarray,
    target_time: float,
    activity_window_duration: float,
    frame_step_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, float]:
    """返回 target_time 附近的统计窗口及对应 frame_step。"""

    frame_step = float(frame_step_fn(absolute_centers))
    half_window = 0.5 * float(activity_window_duration)
    window_start = float(target_time) - half_window
    window_end = float(target_time) + half_window
    frame_mask = np.logical_and(
        absolute_centers >= window_start,
        absolute_centers <= window_end,
    )
    if not np.any(frame_mask):
        target_frame_idx = int(np.argmin(np.abs(absolute_centers - target_time)))
        frame_mask[target_frame_idx] = True
    return frame_mask, frame_step


def select_target_local_indices(
    *,
    config: PipelineConfig,
    segmentation: np.ndarray,
    absolute_centers: np.ndarray,
    target_time: float,
    frame_step_fn: Callable[[np.ndarray], float],
) -> list[int]:
    """围绕 target_time 在多帧范围内挑选需要跟踪的 local speaker。"""

    if segmentation.size == 0 or absolute_centers.size == 0:
        return []

    num_frames, num_locals = segmentation.shape
    if num_frames == 0 or num_locals == 0:
        return []

    frame_mask, frame_step = window_frame_mask(
        absolute_centers=absolute_centers,
        target_time=float(target_time),
        activity_window_duration=float(config.target_activity_window_duration),
        frame_step_fn=frame_step_fn,
    )
    active_scores = segmentation[frame_mask]
    active_binary = active_scores > 0.0
    frame_step = max(1e-6, float(frame_step))
    active_durations = np.sum(active_binary, axis=0).astype(np.float32) * frame_step
    active_mask = active_durations >= float(config.target_min_duration)
    return [
        int(local_idx)
        for local_idx, is_active in enumerate(active_mask.tolist())
        if bool(is_active)
    ]


def summarize_target_local_activity(
    *,
    config: PipelineConfig,
    segmentation: np.ndarray,
    absolute_centers: np.ndarray,
    target_time: float,
    frame_step_fn: Callable[[np.ndarray], float],
) -> list[dict[str, float]]:
    """统计 target_time 附近每个 local slot 的活跃情况。"""

    if segmentation.size == 0 or absolute_centers.size == 0:
        return []

    frame_mask, frame_step = window_frame_mask(
        absolute_centers=absolute_centers,
        target_time=float(target_time),
        activity_window_duration=float(config.target_activity_window_duration),
        frame_step_fn=frame_step_fn,
    )
    if not np.any(frame_mask):
        return []

    window_scores = segmentation[frame_mask]
    window_binary = window_scores > 0.0
    active_durations = np.sum(window_binary, axis=0).astype(np.float32) * float(
        frame_step
    )
    mean_scores = np.mean(window_scores, axis=0)
    max_scores = np.max(window_scores, axis=0)

    summary: list[dict[str, float]] = []
    for local_idx in range(window_scores.shape[1]):
        summary.append(
            {
                "local": int(local_idx),
                "active_duration": float(active_durations[local_idx]),
                "mean_score": float(mean_scores[local_idx]),
                "max_score": float(max_scores[local_idx]),
            }
        )
    summary.sort(
        key=lambda item: (item["active_duration"], item["mean_score"]),
        reverse=True,
    )
    return summary
