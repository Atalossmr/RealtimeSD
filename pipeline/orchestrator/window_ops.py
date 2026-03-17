"""主编排窗口相关工具。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..schema import PipelineConfig


def slice_window(
    *,
    config: PipelineConfig,
    waveform: torch.Tensor,
    target_time: float,
) -> tuple[torch.Tensor, float]:
    """围绕目标帧截取固定长度上下文，并在边界处补零。"""

    chunk_samples = int(round(config.chunk_duration * config.sample_rate))
    start_time = target_time - config.context_left_duration
    end_time = target_time + config.context_right_duration

    total_samples = int(waveform.shape[1])
    start_sample = int(np.floor(start_time * config.sample_rate))
    end_sample = int(np.ceil(end_time * config.sample_rate))

    left_pad = max(0, -start_sample)
    right_pad = max(0, end_sample - total_samples)
    valid_start = max(0, start_sample)
    valid_end = min(total_samples, end_sample)
    chunk = waveform[:, valid_start:valid_end]

    if left_pad > 0 or right_pad > 0:
        chunk = F.pad(chunk, (left_pad, right_pad))
    if chunk.shape[1] < chunk_samples:
        chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[1]))
    elif chunk.shape[1] > chunk_samples:
        chunk = chunk[:, :chunk_samples]

    return chunk, start_time


def target_frame_index(
    absolute_centers: np.ndarray, target_time: float
) -> Optional[int]:
    """找到最接近目标时刻的 segmentation 帧索引。"""

    if absolute_centers.size == 0:
        return None
    return int(np.argmin(np.abs(absolute_centers - target_time)))


def target_frame_speakers(
    *,
    config: PipelineConfig,
    segment_builder,
    segmentation_scores: np.ndarray,
    absolute_centers: np.ndarray,
    target_time: float,
    target_frame_idx: Optional[int],
    local_to_global: dict[int, int],
) -> list[int]:
    """把目标时间附近活跃的 local slot 映射成最终 global speaker。"""

    if target_frame_idx is None or segmentation_scores.size == 0:
        return []

    local_activity_summary = segment_builder.summarize_target_local_activity(
        segmentation_scores,
        absolute_centers,
        target_time,
    )
    summary_by_local = {int(item["local"]): item for item in local_activity_summary}

    aggregate_by_global: dict[int, dict[str, float]] = {}
    frame_scores = segmentation_scores[target_frame_idx]

    for local_idx, global_id in local_to_global.items():
        if local_idx >= len(frame_scores):
            continue
        local_summary = summary_by_local.get(int(local_idx))
        if local_summary is None:
            continue
        entry = aggregate_by_global.setdefault(
            int(global_id),
            {
                "active_duration": 0.0,
                "mean_score": 0.0,
                "target_score": 0.0,
                "num_locals": 0.0,
            },
        )
        prev_duration = entry["active_duration"]
        new_duration = float(local_summary["active_duration"])
        if prev_duration + new_duration > 0:
            entry["mean_score"] = (
                entry["mean_score"] * prev_duration
                + float(local_summary["mean_score"]) * new_duration
            ) / (prev_duration + new_duration)
        entry["active_duration"] += new_duration

        entry["target_score"] = max(
            float(entry["target_score"]),
            float(frame_scores[local_idx]),
        )
        entry["num_locals"] += 1.0

    scored_globals = [
        (
            float(values["active_duration"]),
            float(values["mean_score"]),
            float(values["target_score"]),
            int(global_id),
        )
        for global_id, values in aggregate_by_global.items()
    ]

    scored_globals.sort(reverse=True)
    if not scored_globals:
        return []

    return [
        global_id for _, _, _, global_id in scored_globals[: config.max_frame_speakers]
    ]
