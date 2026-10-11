"""commit 区帧裁剪：RTTM writer 的逐帧遍历逻辑。

当前唯一消费者是 `diarization/cluster/rttm_writer.py`；tools/experiments/
的 oracle 实验脚本镜像同一判定口径（见其 docstring），因此实现留在
diarization.contracts 作为单一口径来源。
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_commit_frames(
    seg_scores: np.ndarray,
    frame_step: float,
    chunk_start: float,
    commit_start: float,
    commit_end: float,
    local_to_global: dict[int, int],
) -> Iterator[tuple[float, float, list[int]]]:
    """逐帧产出 (裁剪后的 frame_start, frame_end, active_global_ids)。

    只输出与提交区 [commit_start, commit_end) 相交的帧；跨界帧裁剪到提交区
    边界。相邻 chunk 的提交区无缝拼接，因此不重复也不遗漏。
    segmentation-3.0 经 powerset 转硬标签（0/1），> 0.0 即活跃；未分配
    global（未建 track / 未过门控）的 local slot 不计入活跃集合，
    该帧仍以空的 active_global_ids 产出。

    Args:
        seg_scores: (num_frames, num_local_slots) 的帧级活跃分数。
        frame_step: 帧移（秒）；<= 0 时按 1e-6 兜底。
        chunk_start: chunk 起始时间（秒）。
        commit_start: 提交区起始时间（秒，含）。
        commit_end: 提交区结束时间（秒，不含）。
        local_to_global: local slot 到 global speaker id 的映射。

    Yields:
        (frame_start, frame_end, active_global_ids) 三元组：裁剪到提交区
        内的帧区间，及该帧活跃 global speaker id 的升序列表（可为空）。
    """

    frame_step = max(1e-6, float(frame_step))
    for frame_idx in range(seg_scores.shape[0]):
        frame_start = chunk_start + frame_idx * frame_step
        frame_end = frame_start + frame_step
        if frame_end <= commit_start + 1e-9:
            continue
        if frame_start >= commit_end - 1e-9:
            break
        frame_start = max(frame_start, commit_start)
        frame_end = min(frame_end, commit_end)
        frame_scores = seg_scores[frame_idx]
        active_globals = sorted(
            {
                int(local_to_global[local_idx])
                for local_idx in range(len(frame_scores))
                if frame_scores[local_idx] > 0.0 and local_idx in local_to_global
            }
        )
        yield frame_start, frame_end, active_globals


__all__ = ["iter_commit_frames"]
