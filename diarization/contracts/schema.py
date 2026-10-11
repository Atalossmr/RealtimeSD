"""chunk 管线的共享数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np


@dataclass
class ChunkObservation:
    """一个 chunk 内某个 local slot 聚合出的待分配观测。

    Attributes:
        local_idx: chunk 内的 local slot 序号。
        start: 观测起始时间（秒）。
        end: 观测结束时间（秒）。
        duration: 观测时长（秒）。
        embedding: 聚合的说话人 embedding；无可用 embedding 时为 None。
        mean_activity: 观测区间内的平均活跃度。
        allow_centroid_update: 是否允许用该观测更新 centroid。
        selection_mode: embedding 选取方式，"non_overlap" | "overlap_fallback"。
    """

    local_idx: int
    start: float
    end: float
    duration: float
    embedding: Optional[np.ndarray]
    mean_activity: float
    allow_centroid_update: bool
    selection_mode: str  # "non_overlap" | "overlap_fallback"


@dataclass
class ChunkArtifacts:
    """一个 chunk 的全部生产物：observations + 帧级输出所需的参数。

    是嵌入提取阶段与聚类阶段之间的交换单元（见 diarization/utils/chunk_io.py）。

    Attributes:
        chunk_index: chunk 序号。
        seg_scores: (num_frames, num_local_slots) 的帧级活跃分数。
        frame_step: 帧移（秒）。
        chunk_start: chunk 起始时间（秒）。
        commit_start: 提交区起始时间（秒）。
        commit_end: 提交区结束时间（秒）。
        observations: 本 chunk 聚合出的待分配观测列表。
    """

    chunk_index: int
    seg_scores: np.ndarray
    frame_step: float
    chunk_start: float
    commit_start: float
    commit_end: float
    observations: list[ChunkObservation]


# 单个 local slot 的分配调试信息。
# "global" 是 Python 保留字，类语法无法声明该字段，因此用函数式语法定义；
# 键名与 `StreamingChunkClusterer._resolve_observation` 写入的运行时 dict 保持一致。
LocalAssignmentDebug = TypedDict(
    "LocalAssignmentDebug",
    {
        "local": int,
        "global": int,  # global speaker id
        "decision": str,
        "similarity": float,
        "start": float,
        "end": float,
        "selection_mode": str,
    },
)


class ChunkDebugInfo(TypedDict):
    """chunk 级调试信息的固定结构。

    Attributes:
        num_centroids_before: 本 chunk 聚类前的 centroid 数。
        num_centroids_after: 本 chunk 聚类后的 centroid 数。
        local_assignments: 各 local slot 的分配调试信息。
        new_speakers: 本 chunk 新建的 global speaker。
        updated_speakers: centroid 被更新的 speaker。
        skipped_updates: 被跳过的 centroid 更新及原因。
        global_speakers: 聚类后全部 global speaker 概览。
    """

    num_centroids_before: int
    num_centroids_after: int
    local_assignments: list[LocalAssignmentDebug]
    new_speakers: list[dict[str, int | float]]
    updated_speakers: list[dict[str, int | float | str]]
    skipped_updates: list[dict[str, int | float | str]]
    global_speakers: list[dict[str, int]]
