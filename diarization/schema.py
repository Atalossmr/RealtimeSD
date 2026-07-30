"""chunk 管线的共享数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np


@dataclass
class ChunkObservation:
    """一个 chunk 内某个 local slot 聚合出的待分配观测。"""

    local_idx: int
    start: float
    end: float
    duration: float
    embedding: Optional[np.ndarray]
    mean_activity: float
    allow_centroid_update: bool
    selection_mode: str  # "non_overlap" | "overlap_fallback"


@dataclass
class SpeakerTurn:
    """写入 RTTM 的说话人时间段（speaker 为内部 global id）。"""

    start: float
    end: float
    speaker_id: int


@dataclass
class ChunkArtifacts:
    """一个 chunk 的全部生产物：observations + 帧级输出所需的参数。

    是嵌入提取阶段与聚类阶段之间的交换单元（见 diarization/utils/chunk_io.py）。
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
# 键名与 `ChunkSpeakerClusterer._resolve_observation` 写入的运行时 dict 保持一致。
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
    """chunk 级调试信息的固定结构。"""

    num_centroids_before: int
    num_centroids_after: int
    local_assignments: list[LocalAssignmentDebug]
    new_speakers: list[dict[str, int | float]]
    updated_speakers: list[dict[str, int | float | str]]
    skipped_updates: list[dict[str, int | float | str]]
    global_speakers: list[dict[str, int]]
