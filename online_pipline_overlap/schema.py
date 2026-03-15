"""在线说话人分离涉及的配置和数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np

from .constants import BASE_DIR


@dataclass
class PipelineConfig:
    """整条在线链路的统一配置。

    当前重构后的目标是“流程完整，但参数尽量少”。
    因此这里只保留真正会影响主流程的参数：
    - 音频调度；
    - segmentation 活跃阈值；
    - 候选片段长度；
    - 全局 speaker 匹配与更新；
    - 输出延迟与 RTTM 写出；
    - 调试与额外导出。
    """

    # 音频与在线调度参数。
    # 这里仍保留左右上下文分开配置，方便在离线回放和低延迟场景之间切换。
    sample_rate: int = 16000
    context_left_duration: float = 5.0
    context_right_duration: float = 5.0
    step: float = 0.5

    @property
    def chunk_duration(self) -> float:
        """返回总上下文时长。"""

        return self.context_left_duration + self.context_right_duration

    # segmentation 与全局说话人数量控制。
    # overlap 版本里，`target_primary_min_duration` 和 `target_overlap_min_duration` 会直接参与“目标时间附近 speaker 筛选”，
    # 用来抑制仅在极少量帧上闪一下的假激活 local slot，并支持主次说话人双阈值。
    tau_active: float = 0.68
    target_primary_min_duration: float = 0.15
    target_overlap_min_duration: float = 0.08
    delta_new: float = 0.68
    max_speakers: int = 10

    # ERes2NetV2 配置。
    model_type: str = "eres2netv2"
    embedding_size: int = 192
    feat_dim: int = 80
    m_channels: int = 64
    normalize_embeddings: bool = True

    # segmentation-3.0 配置。
    segmentation_model: str = "pyannote/segmentation-3.0"
    segmentation_batch_size: int = 1
    hf_token: Optional[str] = None
    hf_cache_dir: str = str(BASE_DIR / "pretrained" / "huggingface")
    device: str = "cpu"

    # 候选片段构造。
    # 重构后不再做复杂排序，只保留最基本的长度与位置限制。
    min_segment_duration: float = 0.35
    min_segment_duration_for_embedding: float = 0.8
    max_segment_duration_for_embedding: float = 2.5
    max_segment_shift_from_center: float = 1.5
    segment_batch_size: int = 8

    # 全局 speaker 维护。
    # 只保留最核心的三个控制量：主匹配阈值、SMA 窗口、重复更新片段重合阈值。
    global_match_threshold: float = 0.7
    merge_threshold: float = 0.8
    sma_window: int = 5
    update_segment_overlap_threshold: float = 0.8

    # 输出控制。
    # overlap 版本默认允许一个目标时刻输出两个 speaker，
    # 这样在重叠说话场景下，第二说话人不会在输出阶段被直接截掉。
    max_frame_speakers: int = 2
    streaming_flush_interval: float = 2.0
    streaming_merge_gap: float = 0.75
    delay_short_speaker_output: bool = False
    speaker_min_total_duration_to_emit: float = 0.0
    output_dir_for_streaming: Optional[str] = None
    save_segmentation_scores: bool = False
    debug: bool = False


@dataclass
class SpeakerTurn:
    """最终输出到 RTTM 的说话人时间段。"""

    start: float
    end: float
    speaker_id: int


@dataclass
class StreamingFrameDecision:
    """某个目标帧最终提交的说话人决策。"""

    time: float
    speakers: list[int]


@dataclass
class SegmentObservation:
    """一条进入全局分配器的观测记录。

    重构后，这个结构只保留真正参与后续流程的字段：
    - 来自哪个窗口、哪个 local slot；
    - 片段时间范围；
    - speaker embedding；
    - 若干最基础的解释性指标。
    """

    window_id: int
    local_idx: int
    start: float
    end: float
    center: float
    embedding: np.ndarray
    score_at_target: float
    mean_activity: float
    speech_ratio: float
    duration: float
    allow_centroid_update: bool
    selection_mode: str


@dataclass
class BufferedDecisionWindow:
    """单个目标帧对应的缓冲窗口。"""

    window_id: int
    target_time: float
    target_local_indices: list[int]
    chunk_start_time: float
    segmentation: np.ndarray
    absolute_centers: np.ndarray
    observations: list[SegmentObservation]


@dataclass
class ResolvedDecisionWindow:
    """已经完成全局 speaker 分配的目标帧窗口。"""

    window: BufferedDecisionWindow
    local_to_global: dict[int, int]
    debug_info: "WindowDebugInfo"


class AssignmentCostMatrixDebug(TypedDict):
    """记录 local x global 的联合分配矩阵。"""

    global_ids: list[int]
    cost_matrix: list[list[float]]
    similarity_matrix: list[list[float]]


LocalAssignmentDebug = TypedDict(
    "LocalAssignmentDebug",
    {
        "local": int,
        "global": int,
        "decision": str,
        "similarity": float,
        "score_at_target": float,
        "mean_activity": float,
        "speech_ratio": float,
        "selection_mode": str,
        "start": float,
        "end": float,
    },
)
"""记录单个 local observation 的最终归属。"""


NewSpeakerDebug = TypedDict(
    "NewSpeakerDebug",
    {
        "local": int,
        "global": int,
        "start": float,
        "end": float,
    },
)
"""记录新建 global speaker 的事件。"""


class MergedSpeakerDebug(TypedDict):
    """记录 global speaker 自动 merge 事件。"""

    large: int
    small: int
    similarity: float
    merged_count: int


UpdatedSpeakerDebug = TypedDict(
    "UpdatedSpeakerDebug",
    {
        "global": int,
        "mode": str,
        "alpha": float,
        "start": float,
        "end": float,
    },
)
"""记录 centroid 成功更新的事件。"""


SkippedUpdateDebug = TypedDict(
    "SkippedUpdateDebug",
    {
        "global": int,
        "reason": str,
        "start": float,
        "end": float,
        "overlap_ratio": float,
        "selection_mode": str,
    },
    total=False,
)
"""记录 observation 未参与 centroid 更新的原因。"""


class GlobalSpeakerDebug(TypedDict):
    """记录当前保留的 global speaker 摘要。"""

    speaker: int
    count: int
    dim: int


class WindowDebugInfo(TypedDict):
    """窗口级调试信息的固定结构。

    显式声明字段后，静态检查器就能知道：
    - 哪些键对应的是 list，可安全 `append` / `extend`
    - 哪些键对应的是数值
    - 哪些键对应的是结构化矩阵数据
    """

    num_centroids_before: int
    num_centroids_after: int
    assignment_cost_matrix: AssignmentCostMatrixDebug | None
    local_assignments: list[LocalAssignmentDebug]
    new_speakers: list[NewSpeakerDebug]
    merged_speakers: list[MergedSpeakerDebug]
    updated_speakers: list[UpdatedSpeakerDebug]
    skipped_updates: list[SkippedUpdateDebug]
    global_speakers: list[GlobalSpeakerDebug]


@dataclass
class ActiveStreamingTurn:
    """RTTM 流式写出阶段的活跃 turn 状态。"""

    start: float
    end: float
    flushed_until: float
