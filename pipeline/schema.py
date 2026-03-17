"""实时说话人识别涉及的配置和数据结构定义。

本文件集中定义：

- 运行配置 `PipelineConfig`
- 运行时共享的数据结构（observation/window/decision/RTTM turn 等）

约定：

- 时间单位统一为“秒”；
- speaker_id 为内部的全局 speaker id；
- RTTM writer 可能对 speaker_id 做重映射（例如延迟输出模式下的连续编号）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict

import numpy as np

from .constants import BASE_DIR


@dataclass
class PipelineConfig:
    """整条实时链路的统一配置。

    参数包括以下方面：
    - 音频调度；
    - segmentation 活跃阈值；
    - 候选片段长度；
    - 全局 speaker 匹配与更新；
    - 输出延迟与 RTTM 写出；
    - 调试与额外导出。
    """

    # 音频与实时调度参数。
    # 这里仍保留左右上下文分开配置，方便在离线回放和低延迟场景之间切换。
    sample_rate: int = 16000
    context_left_duration: float = 5.0
    context_right_duration: float = 5.0
    advance_step: float = 0.5

    @property
    def chunk_duration(self) -> float:
        """返回总上下文时长。"""

        return self.context_left_duration + self.context_right_duration

    # segmentation 与全局说话人数量控制。
    # 目标 speaker 选择规则：在 target_time 前后各半个 target_activity_window_duration 的窗口里，
    # 只有累计活跃时长达到 target_min_duration 的 local slot 才被认为 active。
    target_activity_window_duration: float = 0.5
    target_min_duration: float = 0.08
    new_speaker_threshold: float = 0.68
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
    # 包含匹配/合并阈值、片段长度门控、更新节奏与 observation reuse 相关参数。
    global_match_threshold: float = 0.7
    merge_threshold: float = 0.8
    min_segment_duration_for_new_speaker: float = 0.6
    min_segment_duration_for_centroid_update: float = 0.45
    enable_ema_update: bool = True
    centroid_warmup_window: int = 5
    stable_update_count_threshold: int = 10
    update_segment_overlap_threshold: float = 0.8
    weak_update_similarity_margin: float = 0.15
    weak_update_weight_multiplier: float = 0.25
    enable_observation_reuse: bool = True
    reuse_overlap_threshold: float = 0.9
    reuse_time_horizon: float = 1.0
    reuse_max_recent_records: int = 8

    # 输出控制。
    # overlap 版本默认允许一个目标时刻输出两个 speaker，
    # 这样在重叠说话场景下，第二说话人不会在输出阶段被直接截掉。
    max_frame_speakers: int = 2
    streaming_flush_interval: float = 2.0
    streaming_merge_gap: float = 0.75
    delay_short_speaker_output: bool = True
    output_dir_for_streaming: Optional[str] = None
    show_rttm: bool = False
    debug: bool = False

    # 说话人音轨转录与重叠分离配置。
    enable_speech_separation: bool = False
    separation_model: str = "JusperLee/TIGER-speech"
    min_overlap_duration_to_process: float = 0.3
    separation_required_duration: float = 3.0
    max_overlap_process_interval: float = 3.0
    export_uncertain_speaker_audio: bool = False
    speaker_audio_sample_rate: int = 16000
    speaker_audio_format: str = "wav"


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
    start: float
    end: float
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
class SegmentCandidate:
    """embedding 提取前的候选片段。"""

    window_id: int
    local_idx: int
    start: float
    end: float
    center: float
    score_at_target: float
    mean_activity: float
    speech_ratio: float
    duration: float
    allow_centroid_update: bool
    selection_mode: str


@dataclass
class ReusedObservationDecision:
    """命中复用逻辑后直接继承的 local->global 决策。"""

    local_idx: int
    global_id: int
    start: float
    end: float
    score_at_target: float
    mean_activity: float
    speech_ratio: float
    selection_mode: str
    overlap_ratio: float
    source_target_time: float


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
    reused_observations: list[ReusedObservationDecision] = field(default_factory=list)


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


class ReuseEventDebug(TypedDict):
    """记录复用命中的事件。"""

    local: int
    global_id: int
    start: float
    end: float
    overlap_ratio: float
    source_target_time: float


class WindowDebugInfo(TypedDict):
    """窗口级调试信息的固定结构。

    显式声明字段后，静态检查器就能知道：
    - 哪些键对应的是 list，可安全 `append` / `extend`
    - 哪些键对应的是数值
    - 哪些键对应的是结构化矩阵数据
    """

    num_centroids_before: int
    num_centroids_after: int
    num_reused_observations: int
    num_embedded_observations: int
    assignment_cost_matrix: AssignmentCostMatrixDebug | None
    local_assignments: list[LocalAssignmentDebug]
    new_speakers: list[NewSpeakerDebug]
    merged_speakers: list[MergedSpeakerDebug]
    updated_speakers: list[UpdatedSpeakerDebug]
    skipped_updates: list[SkippedUpdateDebug]
    reuse_events: list[ReuseEventDebug]
    global_speakers: list[GlobalSpeakerDebug]


@dataclass
class ActiveStreamingTurn:
    """RTTM 流式写出阶段的活跃 turn 状态。"""

    start: float
    end: float
    flushed_until: float
