"""extract 前端：波形 -> 逐 chunk 的 PreparedChunk（segmentation + track 构建）。

前端负责 embedding 之前的一切：滑窗/commit 区调度、pyannote
segmentation 局部识别、local track 聚合与选区。embedding 提取由
`common.models.embedding_infer.RegionEmbedder` 承担，两者在
`diarization.extract.extractor.ChunkExtractor` 中组合。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from common.config import ChunkPipelineConfig
from common.models.segmentation_infer import PyannoteStreamingSegmentation
from common.utils.device import resolve_device

from .scheduling import iter_chunk_schedule
from .track_builder import ChunkTrackBuilder, LocalTrack


logger = logging.getLogger(__name__)


@dataclass
class PreparedChunk:
    """前端为一个 chunk 备好的提 embedding 材料。

    Attributes:
        chunk_index: chunk 序号。
        seg_scores: segmentation 逐帧多标签分数矩阵 (num_frames, num_locals)。
        frame_step: 帧步长（秒），由 chunk_duration / 帧数反推。
        chunk_start: chunk 窗口起点（绝对秒）。
        commit_start: 提交区起点（绝对秒）。
        commit_end: 提交区终点（绝对秒）。
        tracks: 本 chunk 的 local track 列表（含提 embedding 的选区）。
    """

    chunk_index: int
    seg_scores: np.ndarray
    frame_step: float
    chunk_start: float
    commit_start: float
    commit_end: float
    tracks: list[LocalTrack] = field(default_factory=list)


class ChunkFrontend:
    """chunk 级前端：segmentation-3.0 局部识别 + local track 构建。"""

    def __init__(self, config: ChunkPipelineConfig):
        """初始化 segmentation 模型与 track 构造器。

        Args:
            config: chunk 管线配置。
        """

        self.config = config
        self.segmentation = PyannoteStreamingSegmentation(
            model_name=config.segmentation_model,
            duration=config.chunk_duration,
            batch_size=config.segmentation_batch_size,
            device=resolve_device(config.device),
            cache_dir=config.hf_cache_dir,
            use_auth_token=config.hf_token,
        )
        self.track_builder = ChunkTrackBuilder(config)

    def iter_prepared_chunks(
        self, waveform: torch.Tensor
    ) -> Iterator[PreparedChunk]:
        """按 chunk 顺序生产 PreparedChunk。

        窗口几何见 scheduling.iter_chunk_schedule；尾段不足一个 chunk 时
        补零后推理。

        Args:
            waveform: 单声道波形，形状 (1, samples)，采样率为
                config.sample_rate。

        Yields:
            PreparedChunk: 逐 chunk 的 segmentation 分数与 local track。
        """

        total_samples = waveform.shape[1]
        for window in iter_chunk_schedule(total_samples, self.config):
            # 尾段不足一个 chunk 时补零：segmentation 按固定 chunk_duration 输入推理。
            chunk = waveform[:, window.start_sample : window.start_sample + window.num_samples]
            if chunk.shape[1] < window.num_samples:
                chunk = F.pad(chunk, (0, window.num_samples - chunk.shape[1]))

            # 局部识别。返回的 centers（帧中心时刻）当前不使用：
            # 帧时间统一由 chunk_start + frame_idx * frame_step 推得。
            seg_scores, _ = self.segmentation(chunk, self.config.sample_rate)
            if seg_scores.size == 0:
                continue
            # chunk 已补零到 chunk_duration，帧在窗口内均匀分布；
            # 帧数由模型输出决定，frame_step 由此反推（约 16.9ms）。
            frame_step = self.config.chunk_duration / seg_scores.shape[0]

            tracks = self.track_builder.build_tracks(
                seg_scores,
                frame_step,
                window.start,
                window.commit_start,
                window.commit_end,
            )

            yield PreparedChunk(
                chunk_index=window.index,
                seg_scores=seg_scores,
                frame_step=frame_step,
                chunk_start=window.start,
                commit_start=window.commit_start,
                commit_end=window.commit_end,
                tracks=tracks,
            )


__all__ = ["PreparedChunk", "ChunkFrontend"]
