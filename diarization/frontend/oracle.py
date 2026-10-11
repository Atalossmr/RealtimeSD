"""oracle 前端：ref RTTM 驱动，与 ChunkFrontend 平级互换。

不运行任何模型：帧级活跃直接来自参考 RTTM（真值），按与
`diarization.frontend.scheduling` 完全相同的滑窗/commit 区调度产出
`PreparedChunk`——seg_scores 为 0/1 硬分数（列 = chunk 提交区内活跃
speaker，int id 升序），track 选区优先独唱区（干净），不足时回退全
活跃区（overlap_fallback，禁止更新 centroid），语义与 ChunkTrackBuilder
对齐。

产出的 PreparedChunk 接 extract 管线的组合助手见
`diarization.extract.oracle.oracle_chunk_artifacts`（此处不 import
extract，避免包间循环依赖）；pipeline 的 --oracle 模式即二者组合。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from diarization.config import ChunkPipelineConfig
from diarization.utils.rttm import FrameActivity, build_frame_activities, load_rttm

from .frontend import PreparedChunk
from .scheduling import iter_chunk_schedule
from .track_builder import LocalTrack


logger = logging.getLogger(__name__)


def _mask_regions(
    frames: list[FrameActivity], mask: list[bool]
) -> list[tuple[float, float]]:
    """连续为真的帧归并为 (start, end) 区间（秒）。

    Args:
        frames: 窗口内逐帧活跃记录（按时间有序）。
        mask: 逐帧布尔掩码，与 frames 等长。

    Returns:
        归并后的 (start, end) 区间列表，按时间升序。
    """

    regions: list[tuple[float, float]] = []
    for (frame_start, frame_end, _), flag in zip(frames, mask):
        if not flag:
            continue
        if regions and frame_start <= regions[-1][1] + 1e-6:
            regions[-1] = (regions[-1][0], frame_end)
        else:
            regions.append((frame_start, frame_end))
    return regions


def _cap_regions(
    regions: list[tuple[float, float]], max_duration: float
) -> list[tuple[float, float]]:
    """从起点截断区间列表，总时长不超过 max_duration。

    Args:
        regions: (start, end) 区间列表（按时间升序）。
        max_duration: 总时长上限（秒）。

    Returns:
        截断后的区间列表。
    """

    capped: list[tuple[float, float]] = []
    total = 0.0
    for start, end in regions:
        if total >= max_duration:
            break
        keep = min(end, start + (max_duration - total))
        capped.append((start, keep))
        total += keep - start
    return capped


class OracleFrontend:
    """ref RTTM 驱动的 oracle 前端（无模型，与 ChunkFrontend 同接口）。"""

    def __init__(self, config: ChunkPipelineConfig):
        """初始化 oracle 前端。

        Args:
            config: chunk 管线配置（取滑窗调度、帧步长与选段门控参数）。
        """

        self.config = config

    def iter_prepared_chunks(
        self, waveform: torch.Tensor, rttm_path: str | Path
    ) -> Iterator[PreparedChunk]:
        """按 chunk 顺序生产 oracle PreparedChunk。

        帧网格：每 chunk round(chunk_duration / oracle_frame_step) 帧，
        frame_step = chunk_duration / 帧数（默认 50ms 整除，与全局网格对齐）。
        seg_scores 为 (num_frames, num_active_speakers) 的 0/1 矩阵，
        列按下述 int speaker id 升序；label -> int id 按字典序映射（0 起）。

        Args:
            waveform: 单声道波形，形状 (1, samples)，采样率为
                config.sample_rate。
            rttm_path: 参考 RTTM 路径（真值标注）。

        Yields:
            PreparedChunk: 逐 chunk 的 oracle 分数矩阵与 local track。
        """

        config = self.config
        total_samples = waveform.shape[1]
        t_end = total_samples / config.sample_rate

        turns_by_label = load_rttm(rttm_path)
        # speaker label 按字典序映射为 int id（0 起），与 RTTM 行序无关、稳定可复现。
        labels = sorted(turns_by_label)
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        turns_by_id = {label_to_id[label]: turns_by_label[label] for label in labels}

        num_frames_per_chunk = max(
            1, int(round(config.chunk_duration / config.oracle_frame_step))
        )
        frame_step = config.chunk_duration / num_frames_per_chunk
        global_frames = build_frame_activities(turns_by_id, frame_step, t_end)

        for window in iter_chunk_schedule(total_samples, config):
            first_frame = int(round(window.start / frame_step))
            window_frames: list[FrameActivity] = []
            for row in range(num_frames_per_chunk):
                frame_idx = first_frame + row
                if frame_idx < len(global_frames):
                    window_frames.append(global_frames[frame_idx])
                else:
                    # 窗口尾部超出音频末尾的帧：无活跃（补零行）。
                    frame_start = frame_idx * frame_step
                    window_frames.append(
                        (frame_start, frame_start + frame_step, [])
                    )

            # 提交区内有活动的 speaker 建列（margin 帧不提交，无需建列）。
            commit_active: set[int] = set()
            for frame_start, frame_end, active in window_frames:
                if frame_start < window.commit_end and frame_end > window.commit_start:
                    commit_active.update(active)
            columns = sorted(commit_active)
            column_of = {speaker_id: col for col, speaker_id in enumerate(columns)}

            seg_scores = np.zeros(
                (num_frames_per_chunk, len(columns)), dtype=np.float32
            )
            for row, (_, _, active) in enumerate(window_frames):
                for speaker_id in active:
                    col = column_of.get(speaker_id)
                    if col is not None:
                        seg_scores[row, col] = 1.0

            tracks = [
                track
                for speaker_id in columns
                if (
                    track := self._build_track(
                        window_frames,
                        speaker_id,
                        column_of[speaker_id],
                        window.commit_start,
                        window.commit_end,
                    )
                )
                is not None
            ]

            yield PreparedChunk(
                chunk_index=window.index,
                seg_scores=seg_scores,
                frame_step=frame_step,
                chunk_start=window.start,
                commit_start=window.commit_start,
                commit_end=window.commit_end,
                tracks=tracks,
            )

    def _build_track(
        self,
        window_frames: list[FrameActivity],
        speaker_id: int,
        local_idx: int,
        commit_start: float,
        commit_end: float,
    ) -> Optional[LocalTrack]:
        """为单个 speaker 构造 oracle local track（选区语义对齐 ChunkTrackBuilder）。

        优先独唱区（仅其一人活跃的帧，干净）；不足
        min_segment_duration_for_embedding 时回退全活跃区
        （overlap_fallback，禁止更新 centroid）；仍不足则放弃。
        提交区内活跃时长 < min_local_activity_duration 时不建 track。

        Args:
            window_frames: 窗口内逐帧活跃记录。
            speaker_id: 目标 speaker id。
            local_idx: 该 speaker 在 seg_scores 中的列号。
            commit_start: 提交区起点（绝对秒）。
            commit_end: 提交区终点（绝对秒）。

        Returns:
            oracle LocalTrack；不满足建 track 条件时返回 None。
        """

        config = self.config
        active_mask = [speaker_id in active for _, _, active in window_frames]
        solo_mask = [active == [speaker_id] for _, _, active in window_frames]

        # 提交区内的活跃时长与边界（帧边界截到提交区）。
        commit_duration = 0.0
        commit_bounds: list[float] = []
        for (frame_start, frame_end, _), flag in zip(window_frames, active_mask):
            if not flag:
                continue
            start = max(frame_start, commit_start)
            end = min(frame_end, commit_end)
            if end > start:
                commit_duration += end - start
                commit_bounds.extend((start, end))
        if commit_duration < config.min_local_activity_duration:
            return None

        solo_regions = _mask_regions(window_frames, solo_mask)
        solo_total = sum(end - start for start, end in solo_regions)
        if solo_total >= config.min_segment_duration_for_embedding:
            regions = _cap_regions(
                solo_regions, config.max_segment_duration_for_embedding
            )
            selection_mode = "non_overlap"
            allow_centroid_update = True
        else:
            active_regions = _mask_regions(window_frames, active_mask)
            active_total = sum(end - start for start, end in active_regions)
            if active_total < config.min_segment_duration_for_embedding:
                return None
            regions = _cap_regions(
                active_regions, config.max_segment_duration_for_embedding
            )
            selection_mode = "overlap_fallback"
            allow_centroid_update = False

        return LocalTrack(
            local_idx=local_idx,
            regions=regions,
            active_start=commit_bounds[0],
            active_end=commit_bounds[-1],
            active_duration=commit_duration,
            mean_activity=1.0,
            allow_centroid_update=allow_centroid_update,
            selection_mode=selection_mode,
        )


__all__ = ["OracleFrontend"]
