"""chunk 管线主编排模块。

每个 10s chunk 的处理流程：

1. 切 chunk（尾部补零到固定长度）；
2. segmentation-3.0 局部识别，得到帧级多标签分数；
3. 每个 local slot 聚合纯净语音提 ERes2NetV2 embedding；
4. `ChunkSpeakerClusterer` Hungarian 分配 local->global，维护全局身份；
5. 帧级活跃 local 映射 global 后写出：confirmed 即时进入 open-turn 管线，
   probationary 先入内存缓冲，身份定案（转正/吸收）后 flush 追加；
6. 音频结束后定案残余 probationary 并 flush，writer 纯追加收尾（全程零重写）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from speakerlab.process.processor import FBank
from speakerlab.utils.fileio import load_audio

from .clusterer import ChunkSpeakerClusterer
from .config import ChunkPipelineConfig
from .rttm_writer import AppendOnlyRTTMWriter
from .schema import ChunkDebugInfo, ChunkObservation
from .track_builder import ChunkTrackBuilder, LocalTrack
from .models.embedding_infer import (
    NativeERes2NetV2SegmentEmbedder,
    load_embedding_model,
)
from .models.segmentation_infer import PyannoteStreamingSegmentation
from .utils import resample_waveform_if_needed, resolve_device


logger = logging.getLogger(__name__)


class ChunkDiarizationPipeline:
    """chunk 级实时说话人分离主控类。"""

    def __init__(
        self,
        config: ChunkPipelineConfig,
        embedding_model_path: Optional[str],
    ):
        self.config = config
        self.device = resolve_device(config.device)

        self.embedding_model = load_embedding_model(
            model_path=embedding_model_path,
            device=self.device,
            model_type=config.model_type,
            feat_dim=config.feat_dim,
            embedding_size=config.embedding_size,
            m_channels=config.m_channels,
        )
        self.feature_extractor = FBank(
            n_mels=config.feat_dim,
            sample_rate=config.sample_rate,
            mean_nor=True,
        )
        self.segmentation = PyannoteStreamingSegmentation(
            model_name=config.segmentation_model,
            duration=config.chunk_duration,
            batch_size=config.segmentation_batch_size,
            device=self.device,
            cache_dir=config.hf_cache_dir,
            use_auth_token=config.hf_token,
        )
        self.embedder = NativeERes2NetV2SegmentEmbedder(
            embedding_model=self.embedding_model,
            feature_extractor=self.feature_extractor,
            sample_rate=config.sample_rate,
            normalize_embeddings=config.normalize_embeddings,
        )
        self.track_builder = ChunkTrackBuilder(config)
        self.clusterer = ChunkSpeakerClusterer(config)

    def reset(self) -> None:
        """为新音频重置全局聚类状态。"""

        self.clusterer = ChunkSpeakerClusterer(self.config)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _format_log_payload(payload: object) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _log_structured(
        self, level: int, prefix: str, title: str, payload: object
    ) -> None:
        logger.log(level, "%s %s:\n%s", prefix, title, self._format_log_payload(payload))

    def _slice_regions(
        self,
        waveform: torch.Tensor,
        regions: list[tuple[float, float]],
    ) -> Optional[torch.Tensor]:
        """按绝对时间从整段波形裁出各 region 并拼接。"""

        sample_rate = self.config.sample_rate
        total_samples = waveform.shape[1]
        pieces: list[torch.Tensor] = []
        for start, end in regions:
            start_sample = max(0, min(int(round(start * sample_rate)), total_samples))
            end_sample = max(start_sample, min(int(round(end * sample_rate)), total_samples))
            if end_sample > start_sample:
                pieces.append(waveform[:, start_sample:end_sample])
        if not pieces:
            return None
        return torch.cat(pieces, dim=1)

    def _embed_tracks(
        self,
        waveform: torch.Tensor,
        tracks: list[LocalTrack],
    ) -> list[ChunkObservation]:
        """批量为各 track 提 embedding 并组装 observation。"""

        waveforms: list[torch.Tensor] = []
        pending: list[LocalTrack] = []
        for track in tracks:
            segment = self._slice_regions(waveform, track.regions)
            if segment is None or segment.shape[1] <= 0:
                continue
            waveforms.append(segment)
            pending.append(track)

        observations: list[ChunkObservation] = []
        # 分批提 embedding，避免单批过大占爆显存。
        batch_size = max(1, int(self.config.segment_batch_size))
        for start in range(0, len(pending), batch_size):
            batch_waveforms = waveforms[start : start + batch_size]
            batch_embeddings = self.embedder.embed_segments(batch_waveforms)
            for track, embedding in zip(
                pending[start : start + batch_size], batch_embeddings
            ):
                observations.append(self.track_builder.to_observation(track, embedding))
        return observations

    def _log_debug_chunk(
        self,
        *,
        chunk_index: int,
        chunk_start: float,
        commit_start: float,
        commit_end: float,
        seg_scores: np.ndarray,
        observations: list[ChunkObservation],
        local_to_global: dict[int, int],
        debug_info: ChunkDebugInfo,
        emitted_frames: int,
    ) -> None:
        """输出 chunk 级调试信息（字段与旧版 window_summary 对齐，便于日志分析工具复用）。"""

        debug_summary = {
            "chunk_index": int(chunk_index),
            "chunk": {
                "start": round(float(chunk_start), 3),
                "commit_start": round(float(commit_start), 3),
                "end": round(float(commit_end), 3),
            },
            "segmentation_summary": {
                "shape": [int(dim) for dim in seg_scores.shape],
                "min": round(float(np.min(seg_scores)), 6),
                "max": round(float(np.max(seg_scores)), 6),
                "mean": round(float(np.mean(seg_scores)), 6),
            },
            "window_state": {
                "observations": int(len(observations)),
                "embedded": int(len(observations)),
                "emitted": int(emitted_frames),
            },
            "assignment": {
                "local_to_global": {
                    str(local_idx): int(global_id)
                    for local_idx, global_id in sorted(local_to_global.items())
                },
                "local_assignments": debug_info["local_assignments"],
            },
            "centroids": {
                "before": int(debug_info["num_centroids_before"]),
                "after": int(debug_info["num_centroids_after"]),
            },
        }
        self._log_structured(logging.DEBUG, "[debug]", "window_summary", debug_summary)

        for key in ("new_speakers", "updated_speakers", "skipped_updates"):
            if debug_info[key]:
                self._log_structured(logging.DEBUG, "[debug]", key, debug_info[key])
        if debug_info["absorb_events"]:
            self._log_structured(
                logging.DEBUG, "[debug]", "absorb_events", debug_info["absorb_events"]
            )
        if debug_info["global_speakers"]:
            self._log_structured(
                logging.INFO,
                "[runtime]",
                "current_global_speakers",
                debug_info["global_speakers"],
            )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        streaming_log_path: str,
        uri: Optional[str] = None,
    ) -> None:
        """按 chunk 顺序处理整段波形并持续写出 RTTM。"""

        waveform = resample_waveform_if_needed(
            waveform, sample_rate, self.config.sample_rate
        )
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(torch.float32)

        total_samples = waveform.shape[1]
        total_duration = total_samples / self.config.sample_rate
        chunk_samples = max(
            1, int(round(self.config.chunk_duration * self.config.sample_rate))
        )
        hop_samples = max(
            1, int(round(self.config.hop_duration * self.config.sample_rate))
        )
        # 提交区为窗口中段 hop 秒，两侧各留 margin 作为边界缓冲；
        # 第一个窗口没有左侧缓冲，直接从 0 开始提交。
        margin_duration = 0.5 * (self.config.chunk_duration - self.config.hop_duration)

        writer = AppendOnlyRTTMWriter(
            streaming_log_path,
            uri or "unknown",
            self.config.min_segment_duration,
            self.config.streaming_merge_gap,
            self.config.show_rttm,
        )

        chunk_index = 0
        chunk_start_sample = 0
        while chunk_start_sample < total_samples:
            chunk = waveform[:, chunk_start_sample : chunk_start_sample + chunk_samples]
            if chunk.shape[1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[1]))

            chunk_start = chunk_start_sample / self.config.sample_rate
            commit_start = (
                chunk_start if chunk_index == 0 else chunk_start + margin_duration
            )
            commit_end = min(
                chunk_start + margin_duration + self.config.hop_duration,
                total_duration,
            )
            if commit_start >= commit_end - 1e-9:
                break

            # 1) 局部识别。
            # 返回的 centers（帧中心时刻）当前不使用：
            # 帧时间统一由 chunk_start + frame_idx * frame_step 推得。
            seg_scores, _ = self.segmentation(chunk, self.config.sample_rate)
            if seg_scores.size == 0:
                chunk_index += 1
                chunk_start_sample += hop_samples
                continue
            # chunk 已补零到 chunk_duration，帧在窗口内均匀分布。
            frame_step = self.config.chunk_duration / seg_scores.shape[0]

            # 2) local track 聚合 + embedding。
            tracks = self.track_builder.build_tracks(
                seg_scores, frame_step, chunk_start
            )
            observations = self._embed_tracks(waveform, tracks)

            # 3) 全局分配 + 身份定案 flush。
            local_to_global, debug_info = self.clusterer.assign_chunk(observations)
            if debug_info["resolved_speakers"]:
                self._log_structured(
                    logging.INFO,
                    "[runtime]",
                    "resolved_speakers",
                    {
                        "chunk_index": int(chunk_index),
                        "resolutions": debug_info["resolved_speakers"],
                    },
                )
            # 定案 flush 先于本 chunk 帧输出：刚转正/被吸收的 speaker，
            # 其缓冲帧按 final_id 落盘，本 chunk 的新帧随后直接续接。
            for resolution in debug_info["resolved_speakers"]:
                writer.flush_speaker(
                    int(resolution["speaker_id"]), int(resolution["final_id"])
                )

            self._log_structured(
                logging.INFO,
                "[runtime]",
                "frame_decision",
                {
                    "chunk_index": int(chunk_index),
                    "chunk_start": round(float(chunk_start), 3),
                    "commit": [
                        round(float(commit_start), 3),
                        round(float(commit_end), 3),
                    ],
                    "local_to_global": {
                        str(local_idx): int(global_id)
                        for local_idx, global_id in sorted(local_to_global.items())
                    },
                },
            )

            # 4) 帧级输出（仅提交区；probationary 仅入缓冲不写出）。
            emitted_frames = writer.consume_chunk(
                seg_scores,
                frame_step,
                chunk_start,
                commit_start,
                commit_end,
                local_to_global,
                deferred_speakers=set(self.clusterer.probationary),
            )

            # 5) 闭合提交区末尾已不再活跃的 turn（沉默确认，提前写出）。
            writer.close_inactive(commit_end)

            if self.config.debug:
                self._log_debug_chunk(
                    chunk_index=chunk_index,
                    chunk_start=chunk_start,
                    commit_start=commit_start,
                    commit_end=commit_end,
                    seg_scores=seg_scores,
                    observations=observations,
                    local_to_global=local_to_global,
                    debug_info=debug_info,
                    emitted_frames=emitted_frames,
                )

            chunk_index += 1
            chunk_start_sample += hop_samples

        # 6) 残余 probationary 定案 + 缓冲 flush + 纯追加收尾（零重写）。
        resolutions = self.clusterer.finalize_redirects()
        if resolutions:
            self._log_structured(
                logging.INFO, "[runtime]", "final_resolutions", resolutions
            )
        for resolution in resolutions:
            writer.flush_speaker(
                int(resolution["speaker_id"]), int(resolution["final_id"])
            )
        writer.finalize()

    def process_file(self, wav_path: str) -> str:
        """处理单个音频文件并返回生成的 RTTM 路径。"""

        self.reset()
        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)

        if self.config.output_dir_for_streaming is None:
            raise ValueError(
                "output_dir_for_streaming must be set before processing audio"
            )
        streaming_log_path = str(
            Path(self.config.output_dir_for_streaming)
            / f"{Path(wav_path).stem}.streaming.rttm"
        )

        self.process_waveform(
            waveform,
            self.config.sample_rate,
            streaming_log_path=streaming_log_path,
            uri=Path(wav_path).stem,
        )
        return streaming_log_path


__all__ = ["ChunkDiarizationPipeline"]
