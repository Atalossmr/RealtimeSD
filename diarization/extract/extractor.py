"""嵌入提取阶段：音频 -> 逐 chunk 的 ChunkArtifacts（含 embedding）。

`ChunkExtractor` 是 chunk 生产的唯一来源：端到端流程（diarization/pipeline.py）与
提取 CLI（extract/app.py）共用它，保证两个阶段的 chunk 切分、
segmentation、track 聚合与 embedding 完全一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from speakerlab.process.processor import FBank
from speakerlab.utils.fileio import load_audio

from ..utils import save_chunks
from ..config import ChunkPipelineConfig
from ..schema import ChunkArtifacts, ChunkObservation
from ..utils import resample_waveform_if_needed, resolve_device
from .models.embedding_infer import (
    NativeERes2NetV2SegmentEmbedder,
    load_embedding_model,
)
from .models.segmentation_infer import PyannoteStreamingSegmentation
from .track_builder import ChunkTrackBuilder, LocalTrack


logger = logging.getLogger(__name__)


class ChunkExtractor:
    """chunk 级嵌入提取器：segmentation-3.0 局部识别 + ERes2NetV2 embedding。"""

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

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """重采样到目标采样率、转单声道 float32。"""

        waveform = resample_waveform_if_needed(
            waveform, sample_rate, self.config.sample_rate
        )
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.to(torch.float32)

    def iter_chunk_artifacts(self, waveform: torch.Tensor):
        """按 chunk 顺序生产 ChunkArtifacts。"""

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

            yield ChunkArtifacts(
                chunk_index=chunk_index,
                seg_scores=seg_scores,
                frame_step=frame_step,
                chunk_start=chunk_start,
                commit_start=commit_start,
                commit_end=commit_end,
                observations=observations,
            )

            chunk_index += 1
            chunk_start_sample += hop_samples

    def extract_file(self, wav_path: str) -> str:
        """提取阶段入口：生产 chunk artifacts 并落盘为 <stem>.chunks.npz。"""

        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)
        if self.config.output_dir_for_streaming is None:
            raise ValueError(
                "output_dir_for_streaming must be set before extracting chunks"
            )
        uri = Path(wav_path).stem
        chunks_path = str(
            Path(self.config.output_dir_for_streaming) / f"{uri}.chunks.npz"
        )

        waveform = self.prepare_waveform(waveform, self.config.sample_rate)
        artifacts = list(self.iter_chunk_artifacts(waveform))
        save_chunks(chunks_path, uri, artifacts)
        logger.info("[extract] saved %d chunks to %s", len(artifacts), chunks_path)
        return chunks_path


__all__ = ["ChunkExtractor"]
