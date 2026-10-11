"""嵌入提取阶段组合层：音频 -> 逐 chunk 的 ChunkArtifacts（含 embedding）。

`ChunkExtractor` 是 chunk 生产的唯一来源：端到端组合层（pipeline 的
`run_diarization`）与提取 CLI（diarization/extract/app.py）共用它，保证
两个阶段的 chunk 切分、segmentation、track 聚合与 embedding 完全一致。

内部为两层组合：
- 前端 `ChunkFrontend`（diarization/frontend/，与 extract 平级）：
  滑窗调度 + segmentation + track 构建；
- 提取器 `RegionEmbedder`（diarization/models/embedding_infer.py）：
  波形 + 区间 -> embedding，oracle 模式（pipeline --oracle）亦复用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch

from diarization.speakerlab.utils.fileio import load_audio

from diarization.config import ChunkPipelineConfig
from diarization.contracts import ChunkArtifacts, ChunkObservation
from diarization.models.embedding_infer import RegionEmbedder, build_region_embedder
from diarization.utils.audio import prepare_mono_waveform
from diarization.utils.device import resolve_device

from ..frontend import ChunkFrontend, LocalTrack
from ..utils import save_chunks


logger = logging.getLogger(__name__)


def embed_tracks(
    region_embedder: RegionEmbedder,
    track_builder,
    waveform: torch.Tensor,
    tracks: list[LocalTrack],
) -> list[ChunkObservation]:
    """批量为各 track 提 embedding 并组装 observation。

    裁不出波形的 track 跳过（不产 observation）；退化 embedding 由
    RegionEmbedder 记为 None（保留 observation，交由下游按"无
    embedding"处理）。oracle 前端路径（diarization/extract/oracle.py）
    与 ChunkExtractor 共用本函数，保证两条路径的 embedding 行为一致。

    Args:
        region_embedder: 区域 embedding 提取器。
        track_builder: 提供 to_observation 的 track 构造器。
        waveform: 整段波形（非当前 chunk），形状 (channels, samples)。
        tracks: 当前 chunk 的 local track 列表。

    Returns:
        与有效 track 一一对应的 ChunkObservation 列表。
    """

    segments: list[torch.Tensor] = []
    pending: list[LocalTrack] = []
    for track in tracks:
        # 从整段波形（而非当前 chunk）裁剪：track 区域可能落在提交区外的
        # margin 里，但都在本 chunk 覆盖的时间范围内。
        segment = region_embedder.slice_regions(waveform, track.regions)
        if segment is None or segment.shape[1] <= 0:
            continue
        segments.append(segment)
        pending.append(track)

    return [
        track_builder.to_observation(track, embedding)
        for track, embedding in zip(
            pending, region_embedder.embed_waveforms(segments)
        )
    ]


class ChunkExtractor:
    """chunk 级嵌入提取器：前端 + RegionEmbedder 的组合 facade。"""

    def __init__(
        self,
        config: ChunkPipelineConfig,
        embedding_model_path: Optional[str],
    ):
        """初始化前端与区域 embedding 提取器。

        Args:
            config: chunk 管线配置。
            embedding_model_path: embedding 模型权重路径；为 None 时按
                model_type 解析默认路径。
        """

        self.config = config
        self.device = resolve_device(config.device)
        self.frontend = ChunkFrontend(config)
        self.region_embedder: RegionEmbedder = build_region_embedder(
            config, embedding_model_path
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def prepare_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """重采样到目标采样率、转单声道 float32。

        Args:
            waveform: 原始波形，一维或 (channels, samples)。
            sample_rate: 原始采样率。

        Returns:
            形状 (1, samples) 的 float32 单声道波形，采样率为
            config.sample_rate。
        """

        return prepare_mono_waveform(waveform, sample_rate, self.config.sample_rate)

    def iter_chunk_artifacts(
        self, waveform: torch.Tensor
    ) -> Iterator[ChunkArtifacts]:
        """按 chunk 顺序生产 ChunkArtifacts。

        Args:
            waveform: 单声道波形，形状 (1, samples)，采样率为
                config.sample_rate。

        Yields:
            ChunkArtifacts: 逐 chunk 的 segmentation 分数与 observation。
        """

        for prepared in self.frontend.iter_prepared_chunks(waveform):
            observations = embed_tracks(
                self.region_embedder,
                self.frontend.track_builder,
                waveform,
                prepared.tracks,
            )
            yield ChunkArtifacts(
                chunk_index=prepared.chunk_index,
                seg_scores=prepared.seg_scores,
                frame_step=prepared.frame_step,
                chunk_start=prepared.chunk_start,
                commit_start=prepared.commit_start,
                commit_end=prepared.commit_end,
                observations=observations,
            )

    def extract_file(self, wav_path: str) -> str:
        """提取阶段入口：生产 chunk artifacts 并落盘为 <stem>.chunks.npz。

        Args:
            wav_path: 音频文件路径。

        Returns:
            落盘的 npz 路径（<output_dir>/embeddings/<stem>.chunks.npz）。

        Raises:
            ValueError: config.output_dir 未设置时。
        """

        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)
        if self.config.output_dir is None:
            raise ValueError(
                "output_dir must be set before extracting chunks"
            )
        uri = Path(wav_path).stem
        chunks_path = str(
            Path(self.config.output_dir)
            / "embeddings"
            / f"{uri}.chunks.npz"
        )

        waveform = self.prepare_waveform(waveform, self.config.sample_rate)
        artifacts = list(self.iter_chunk_artifacts(waveform))
        save_chunks(chunks_path, uri, artifacts)
        logger.info("[extract] saved %d chunks to %s", len(artifacts), chunks_path)
        return chunks_path


__all__ = ["ChunkExtractor", "embed_tracks"]
