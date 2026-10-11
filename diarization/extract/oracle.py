"""oracle 前端与 extract 管线的组合：PreparedChunk -> ChunkArtifacts。

`OracleFrontend`（diarization/frontend/oracle.py）只产 PreparedChunk；
本模块用调用方提供的 RegionEmbedder 提 embedding 组装 ChunkArtifacts，
进而喂 run_clustering / save_chunks（oracle 上限实验）。组合放 extract
侧（extract 依赖 frontend，反向则循环）。

`OracleChunkExtractor` 与 `ChunkExtractor` 对偶：pipeline --oracle 整管线
与提取 CLI（diarization/extract/app.py --oracle）共用它，保证两条路径的
chunk 切分与 embedding 完全一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch

from diarization.contracts import ChunkArtifacts
from diarization.speakerlab.utils.fileio import load_audio

from diarization.config import ChunkPipelineConfig
from diarization.models.embedding_infer import build_region_embedder
from diarization.utils.audio import prepare_mono_waveform

from ..frontend import ChunkTrackBuilder, OracleFrontend, PreparedChunk
from ..utils import save_chunks
from .extractor import embed_tracks


logger = logging.getLogger(__name__)


def oracle_chunk_artifacts(
    prepared_chunks: Iterator[PreparedChunk],
    waveform: torch.Tensor,
    region_embedder,
    track_builder: ChunkTrackBuilder,
) -> Iterator[ChunkArtifacts]:
    """oracle PreparedChunk + RegionEmbedder -> ChunkArtifacts（接 extract 管线）。

    embedding 提取复用 embed_tracks，与实时前端路径行为一致。
    region_embedder 由调用方构建（如
    diarization.models.embedding_infer.build_region_embedder）。

    Args:
        prepared_chunks: OracleFrontend.iter_prepared_chunks 的输出。
        waveform: 整段波形，形状 (1, samples)。
        region_embedder: 调用方提供的 RegionEmbedder。
        track_builder: 提供 to_observation 的 ChunkTrackBuilder。

    Yields:
        ChunkArtifacts: 可直接喂 run_clustering / save_chunks。
    """

    for prepared in prepared_chunks:
        observations = embed_tracks(
            region_embedder, track_builder, waveform, prepared.tracks
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


def resolve_ref_rttm(ref: Optional[str], audio_path: str) -> str:
    """解析音频对应的参考 RTTM 路径（oracle 模式）。

    Args:
        ref: RTTM 文件或含 {uri}.rttm 的目录。
        audio_path: 音频路径（uri 取其文件名主干）。

    Returns:
        参考 RTTM 路径。

    Raises:
        ValueError: ref 未提供，或按目录解析的 RTTM 不存在。
    """

    if ref is None:
        raise ValueError("oracle 模式需要 --ref 指定参考 RTTM（文件或目录）")
    ref_path = Path(ref)
    if ref_path.is_dir():
        candidate = ref_path / f"{Path(audio_path).stem}.rttm"
        if not candidate.is_file():
            raise ValueError(f"参考 RTTM 不存在: {candidate}")
        return str(candidate)
    return str(ref_path)


class OracleChunkExtractor:
    """oracle 模式的 chunk 提取器：ref RTTM 前端 + RegionEmbedder。

    与 `ChunkExtractor` 对偶：extract_file 落盘
    <output_dir>/embeddings/<stem>.chunks.npz，供聚类阶段
    （diarization/cluster/app.py）与参数扫描（tools/sweep_params.py）复用。

    Attributes:
        config: 管线配置。
        oracle_frontend: ref RTTM 驱动的前端（无 segmentation 模型）。
        region_embedder: 区间 embedding 提取器。
        track_builder: local track -> observation 构建器。
    """

    def __init__(self, config: ChunkPipelineConfig, model_path: Optional[str] = None):
        """加载 embedding 模型并初始化 oracle 前端。

        Args:
            config: 管线配置。
            model_path: 显式模型路径覆盖。
        """

        self.config = config
        self.oracle_frontend = OracleFrontend(config)
        self.region_embedder = build_region_embedder(config, model_path)
        self.track_builder = ChunkTrackBuilder(config)

    def iter_chunk_artifacts(
        self, waveform: torch.Tensor, ref_rttm: str
    ) -> Iterator[ChunkArtifacts]:
        """prepare 波形并生产逐 chunk 的 ChunkArtifacts（含 embedding）。

        Args:
            waveform: load_audio 读出的原始波形。
            ref_rttm: 该音频对应的参考 RTTM 路径。

        Yields:
            ChunkArtifacts: 可直接喂 run_clustering / save_chunks。
        """

        waveform = prepare_mono_waveform(
            waveform, self.config.sample_rate, self.config.sample_rate
        )
        return oracle_chunk_artifacts(
            self.oracle_frontend.iter_prepared_chunks(waveform, ref_rttm),
            waveform,
            self.region_embedder,
            self.track_builder,
        )

    def extract_file(self, wav_path: str, ref_rttm: str) -> str:
        """提取阶段入口（oracle）：生产 chunk artifacts 并落盘为 <stem>.chunks.npz。

        Args:
            wav_path: 音频文件路径。
            ref_rttm: 该音频对应的参考 RTTM 路径。

        Returns:
            落盘的 npz 路径（<output_dir>/embeddings/<stem>.chunks.npz）。

        Raises:
            ValueError: config.output_dir 未设置时。
        """

        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)
        if self.config.output_dir is None:
            raise ValueError("output_dir must be set before extracting chunks")
        uri = Path(wav_path).stem
        chunks_path = str(
            Path(self.config.output_dir) / "embeddings" / f"{uri}.chunks.npz"
        )

        artifacts = list(self.iter_chunk_artifacts(waveform, ref_rttm))
        save_chunks(chunks_path, uri, artifacts)
        logger.info("[extract] saved %d chunks to %s", len(artifacts), chunks_path)
        return chunks_path


__all__ = ["OracleChunkExtractor", "oracle_chunk_artifacts", "resolve_ref_rttm"]
