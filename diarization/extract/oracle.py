"""oracle 前端与 extract 管线的组合：PreparedChunk -> ChunkArtifacts。

`OracleFrontend`（diarization/frontend/oracle.py）只产 PreparedChunk；
本模块用调用方提供的 RegionEmbedder 提 embedding 组装 ChunkArtifacts，
进而喂 run_clustering / save_chunks（oracle 上限实验）。组合放 extract
侧（extract 依赖 frontend，反向则循环）。
"""

from __future__ import annotations

from typing import Iterator

import torch

from diarization.contracts import ChunkArtifacts

from ..frontend import ChunkTrackBuilder, PreparedChunk
from .extractor import embed_tracks


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


__all__ = ["oracle_chunk_artifacts"]
