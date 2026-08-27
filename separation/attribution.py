"""身份确认（attribution）：分离音轨与 global speaker 的归属判定纯函数。

从 exporter.py 拆出，全部为无状态函数，阈值与上下文由调用方（exporter）传入：

- `overlap_regions`：重叠帧归并连续区间；
- `cosine` / `reference_embeddings`：相似度与参照 embedding；
- `energy_gate`：重叠帧区间上 音轨RMS/混合RMS 门控（含 energy_gate 结构化日志）；
- `match_pair`：2x2 embedding 互斥匹配（含 pair_match 结构化日志）；
- `best_match`：单路 winner 匹配（一路过门控时的归属判定）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from common.contracts import ChunkArtifacts
from common.utils import log_structured

from .utils import AudioSource, rms_over_regions


logger = logging.getLogger(__name__)

# 帧级活跃记录：(frame_start, frame_end, active_global_ids)。
FrameActivity = tuple[float, float, list[int]]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def overlap_regions(
    overlap_frames: list[FrameActivity],
) -> list[tuple[float, float]]:
    """把重叠帧合成连续区间（时间上相邻的帧归并为一个区间）。"""

    regions: list[tuple[float, float]] = []
    for frame_start, frame_end, _ in overlap_frames:
        if regions and frame_start <= regions[-1][1] + 1e-6:
            regions[-1] = (regions[-1][0], frame_end)
        else:
            regions.append((frame_start, frame_end))
    return regions


def reference_embeddings(
    chunk: ChunkArtifacts, local_to_global: dict[int, int]
) -> dict[int, np.ndarray]:
    """各 global speaker 的参照 embedding：候选 speaker 用本 chunk 的观测。

    按 clusterer 契约，local_to_global 中的 slot 必有带 embedding 的
    observation（提不出 embedding 的 slot 不会进入分配），因此候选
    一定能被 observation 覆盖。
    """

    refs: dict[int, np.ndarray] = {}
    for obs in chunk.observations:
        global_id = local_to_global.get(obs.local_idx)
        if global_id is not None and obs.embedding is not None:
            refs[int(global_id)] = obs.embedding
    return refs


def energy_gate(
    mix: torch.Tensor,
    track_sources: list[AudioSource],
    regions: list[tuple[float, float]],
    sample_rate: int,
    threshold: float,
    chunk_ctx: ChunkArtifacts,
) -> list[int]:
    """能量门控：重叠帧区间上 音轨RMS / 混合RMS，返回过门控的 track 下标列表。

    mix 为零点对应 chunk_ctx.commit_start 的混合波形；门控事件写
    energy_gate 结构化日志（payload 字段与 exporter 内联实现一致）。
    """

    mix_rms = rms_over_regions(mix, chunk_ctx.commit_start, regions, sample_rate)
    passed: list[int] = []
    gate_tracks: list[dict] = []
    for track_idx, (track, origin) in enumerate(track_sources):
        ratio = rms_over_regions(track, origin, regions, sample_rate) / mix_rms
        track_passed = ratio >= threshold
        if track_passed:
            passed.append(track_idx)
        gate_tracks.append(
            {
                "track": int(track_idx),
                "ratio": round(ratio, 4),
                "passed": bool(track_passed),
            }
        )
    log_structured(
        logger,
        logging.INFO,
        "[separation]",
        "energy_gate",
        {
            "chunk_index": int(chunk_ctx.chunk_index),
            "commit": [
                round(float(chunk_ctx.commit_start), 3),
                round(float(chunk_ctx.commit_end), 3),
            ],
            "mix_rms": round(mix_rms, 6),
            "tracks": gate_tracks,
            "threshold": float(threshold),
        },
    )
    return passed


def match_pair(
    chunk_index: int,
    track_embeddings: list[np.ndarray],
    candidates: list[int],
    refs: dict[int, np.ndarray],
    min_sim: float,
) -> Optional[dict[int, int]]:
    """2x2 互斥匹配：返回 track_idx -> global_id；总相似度过低返回 None。"""

    g0, g1 = candidates
    if g0 not in refs or g1 not in refs:
        return None
    s00 = cosine(track_embeddings[0], refs[g0])
    s01 = cosine(track_embeddings[0], refs[g1])
    s10 = cosine(track_embeddings[1], refs[g0])
    s11 = cosine(track_embeddings[1], refs[g1])
    if s00 + s11 >= s01 + s10:
        mapping, pair_sims = {0: g0, 1: g1}, (s00, s11)
    else:
        mapping, pair_sims = {0: g1, 1: g0}, (s01, s10)
    accepted = min(pair_sims) >= min_sim
    log_structured(
        logger,
        logging.INFO,
        "[separation]",
        "pair_match",
        {
            "chunk_index": int(chunk_index),
            "candidates": [int(g0), int(g1)],
            "sims": {
                "track0": [round(s00, 4), round(s01, 4)],
                "track1": [round(s10, 4), round(s11, 4)],
            },
            "mapping": {"track0": int(mapping[0]), "track1": int(mapping[1])},
            "min_sim": round(min(pair_sims), 4),
            "threshold": float(min_sim),
            "accepted": bool(accepted),
        },
    )
    if not accepted:
        return None
    return mapping


def best_match(
    embedding: np.ndarray,
    candidates: list[int],
    refs: dict[int, np.ndarray],
) -> Optional[tuple[int, float]]:
    """单路 winner 匹配：返回 (winner_global, similarity)；无可用参照返回 None。"""

    sims = {g: cosine(embedding, refs[g]) for g in candidates if g in refs}
    if not sims:
        return None
    winner_global = max(sims, key=sims.get)
    return winner_global, sims[winner_global]


__all__ = [
    "FrameActivity",
    "cosine",
    "overlap_regions",
    "reference_embeddings",
    "energy_gate",
    "match_pair",
    "best_match",
]
