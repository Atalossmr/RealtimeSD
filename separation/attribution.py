"""身份确认（attribution）：分离音轨与 speaker 的归属判定纯函数。

全部为无状态函数，阈值与上下文由调用方传入：

- `overlap_regions`：重叠帧归并连续区间；
- `cosine`：余弦相似度；
- `energy_gate`：重叠帧区间上 音轨RMS/混合RMS 门控（含 energy_gate 结构化日志）；
- `match_pair`：2x2 embedding 互斥匹配（含 pair_match 结构化日志）；
- `best_match`：单路 winner 匹配（一路过门控时的归属判定）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from common.utils import log_structured

from .rttm import FrameActivity
from .utils import AudioSource, rms_over_regions


logger = logging.getLogger(__name__)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两向量的余弦相似度。

    Args:
        a: 向量 a。
        b: 向量 b。

    Returns:
        余弦相似度（零向量按 1e-12 兜底避免除零）。
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def overlap_regions(
    overlap_frames: list[FrameActivity],
) -> list[tuple[float, float]]:
    """把重叠帧合成连续区间（时间上相邻的帧归并为一个区间）。

    Args:
        overlap_frames: 重叠帧的帧级活跃记录（按时间有序）。

    Returns:
        归并后的 (start, end) 区间列表。
    """

    regions: list[tuple[float, float]] = []
    for frame_start, frame_end, _ in overlap_frames:
        if regions and frame_start <= regions[-1][1] + 1e-6:
            regions[-1] = (regions[-1][0], frame_end)
        else:
            regions.append((frame_start, frame_end))
    return regions


def energy_gate(
    mix: torch.Tensor,
    track_sources: list[AudioSource],
    regions: list[tuple[float, float]],
    sample_rate: int,
    threshold: float,
    mix_origin: float,
    window: tuple[float, float],
    window_index: int,
) -> list[int]:
    """能量门控：重叠帧区间上 音轨RMS / 混合RMS，返回过门控的 track 下标列表。

    门控事件写 energy_gate 结构化日志（含各轨 ratio 与是否过门控）。

    Args:
        mix: 混合波形（[1, T]，零点对应绝对时刻 mix_origin）。
        track_sources: 各分离音轨的 (波形, 零点绝对时刻)。
        regions: 重叠帧归并后的 (start, end) 区间列表（绝对时刻）。
        sample_rate: 采样率。
        threshold: 过门控的 RMS 比值阈值。
        mix_origin: mix 零点对应的绝对时刻（秒），用于定位区间切片。
        window: 本分离窗口的 (start, end) 绝对时刻（仅用于日志定位）。
        window_index: 本分离窗口序号（仅用于日志）。

    Returns:
        过门控的音轨下标列表（按 track 顺序）。
    """

    mix_rms = rms_over_regions(mix, mix_origin, regions, sample_rate)
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
            "window_index": int(window_index),
            "window": [round(float(window[0]), 3), round(float(window[1]), 3)],
            "mix_rms": round(mix_rms, 6),
            "tracks": gate_tracks,
            "threshold": float(threshold),
        },
    )
    return passed


def match_pair(
    window_index: int,
    track_embeddings: list[np.ndarray],
    candidates: list[int],
    refs: dict[int, np.ndarray],
    min_sim: float,
) -> Optional[dict[int, int]]:
    """2x2 互斥匹配：返回 track_idx -> speaker_id；匹配不可靠时返回 None。

    两种一一对应方案取总相似度较高者；任一对的相似度低于 min_sim
    （min(pair_sims) < min_sim）或候选缺参照 embedding 时判不可靠。
    匹配事件写 pair_match 结构化日志。

    Args:
        window_index: 本分离窗口序号（仅用于日志）。
        track_embeddings: 两路分离音轨的 embedding（按下标对应 track 0/1）。
        candidates: 两个候选 speaker id。
        refs: speaker id -> 参照 embedding。
        min_sim: 单对匹配的最小相似度阈值。

    Returns:
        track 下标 -> speaker id 的映射；不可靠时返回 None。
    """

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
            "window_index": int(window_index),
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
    """单路 winner 匹配：返回 (winner_speaker, similarity)；无可用参照返回 None。

    Args:
        embedding: 过门控音轨的 embedding。
        candidates: 候选 speaker id 列表。
        refs: speaker id -> 参照 embedding。

    Returns:
        (相似度最高的 speaker id, 相似度)；候选均无参照时返回 None。
    """

    sims = {g: cosine(embedding, refs[g]) for g in candidates if g in refs}
    if not sims:
        return None
    winner_speaker = max(sims, key=sims.get)
    return winner_speaker, sims[winner_speaker]


__all__ = [
    "FrameActivity",
    "cosine",
    "overlap_regions",
    "energy_gate",
    "match_pair",
    "best_match",
]
