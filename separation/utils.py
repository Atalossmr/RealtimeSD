"""separation 子包的音频切片与能量工具（纯函数）。"""

from __future__ import annotations

from typing import Optional

import torch

# 音频来源：(波形 [1, T], 该波形零点对应的绝对时刻)。
AudioSource = tuple[torch.Tensor, float]


def slice_absolute(
    source: torch.Tensor,
    origin: float,
    start: float,
    end: float,
    sample_rate: int,
) -> Optional[torch.Tensor]:
    """从 source（零点对应绝对时刻 origin）裁出 [start, end)，越界截断。

    Args:
        source: 源波形（[1, T]）。
        origin: source 零点对应的绝对时刻（秒）。
        start: 裁剪起点绝对时刻（秒）。
        end: 裁剪终点绝对时刻（秒）。
        sample_rate: 采样率。

    Returns:
        裁出的波形（[1, T]）；区间为空（越界截断后无样本）时返回 None。
    """

    total = source.shape[1]
    start_sample = max(
        0, min(int(round((start - origin) * sample_rate)), total)
    )
    end_sample = max(
        start_sample, min(int(round((end - origin) * sample_rate)), total)
    )
    if end_sample <= start_sample:
        return None
    return source[:, start_sample:end_sample]


def rms(waveform: torch.Tensor) -> float:
    """计算波形的 RMS 能量（加 1e-12 避免全零时恰为 0）。

    Args:
        waveform: 波形张量。

    Returns:
        RMS 值；空张量返回 0.0。
    """
    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.float() ** 2))) + 1e-12


def rms_over_regions(
    source: torch.Tensor,
    origin: float,
    regions: list[tuple[float, float]],
    sample_rate: int,
) -> float:
    """计算 source 在多个绝对时刻区间上的整体 RMS。

    Args:
        source: 源波形（[1, T]）。
        origin: source 零点对应的绝对时刻（秒）。
        regions: (start, end) 绝对时刻区间列表。
        sample_rate: 采样率。

    Returns:
        各区间切片拼接后的 RMS；所有区间均为空时返回 0.0。
    """
    pieces = [
        piece
        for start, end in regions
        if (piece := slice_absolute(source, origin, start, end, sample_rate))
        is not None
    ]
    if not pieces:
        return 0.0
    return rms(torch.cat(pieces, dim=1))
