"""音频处理工具。"""

from __future__ import annotations

import torch
import torchaudio


def resample_waveform_if_needed(
    waveform: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    """必要时把音频重采样到目标采样率。

    Args:
        waveform: 输入波形张量。
        orig_sr: 原始采样率。
        target_sr: 目标采样率。

    Returns:
        采样率为 target_sr 的波形；原采样率已等于目标采样率时原样返回。
    """

    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


__all__ = ["resample_waveform_if_needed"]
