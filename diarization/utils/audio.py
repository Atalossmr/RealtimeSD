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


def prepare_mono_waveform(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int
) -> torch.Tensor:
    """重采样到目标采样率、转单声道 float32。

    Args:
        waveform: 原始波形，一维或 (channels, samples)。
        sample_rate: 原始采样率。
        target_sample_rate: 目标采样率。

    Returns:
        形状 (1, samples) 的 float32 单声道波形，采样率为
        target_sample_rate。
    """

    waveform = resample_waveform_if_needed(waveform, sample_rate, target_sample_rate)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(torch.float32)


__all__ = ["resample_waveform_if_needed", "prepare_mono_waveform"]
