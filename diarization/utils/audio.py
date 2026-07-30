"""音频处理工具。"""

from __future__ import annotations

import torch
import torchaudio


def resample_waveform_if_needed(
    waveform: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    """必要时把音频重采样到目标采样率。"""

    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


__all__ = ["resample_waveform_if_needed"]
