"""语音分离推理封装。"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


class TIGERSeparator:
    """TIGER语音分离模型封装。"""

    def __init__(self, model_name: str, cache_dir: str, device: torch.device):
        import look2hear.models

        os.makedirs(cache_dir, exist_ok=True)
        self.model = look2hear.models.TIGER.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = 16000

    @torch.no_grad()
    def separate(self, waveform: torch.Tensor) -> torch.Tensor:
        orig_len = waveform.shape[-1]
        target_len = int(self.sample_rate * 3)
        if orig_len < target_len:
            waveform = F.pad(waveform, (0, target_len - orig_len))

        audio = waveform.to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_input = audio.unsqueeze(0)

        ests_speech = self.model(audio_input).squeeze(0)
        return ests_speech[:, :orig_len].cpu()


__all__ = ["TIGERSeparator"]
