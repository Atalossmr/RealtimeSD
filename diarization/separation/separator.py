"""TIGER 语音分离的惰性封装：首次调用时才加载模型。"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from ..config import ChunkPipelineConfig
from .audio import slice_absolute


logger = logging.getLogger(__name__)


class TigerSeparator:
    """commit 区音频的 2 路分离（TIGER，固定 2 路输出、16kHz）。"""

    def __init__(self, config: ChunkPipelineConfig, sample_rate: int, device):
        self.config = config
        self.sample_rate = sample_rate
        self.device = device
        # TIGER 惰性加载：全程无重叠时不付出模型加载开销。
        self._tiger = None

    def _load(self) -> None:
        import look2hear.models

        logger.info(
            "[separation] loading TIGER model %s", self.config.separation_model
        )
        model = look2hear.models.TIGER.from_pretrained(
            self.config.separation_model, cache_dir=self.config.hf_cache_dir
        )
        model.to(self.device)
        model.eval()
        self._tiger = model

    def separate(
        self, waveform: torch.Tensor, commit_start: float, commit_end: float
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """分离 commit 区音频，返回 (混合波形, 分离音轨 [2, T])，长度已对齐。"""

        if self._tiger is None:
            self._load()
        mix = slice_absolute(waveform, 0.0, commit_start, commit_end, self.sample_rate)
        if mix is None or mix.shape[1] < self.sample_rate // 4:
            return None
        audio_input = mix.unsqueeze(0).to(self.device)  # [1, 1, T]
        with torch.inference_mode():
            ests = self._tiger(audio_input)  # [1, num_spk, T]
        tracks = ests.squeeze(0).cpu()
        if tracks.ndim == 1:
            tracks = tracks.unsqueeze(0)
        aligned = min(tracks.shape[-1], mix.shape[-1])
        return mix[:, :aligned], tracks[:, :aligned]
