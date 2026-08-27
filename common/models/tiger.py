"""TIGER 语音分离的惰性封装：首次调用时才加载模型。"""

from __future__ import annotations

import logging
from typing import Optional

import torch


logger = logging.getLogger(__name__)


class TigerSeparator:
    """commit 区音频的 2 路分离（TIGER，固定 2 路输出、16kHz）。"""

    def __init__(self, model_name: str, hf_cache_dir: str, sample_rate: int, device):
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir
        self.sample_rate = sample_rate
        self.device = device
        # TIGER 惰性加载：全程无重叠时不付出模型加载开销。
        self._tiger = None

    def _load(self) -> None:
        import look2hear.models

        logger.info("[separation] loading TIGER model %s", self.model_name)
        model = look2hear.models.TIGER.from_pretrained(
            self.model_name, cache_dir=self.hf_cache_dir
        )
        model.to(self.device)
        model.eval()
        self._tiger = model

    def separate(
        self, mix: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """分离一段混合波形 [1, T]，返回 (混合波形, 分离音轨 [2, T])，长度已对齐。

        波形切片由调用方完成（见 separation/exporter.py）；过短的输入返回 None。
        """

        if self._tiger is None:
            self._load()
        if mix.shape[1] < self.sample_rate // 4:
            return None
        audio_input = mix.unsqueeze(0).to(self.device)  # [1, 1, T]
        with torch.inference_mode():
            ests = self._tiger(audio_input)  # [1, num_spk, T]
        tracks = ests.squeeze(0).cpu()
        if tracks.ndim == 1:
            tracks = tracks.unsqueeze(0)
        aligned = min(tracks.shape[-1], mix.shape[-1])
        return mix[:, :aligned], tracks[:, :aligned]
