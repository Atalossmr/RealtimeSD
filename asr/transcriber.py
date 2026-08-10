"""段级转写器：短段单次推理，长段滑窗续写（无线程/队列，纯计算封装）。

设计要点：

- 段独立转写，不跨段传 prev_text（实测：prev_text 语义是"本段音频前缀
  的转写"，跨段文本与音频对不上时模型会判转写完成而提前 EOS）；
- 超过 asr_max_segment_duration 的长段按固定窗口 + asr_window_overlap
  重叠滑窗推理：每片 prev = 本段已累计文本尾部（token 预算封顶），模型
  把 prev 尾部与窗口重叠区对齐、只续写新内容，单次推理成本有界；拼接处
  做后缀-前缀去重兜底。
"""

from __future__ import annotations

import torch

from .model import FunASRNanoASR


# 拼接去重的最大检测长度（字符），防止模型续写时重吐重叠区文本。
_MAX_STITCH_OVERLAP_CHARS = 50


class SegmentTranscriber:
    """逐段转写器：整个文件/目录共用一个实例（模型惰性加载一次）。"""

    def __init__(self, config):
        self.config = config
        self.sample_rate = int(config.sample_rate)
        self._asr = FunASRNanoASR(config)

    def warmup(self) -> None:
        """立即加载模型（默认惰性到首次转写；跟随模式下提前加载以便与
        管线启动阶段重叠，首个音频段闭合即可开始转写）。"""

        self._asr._load()

    def transcribe_segment(self, waveform: torch.Tensor) -> str:
        """转写一段音频 [1, T]（16k 单声道），返回文本（空串表示无内容）。"""

        duration = waveform.shape[1] / self.sample_rate
        if duration <= float(self.config.asr_max_segment_duration):
            return self._asr.transcribe(waveform).strip()
        return self._transcribe_long(waveform).strip()

    @staticmethod
    def _strip_stitch_overlap(accumulated: str, new_text: str) -> str:
        """去掉 new_text 开头与 accumulated 结尾重复的部分（续写重吐兜底）。"""

        max_k = min(len(accumulated), len(new_text), _MAX_STITCH_OVERLAP_CHARS)
        for k in range(max_k, 0, -1):
            if accumulated.endswith(new_text[:k]):
                return new_text[k:]
        return new_text

    def _transcribe_long(self, waveform: torch.Tensor) -> str:
        """长段滑窗推理：每次推进 step 秒，输入窗口 = 已转写前缀尾部 overlap
        秒 + 新增 step 秒（总长 ≤ W），prev = 已累计文本尾部。

        模型把 prev 尾部与窗口头部的重叠区对齐、只续写新增部分；prev 覆盖的
        音频必须是输入窗口的前缀，因此窗口不能从"新内容"处开始，必须带上
        overlap 秒的已转写前缀。
        """

        window = int(float(self.config.asr_max_segment_duration) * self.sample_rate)
        overlap = int(float(self.config.asr_window_overlap) * self.sample_rate)
        step = window - overlap  # 每次推进的新增音频（采样点）
        if step <= 0:
            raise ValueError(
                "asr_window_overlap must be smaller than asr_max_segment_duration"
            )
        budget = int(self.config.asr_prev_text_max_tokens)
        if budget <= 0:
            # 自动预算：prev 只需覆盖窗口头部的 overlap 区，按中文会话
            # ~4 token/s 估计。预算过大（prev 估计覆盖 ≥ 窗口音频总长）
            # 会触发模型判转写完成而提前 EOS（实测确认）。
            budget = max(16, int(overlap / self.sample_rate * 4))
        total = waveform.shape[1]
        accumulated = ""
        confirmed_end = 0  # 已转写覆盖到的采样点位置（近似）
        while confirmed_end < total:
            audio_start = max(0, confirmed_end - overlap)
            audio_end = min(confirmed_end + step, total)
            prev = self._asr.tail_text(accumulated, budget) if accumulated else ""
            piece_text = self._asr.transcribe(waveform[:, audio_start:audio_end], prev)
            piece_text = self._strip_stitch_overlap(accumulated, piece_text)
            accumulated += piece_text
            confirmed_end += step
        return accumulated


__all__ = ["SegmentTranscriber"]
