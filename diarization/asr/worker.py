"""ASR 后台 worker：队列消费 exporter 推出的音频段，转写并落盘 transcript。

设计要点：

- `submit` 即 exporter 的 SegmentCallback，只入队不阻塞 chunk 循环；
- 段独立转写，不跨段传 prev_text（实测：prev_text 语义是"本段音频前缀
  的转写"，跨段文本与音频对不上时模型会判转写完成而提前 EOS）；
- 超过 asr_max_segment_duration 的长段按固定窗口 + asr_window_overlap
  重叠滑窗推理：每片 prev = 本段已累计文本尾部（token 预算封顶），模型
  把 prev 尾部与窗口重叠区对齐、只续写新内容，单次推理成本有界；拼接处
  做后缀-前缀去重兜底；
- 段到达顺序非全局时间序（长段后闭合），排序只在 finalize 落盘时做。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

import torch

from ..utils import log_structured
from .model import FunASRNanoASR


logger = logging.getLogger(__name__)

# 拼接去重的最大检测长度（字符），防止模型续写时重吐重叠区文本。
_MAX_STITCH_OVERLAP_CHARS = 50


class ASRWorker:
    """逐文件一个实例：后台线程消费音频段并转写，finalize 时落盘。"""

    def __init__(self, config, uri: str, output_dir: str):
        self.config = config
        self.uri = uri
        self.output_dir = Path(output_dir)
        self.sample_rate = int(config.sample_rate)
        self._asr = FunASRNanoASR(config)
        self._queue: queue.Queue = queue.Queue()
        # (start, end, speaker_id, text)，仅 worker 线程读写。
        self._results: list[tuple[float, float, int, str]] = []
        self._thread = threading.Thread(
            target=self._run, name=f"asr-worker-{uri}", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # 回调入口（exporter SegmentCallback）
    # ------------------------------------------------------------------

    def submit(
        self,
        uri: str,
        speaker_id: int,
        start: float,
        end: float,
        waveform: torch.Tensor,
    ) -> None:
        self._queue.put((int(speaker_id), float(start), float(end), waveform))

    # ------------------------------------------------------------------
    # worker 线程
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._handle(*item)
            except Exception:
                logger.exception("[asr] segment transcription failed")
            finally:
                self._queue.task_done()

    def _handle(
        self, speaker_id: int, start: float, end: float, waveform: torch.Tensor
    ) -> None:
        duration = end - start
        if duration <= float(self.config.asr_max_segment_duration):
            text = self._asr.transcribe(waveform)
        else:
            text = self._transcribe_long(waveform)
        text = text.strip()
        if not text:
            logger.info("[asr] empty text: spk%d [%.3f, %.3f]", speaker_id, start, end)
            return
        self._results.append((start, end, speaker_id, text))
        log_structured(
            logger,
            logging.INFO,
            "[asr]",
            "segment_done",
            {
                "speaker_id": speaker_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "text": text,
            },
        )

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

    # ------------------------------------------------------------------
    # 收尾与落盘
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """音频结束：等队列清空、线程退出，按 start 排序写 transcript。"""

        self._queue.put(None)
        self._thread.join()

        results = sorted(self._results, key=lambda r: (r[0], r[1]))
        jsonl_path = self.output_dir / f"{self.uri}.transcript.jsonl"
        txt_path = self.output_dir / f"{self.uri}.transcript.txt"
        with (
            open(jsonl_path, "w", encoding="utf-8") as jsonl_file,
            open(txt_path, "w", encoding="utf-8") as txt_file,
        ):
            for start, end, speaker_id, text in results:
                jsonl_file.write(
                    json.dumps(
                        {
                            "uri": self.uri,
                            "speaker_id": speaker_id,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                txt_file.write(f"[{start:9.3f} - {end:9.3f}] spk{speaker_id}: {text}\n")
        logger.info(
            "[asr] wrote %d segments to %s / %s",
            len(results),
            jsonl_path,
            txt_path,
        )


__all__ = ["ASRWorker"]
