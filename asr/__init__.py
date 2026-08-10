"""ASR 转写包：Fun-ASR-Nano 段级推理（独立于 diarization pipeline 的转写阶段）。

diarization pipeline 只负责经 exporter 导出逐 speaker 音频段（wav +
manifest）；转写由 `asr.app`（根目录 transcribe.py 入口）读取输出目录逐段
完成：短段单次推理，长段按固定窗口切片续写（prev = 本段已累计文本尾部，
token 预算封顶），推理成本有界。
"""

from .app import main
from .transcriber import SegmentTranscriber


__all__ = ["main", "SegmentTranscriber"]
