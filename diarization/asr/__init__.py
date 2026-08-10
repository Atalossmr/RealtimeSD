"""ASR 转写模块：Fun-ASR-Nano 段级推理 + 后台 worker 线程。

exporter 闭合的逐 speaker 音频段经 `ASRWorker.submit` 入队，worker 线程
逐段调 Fun-ASR-Nano 识别，prev_text 取全局已确认段中时间上最近的若干段
文本（整段取舍、token 预算封顶）；长段按固定窗口切片续写，推理成本有界。
"""

from .worker import ASRWorker


__all__ = ["ASRWorker"]
