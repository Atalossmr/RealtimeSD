"""chunk 滑窗调度：commit 区几何计算（frontend 与 oracle frontend 共用）。

滑窗按 hop_duration 步进，每个窗口只提交中段 hop 秒（两侧留 margin 作
边界缓冲；首窗左 margin 一并提交，尾窗截到音频末尾）。本模块只算窗口
几何，不碰音频与模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from common.config import ChunkPipelineConfig


@dataclass
class ChunkWindow:
    """一个滑窗窗口的时间几何。

    Attributes:
        index: chunk 序号。
        start_sample: 窗口起点采样点下标。
        num_samples: 窗口长度（采样点，= chunk_duration * sample_rate）。
        start: 窗口起点（绝对秒）。
        commit_start: 提交区起点（绝对秒）。
        commit_end: 提交区终点（绝对秒）。
    """

    index: int
    start_sample: int
    num_samples: int
    start: float
    commit_start: float
    commit_end: float


def iter_chunk_schedule(
    total_samples: int, config: ChunkPipelineConfig
) -> Iterator[ChunkWindow]:
    """按 chunk 顺序产出窗口几何。

    滑窗按 hop_duration 步进，每个窗口只提交中段 hop 秒（两侧留 margin
    作边界缓冲；首窗左 margin 一并提交，尾窗截到音频末尾）；相邻窗口的
    提交区无缝拼接、互不重叠。音频尾部剩余不足一个 margin 时提前结束
    （最后一个窗口的提交区为空，不产出）。

    Args:
        total_samples: 音频总采样点数。
        config: chunk 管线配置（取 sample_rate/chunk_duration/hop_duration）。

    Yields:
        ChunkWindow: 逐窗口的时间几何。
    """

    total_duration = total_samples / config.sample_rate
    chunk_samples = max(1, int(round(config.chunk_duration * config.sample_rate)))
    hop_samples = max(1, int(round(config.hop_duration * config.sample_rate)))
    # 提交区为窗口中段 hop 秒，两侧各留 margin 作为边界缓冲；
    # 第一个窗口没有左侧缓冲，直接从 0 开始提交。
    margin_duration = 0.5 * (config.chunk_duration - config.hop_duration)

    chunk_index = 0
    chunk_start_sample = 0
    while chunk_start_sample < total_samples:
        chunk_start = chunk_start_sample / config.sample_rate
        # 提交区 = 窗口中段 hop 秒。首窗没有左邻窗，左侧 margin 也一并提交，
        # 否则音频开头 (chunk-hop)/2 秒会永远没有输出。
        commit_start = chunk_start if chunk_index == 0 else chunk_start + margin_duration
        # 尾窗提交区截到实际音频末尾；相邻窗口的提交区无缝拼接、互不重叠。
        commit_end = min(
            chunk_start + margin_duration + config.hop_duration,
            total_duration,
        )
        # 音频尾部剩余不足一个 margin 时，最后一个窗口的提交区为空，直接结束。
        if commit_start >= commit_end - 1e-9:
            break

        yield ChunkWindow(
            index=chunk_index,
            start_sample=chunk_start_sample,
            num_samples=chunk_samples,
            start=chunk_start,
            commit_start=commit_start,
            commit_end=commit_end,
        )

        chunk_index += 1
        chunk_start_sample += hop_samples


__all__ = ["ChunkWindow", "iter_chunk_schedule"]
