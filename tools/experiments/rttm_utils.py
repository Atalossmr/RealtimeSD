"""上限实验（tools/experiments/）的共享工具：RTTM 解析与区间运算。

RTTM 解析的权威实现是 `separation.rttm.load_rttm`（返回结构相同：
{speaker: [(start, end), ...]}，按 start 排序），此处 re-export 以兼容
既有 `from rttm_utils import load_rttm` 的写法；本模块自身只保留区间运算。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 本模块被同目录脚本以 `from rttm_utils import ...` 方式导入，自行补齐仓库根，
# 保证 separation 可 import（不依赖调用方的 sys.path 设置顺序）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from separation.rttm import load_rttm  # noqa: E402


def union_duration(segments: list[tuple[float, float]]) -> float:
    """区间并集总时长（输入无需有序）。

    Args:
        segments: [(start, end), ...] 区间列表。

    Returns:
        并集总时长（秒）。
    """

    total = 0.0
    current_start: float | None = None
    current_end = 0.0
    for start, end in sorted(segments):
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def overlap_histogram(
    turns: dict[str, list[tuple[float, float]]], step: float = 0.01
) -> dict[int, float]:
    """按同时说话人数统计时长：{1: 单人时长, 2: 双人重叠时长, 3: ...}。

    以 step 为帧长扫描（10ms 足够精确，比区间事件法实现简单）。

    Args:
        turns: load_rttm 的返回。
        step: 扫描帧长（秒）。

    Returns:
        {同时说话人数: 时长（秒）}；无语音返回空 dict。
    """

    all_segments = [seg for segments in turns.values() for seg in segments]
    if not all_segments:
        return {}
    t_end = max(end for _, end in all_segments)
    num_frames = int(t_end / step) + 1
    counts = [0] * num_frames
    for segments in turns.values():
        for start, end in segments:
            first = int(start / step)
            last = min(int(end / step), num_frames - 1)
            for frame in range(first, last + 1):
                counts[frame] += 1
    histogram: dict[int, float] = {}
    for count in counts:
        if count > 0:
            histogram[count] = histogram.get(count, 0.0) + step
    return histogram


__all__ = ["load_rttm", "union_duration", "overlap_histogram"]
