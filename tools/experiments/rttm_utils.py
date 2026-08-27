"""上限实验（tools/experiments/）的共享工具：RTTM 解析与区间运算。"""

from __future__ import annotations

from pathlib import Path


def load_rttm(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """解析 RTTM，返回 {speaker: [(start, end), ...]}，按 start 排序。"""

    turns: dict[str, list[tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            turns.setdefault(speaker, []).append((start, start + duration))
    for segments in turns.values():
        segments.sort()
    return turns


def union_duration(segments: list[tuple[float, float]]) -> float:
    """区间并集总时长（输入无需有序）。"""

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
