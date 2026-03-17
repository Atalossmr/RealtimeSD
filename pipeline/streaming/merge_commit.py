"""streaming RTTM overlap 去重相关工具。"""

from __future__ import annotations


def subtract_overlaps(
    *,
    intervals: list[tuple[float, float]],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """从 [start, end) 中裁掉已写 RTTM 的重叠部分。"""

    start = float(start)
    end = float(end)
    if end <= start:
        return []
    if not intervals:
        return [(start, end)]

    remaining: list[tuple[float, float]] = [(start, end)]
    for written_start, written_end in intervals:
        if not remaining:
            break
        new_remaining: list[tuple[float, float]] = []
        for cur_start, cur_end in remaining:
            if written_end <= cur_start + 1e-9 or written_start >= cur_end - 1e-9:
                new_remaining.append((cur_start, cur_end))
                continue
            if cur_start + 1e-9 < written_start:
                new_remaining.append((cur_start, min(cur_end, written_start)))
            if cur_end - 1e-9 > written_end:
                new_remaining.append((max(cur_start, written_end), cur_end))
        remaining = new_remaining

    return [(s, e) for s, e in remaining if e - s > 1e-6]


def register_written_interval(
    *,
    intervals: list[tuple[float, float]],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """登记一个已写出的 RTTM 区间，并保持有序且不重叠。"""

    start = float(start)
    end = float(end)
    if end <= start:
        return intervals

    items = list(intervals)
    items.append((start, end))
    items.sort(key=lambda item: item[0])

    merged: list[tuple[float, float]] = []
    for seg_start, seg_end in items:
        if not merged:
            merged.append((seg_start, seg_end))
            continue
        last_start, last_end = merged[-1]
        if seg_start <= last_end + 1e-6:
            merged[-1] = (last_start, max(last_end, seg_end))
        else:
            merged.append((seg_start, seg_end))
    return merged
