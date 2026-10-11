"""RTTM 解析与帧级活跃生成（离线分离工具的输入侧）。

只依赖标准库：tools/ 下的分析脚本直接 import 本模块取权威 `load_rttm`，
不触发 torch 等重依赖。
"""

from __future__ import annotations

import math
from pathlib import Path

# 帧级活跃记录：(frame_start, frame_end, active_speaker_ids)。
# 定义在本模块（build_frame_activities 的产物类型），attribution 从这里取用。
FrameActivity = tuple[float, float, list[int]]


def load_rttm(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """解析 RTTM，返回 {speaker label: [(start, end), ...]}（按 start 排序）。

    仅接受 SPEAKER 行（parts[3]=start, parts[4]=duration, parts[7]=speaker
    label）；非 SPEAKER 行与不足 8 列的短行跳过。

    Args:
        path: RTTM 文件路径。

    Returns:
        {speaker label: [(start, end), ...]}，各 speaker 的段按 start 排序。
    """

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


def build_frame_activities(
    turns_by_id: dict[int, list[tuple[float, float]]],
    frame_step: float,
    t_end: float,
) -> list[FrameActivity]:
    """以 frame_step 扫描 [0, t_end)，生成逐帧活跃 speaker id 列表。

    帧与 turn 区间有交集（turn_start < frame_end 且 turn_end > frame_start）
    即视为该 speaker 在该帧活跃；无活跃 speaker 的帧 active_ids 为空列表
    （保留在结果中，保证帧网格无缝覆盖全时间线）。t_end 之后（越界）的帧
    不生成。

    Args:
        turns_by_id: speaker id -> [(start, end), ...]（各列表按 start 排序）。
        frame_step: 帧步长（秒）。
        t_end: 扫描终点（秒，一般为音频时长）。

    Returns:
        按时间有序的 (frame_start, frame_end, active_ids) 列表；
        active_ids 按 speaker id 升序。
    """

    num_frames = 0
    while num_frames * frame_step < t_end - 1e-9:
        num_frames += 1
    # 每 speaker 的逐帧活跃标记：turn [s, e) 覆盖帧 i 当且仅当
    # i*step < e 且 (i+1)*step > s，即 i ∈ [floor(s/step), ceil(e/step)-1]。
    active_flags: dict[int, list[bool]] = {}
    for speaker_id, segments in turns_by_id.items():
        flags = [False] * num_frames
        for start, end in segments:
            first = max(0, int(math.floor(start / frame_step)))
            last = min(num_frames - 1, int(math.ceil(end / frame_step)) - 1)
            for frame_idx in range(first, last + 1):
                flags[frame_idx] = True
        active_flags[speaker_id] = flags

    frames: list[FrameActivity] = []
    for frame_idx in range(num_frames):
        frame_start = frame_idx * frame_step
        frame_end = min(frame_start + frame_step, t_end)
        active = [
            speaker_id
            for speaker_id in sorted(active_flags)
            if active_flags[speaker_id][frame_idx]
        ]
        frames.append((frame_start, frame_end, active))
    return frames


__all__ = ["FrameActivity", "load_rttm", "build_frame_activities"]
