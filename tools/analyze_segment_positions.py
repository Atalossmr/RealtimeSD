#!/usr/bin/env python3
"""导出音频段位置可靠性分析：对照参考 RTTM 逐段统计时长构成。

用法：
    python tools/analyze_segment_positions.py \
        --ref datasets/aishell4-test/rttm/L_R003S01C02.rttm \
        --sys exp/sep_export_test/aishell4_sep_test.streaming.rttm \
        --manifest exp/sep_export_test/aishell4_sep_test.segments.jsonl \
        --offset 300 --duration 120

说明：
- 测试音频截取自原文件 offset 秒起 duration 秒，参考 RTTM 裁剪到同区间并平移；
- 系统 speaker id 按与参考 speaker 的时间重叠总量做贪心映射；
- 逐段把 [start, end) 跨度按 10ms 栅格拆为：映射 speaker 语音 / 他人语音 /
  非语音 三类占比，并汇总全局统计。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load_rttm(path: str) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            turns.append((start, start + duration, parts[7]))
    return turns


def clip_and_shift(
    turns: list[tuple[float, float, str]], offset: float, duration: float
) -> list[tuple[float, float, str]]:
    end = offset + duration
    clipped = []
    for start, stop, speaker in turns:
        s, e = max(start, offset) - offset, min(stop, end) - offset
        if e > s:
            clipped.append((s, e, speaker))
    return clipped


def activity_masks(
    turns: list[tuple[float, float, str]], duration: float, step: float = 0.01
):
    """10ms 栅格上的 speaker -> bool 掩码。"""

    n = int(duration / step) + 1
    speakers = sorted({speaker for _, _, speaker in turns})
    masks = {speaker: [False] * n for speaker in speakers}
    for start, stop, speaker in turns:
        i0, i1 = int(start / step), min(n, int(stop / step + 0.9999))
        for i in range(i0, i1):
            masks[speaker][i] = True
    return masks, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sys", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    ref_turns = clip_and_shift(load_rttm(args.ref), args.offset, args.duration)
    sys_turns = load_rttm(args.sys)
    segments = [
        json.loads(line)
        for line in open(args.manifest, encoding="utf-8")
        if line.strip()
    ]

    ref_masks, n = activity_masks(ref_turns, args.duration, args.step)
    any_ref = [False] * n
    for mask in ref_masks.values():
        for i in range(n):
            any_ref[i] = any_ref[i] or mask[i]

    # ---- 系统 global id -> 参考 speaker：按时间重叠总量贪心映射 ----
    overlap_time: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for start, stop, sys_spk in sys_turns:
        i0, i1 = int(start / args.step), min(n, int(stop / args.step + 0.9999))
        for ref_spk, mask in ref_masks.items():
            overlap_time[sys_spk][ref_spk] += (
                sum(mask[i0:i1]) * args.step
            )
    # 多对一映射（argmax）：流式聚类会产生 false split，允许多个系统 id
    # 映射到同一参考 speaker。
    mapping: dict[str, str] = {
        sys_spk: max(row, key=row.get)
        for sys_spk, row in overlap_time.items()
        if row
    }

    print("speaker 映射（系统 -> 参考）:", {k: mapping[k] for k in sorted(mapping)})

    # ---- 逐段时长构成 ----
    print(
        f"\n{'sys':>4} {'ref':>6} {'span':>17} {'自身语音':>8} {'他人语音':>8} {'非语音':>8}"
    )
    totals = defaultdict(float)
    rows = []
    for seg in segments:
        sys_spk = str(seg["speaker_id"])
        ref_spk = mapping.get(sys_spk)
        i0 = int(seg["start"] / args.step)
        i1 = min(n, int(seg["end"] / args.step + 0.9999))
        own = other = silence = 0
        for i in range(i0, i1):
            if ref_spk is not None and ref_masks[ref_spk][i]:
                own += 1
            elif any_ref[i]:
                other += 1
            else:
                silence += 1
        total = max(1, i1 - i0)
        own_r, other_r, sil_r = own / total, other / total, silence / total
        totals["own"] += own * args.step
        totals["other"] += other * args.step
        totals["silence"] += silence * args.step
        totals["all"] += total * args.step
        flag = " <-- 他人语音>20%" if other_r > 0.2 else ""
        rows.append((seg, ref_spk, own_r, other_r, sil_r, flag))

    for seg, ref_spk, own_r, other_r, sil_r, flag in rows:
        print(
            f"{seg['speaker_id']:>4} {str(ref_spk):>6} "
            f"{seg['start']:>8.3f}-{seg['end']:<8.3f} "
            f"{own_r:>8.1%} {other_r:>8.1%} {sil_r:>8.1%}{flag}"
        )

    print(
        f"\n汇总（全部 {len(rows)} 段，共 {totals['all']:.1f}s）："
        f"自身语音 {totals['own'] / totals['all']:.1%}，"
        f"他人语音 {totals['other'] / totals['all']:.1%}，"
        f"非语音 {totals['silence'] / totals['all']:.1%}"
    )
    # 仅看经 TIGER 分离的重叠段（同一窗内有两个以上 speaker 段即该窗发生过分离）
    by_window = defaultdict(list)
    for seg, ref_spk, *_ in rows:
        by_window[(int(seg["start"] // 5), seg["start"])].append(seg)
    overlapped = [
        (seg, ref_spk, own_r, other_r, sil_r)
        for (seg, ref_spk, own_r, other_r, sil_r, _) in rows
        if any(
            abs(other_seg["start"] - seg["start"]) < 5
            and other_seg["speaker_id"] != seg["speaker_id"]
            and other_seg["start"] < seg["end"]
            and seg["start"] < other_seg["end"]
            for other_seg, *_ in rows
        )
    ]
    if overlapped:
        own_t = sum((s["end"] - s["start"]) * o for s, _, o, _, _ in overlapped)
        other_t = sum((s["end"] - s["start"]) * o for s, _, _, o, _ in overlapped)
        all_t = sum(s["end"] - s["start"] for s, *_ in overlapped)
        print(
            f"重叠窗段（{len(overlapped)} 段，{all_t:.1f}s）："
            f"自身语音 {own_t / all_t:.1%}，他人语音 {other_t / all_t:.1%}"
        )


if __name__ == "__main__":
    main()
