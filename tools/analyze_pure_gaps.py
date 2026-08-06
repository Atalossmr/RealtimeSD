#!/usr/bin/env python3
"""统计同一 local slot 相邻纯净区之间的间隔，评估小间隔桥接合并的价值。

间隔 = 同一 slot 两个纯净连通区之间的帧段。间隔帧分两类：
- overlap 帧：该 slot 活跃但同时有别人在说话（合并会把重叠语音吸进"纯净"区）
- inactive 帧：该 slot 不活跃（静默/其他人独说，合并无害）

用法：.venv/bin/python tools/analyze_pure_gaps.py <chunks_npz_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def connected_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """连续为真的帧区间（半开）。"""

    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def collect_gaps(npz_dir: Path):
    """遍历全部 chunk/slot，收集 (gap_frames, overlap_frames_in_gap, 左区长, 右区长)。"""

    gaps: list[tuple[int, int, int, int]] = []
    pure_durations: list[int] = []  # 全部纯净连通区长度（帧）
    for path in sorted(npz_dir.glob("*.chunks.npz")):
        data = np.load(path, allow_pickle=False)
        seg_offset = 0
        for num_frames, num_locals in zip(
            data["seg_num_frames"], data["seg_num_locals"]
        ):
            size = int(num_frames) * int(num_locals)
            seg = data["seg_values"][seg_offset : seg_offset + size].reshape(
                int(num_frames), int(num_locals)
            )
            seg_offset += size
            active = seg > 0.0
            overlap = np.sum(active, axis=1) >= 2
            for slot in range(active.shape[1]):
                pure = active[:, slot] & ~overlap
                runs = connected_runs(pure)
                pure_durations.extend(end - start for start, end in runs)
                for (ls, le), (rs, re_) in zip(runs, runs[1:]):
                    gap_overlap = int(np.sum(overlap[le:rs]))
                    gaps.append((rs - le, gap_overlap, le - ls, re_ - rs))
    return gaps, pure_durations


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    gaps, pure_durations = collect_gaps(Path(sys.argv[1]))
    frame_ms = 1000.0 * 10.0 / 589  # segmentation-3.0 帧步长 ≈ 16.9ms

    runs = np.array(pure_durations)
    print(f"pure regions: {len(runs)}")
    print(
        f"  length (frames): mean={runs.mean():.1f} "
        f"p50={np.percentile(runs, 50):.0f} p90={np.percentile(runs, 90):.0f}"
    )
    print(
        f"  >= 1.5s ({int(1500 / frame_ms)}f): {(runs * frame_ms >= 1500).mean():.1%}"
    )
    print()

    if not gaps:
        print("no gaps found")
        return
    g = np.array(gaps, dtype=float)
    gap_len, gap_ov, left_len, right_len = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    print(f"gaps between consecutive pure regions (same slot): {len(g)}")
    print(
        f"  gap length (frames): mean={gap_len.mean():.1f} "
        f"p50={np.percentile(gap_len, 50):.0f} p90={np.percentile(gap_len, 90):.0f}"
    )
    only_overlap = (gap_ov == gap_len).mean()
    only_inactive = (gap_ov == 0).mean()
    print(
        f"  composition: all-overlap={only_overlap:.1%} "
        f"all-inactive={only_inactive:.1%} mixed={1 - only_overlap - only_inactive:.1%}"
    )
    print()

    # 间隔长度直方图 × 构成。
    edges = [1, 2, 3, 4, 6, 11, 21, 51, 10**9]
    labels = ["1f", "2f", "3f", "4-5f", "6-10f", "11-20f", "21-50f", ">50f"]
    print("gap length histogram (frame ~= 16.9ms):")
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        sel = (gap_len >= lo) & (gap_len < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        ov = (gap_ov[sel] == gap_len[sel]).mean()
        ina = (gap_ov[sel] == 0).mean()
        print(
            f"  {label:>7}: {n:7d} ({n / len(gaps):5.1%})  "
            f"all-overlap={ov:5.1%} all-inactive={ina:5.1%}"
        )
    print()

    # 桥接阈值 K：合并 gap <= K 帧后，纯净区时长与 >=1.5s 比例的变化；
    # 以及被吸进"纯净"区的 overlap 帧总量（污染）。
    print("bridge gaps <= K frames:")
    for k in (1, 2, 3, 5, 10, 20):
        # 逐 slot 重建太长，这里用上下界近似：
        # 可桥接的 gap 数、因此消失的 region 边界数、吸进去的 overlap 帧数。
        bridgeable = gap_len <= k
        absorbed_ov = int(gap_ov[bridgeable].sum())
        # 合并后 region 数减少、长度增加；用 Monte-Carlo 太重，
        # 直接统计被桥接 gap 两侧的短 region 有多少能因此达到 1.5s。
        win_frames = 1500.0 / frame_ms
        short_side = np.minimum(left_len, right_len)
        rescued = (
            bridgeable & (short_side < win_frames) & (left_len + right_len >= win_frames)
        ).sum()
        print(
            f"  K={k:2d}: bridged={int(bridgeable.sum()):7d} "
            f"({bridgeable.mean():5.1%} of gaps)  "
            f"overlap frames absorbed={absorbed_ov:6d}  "
            f"short regions rescued to >=1.5s: {int(rescued)}"
        )


if __name__ == "__main__":
    main()
