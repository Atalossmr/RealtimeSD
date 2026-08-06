#!/usr/bin/env python3
"""统计 chunks.npz 中用于提 embedding 的拼接时长（segment_duration_for_embedding）分布。

用法：.venv/bin/python tools/analyze_segment_durations.py <chunks_npz_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_durations(npz_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """收集全部 observation 的 duration 与 selection_mode。"""

    durations: list[np.ndarray] = []
    modes: list[np.ndarray] = []
    for path in sorted(npz_dir.glob("*.chunks.npz")):
        data = np.load(path, allow_pickle=False)
        has_embedding = data["has_embedding"]
        durations.append(data["duration"][has_embedding])
        modes.append(data["selection_mode"][has_embedding].astype(str))
    if not durations:
        return np.zeros(0), np.zeros(0, dtype=str)
    return np.concatenate(durations), np.concatenate(modes)


def report(durations: np.ndarray, modes: np.ndarray) -> None:
    total = len(durations)
    print(f"observations (with embedding): {total}")
    print()

    for mode in ("non_overlap", "overlap_fallback"):
        sel = durations[modes == mode]
        if sel.size == 0:
            continue
        q = np.percentile(sel, [10, 25, 50, 75, 90, 99])
        print(f"[{mode}] n={sel.size} ({sel.size / total:.1%})")
        print(
            f"  mean={sel.mean():.3f}s std={sel.std():.3f}s "
            f"min={sel.min():.3f}s max={sel.max():.3f}s"
        )
        print(
            f"  p10={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} "
            f"p75={q[3]:.3f} p90={q[4]:.3f} p99={q[5]:.3f}"
        )
        for thr in (0.75, 1.0, 1.5, 2.0, 3.0):
            print(f"  >= {thr:.2f}s: {(sel >= thr).mean():.1%}")
        # 触顶 4s 上限（max_segment_duration_for_embedding）的比例。
        print(f"  hit 4.0s cap (>=3.99s): {(sel >= 3.99).mean():.1%}")
        print()

    # 直方图（全部 observation 合并）。
    bins = np.array([0.0, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.01])
    hist, _ = np.histogram(durations, bins=bins)
    print("[histogram] all modes")
    for lo, hi, count in zip(bins[:-1], bins[1:], hist):
        bar = "#" * int(round(60 * count / max(1, hist.max())))
        print(f"  {lo:4.2f}-{hi:4.2f}s: {count:7d} ({count / total:6.1%}) {bar}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    durations, modes = load_durations(Path(sys.argv[1]))
    if durations.size == 0:
        raise SystemExit("no observations found")
    report(durations, modes)


if __name__ == "__main__":
    main()
