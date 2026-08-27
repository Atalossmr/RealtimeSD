#!/usr/bin/env python3
"""E1 数据集固有下限统计：不改模型，解析参考 RTTM 量化架构结构性下限。

统计项（逐文件 + 全局）：
- 语音总时长（并集）、重叠语音（>=2 人）时长与占比 —— DER 计入重叠时的固有难度；
- >=3 人同时说话时长与占比 —— pyannote segmentation-3.0 局部最多 3 人/窗、
  帧级最多 2 人重叠，这部分是分割模型的结构性不可达下限；
- 说话人数、最短板说话人总时长（centroid 收敛难度）；
- 短于 min_segment_duration 的 turn 数量与时长（闭合时会被丢弃，直接贡献 MS）。

用法：
  .venv/bin/python tools/experiments/dataset_stats.py \
      --ref datasets/task1-6/rttm --min-segment-duration 0.30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rttm_utils import load_rttm, overlap_histogram, union_duration  # noqa: E402


def stats_for_file(
    rttm_path: Path, min_segment_duration: float
) -> dict[str, float | int]:
    turns = load_rttm(rttm_path)
    all_segments = [seg for segments in turns.values() for seg in segments]
    speech_total = union_duration(all_segments)
    histogram = overlap_histogram(turns)
    overlap = sum(d for k, d in histogram.items() if k >= 2)
    overlap3 = sum(d for k, d in histogram.items() if k >= 3)

    short_turns = [s for s, e in all_segments if e - s < min_segment_duration]
    short_duration = sum(e - s for s, e in short_turns)

    per_speaker = {
        spk: union_duration(segments) for spk, segments in turns.items()
    }
    return {
        "num_speakers": len(turns),
        "num_turns": len(all_segments),
        "speech_total": speech_total,
        "overlap": overlap,
        "overlap_pct": 100.0 * overlap / speech_total if speech_total else 0.0,
        "overlap3": overlap3,
        "overlap3_pct": 100.0 * overlap3 / speech_total if speech_total else 0.0,
        "short_turns": len(short_turns),
        "short_duration": short_duration,
        "short_pct": 100.0 * short_duration / speech_total if speech_total else 0.0,
        "min_speaker_duration": min(per_speaker.values()) if per_speaker else 0.0,
        "max_speaker_duration": max(per_speaker.values()) if per_speaker else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--min-segment-duration", type=float, default=0.30)
    parser.add_argument("--output", default=None, help="可选：结果写入该文本文件")
    args = parser.parse_args()

    rttm_files = sorted(Path(args.ref).glob("*.rttm"))
    if not rttm_files:
        raise SystemExit(f"no .rttm under {args.ref}")

    lines: list[str] = []
    header = (
        f"{'file':<6} {'spk':>4} {'turns':>6} {'speech':>8} "
        f"{'ovl%':>6} {'>=3spk%':>8} {'short%':>7} {'min_spk':>8} {'max_spk':>8}"
    )
    lines.append(header)
    totals: dict[str, float] = {
        key: 0.0
        for key in ("speech_total", "overlap", "overlap3", "short_duration")
    }
    total_short_turns = 0
    for rttm_path in rttm_files:
        stats = stats_for_file(rttm_path, args.min_segment_duration)
        lines.append(
            f"{rttm_path.stem:<6} {stats['num_speakers']:>4} {stats['num_turns']:>6} "
            f"{stats['speech_total']:>8.1f} {stats['overlap_pct']:>6.2f} "
            f"{stats['overlap3_pct']:>8.3f} {stats['short_pct']:>7.3f} "
            f"{stats['min_speaker_duration']:>8.1f} {stats['max_speaker_duration']:>8.1f}"
        )
        for key in totals:
            totals[key] += float(stats[key])
        total_short_turns += int(stats["short_turns"])

    speech = totals["speech_total"]
    lines.append("-" * len(header))
    lines.append(
        f"TOTAL  spk=-  turns=-  speech={speech:.1f}s "
        f"overlap={100.0 * totals['overlap'] / speech:.2f}% "
        f">=3spk={100.0 * totals['overlap3'] / speech:.3f}% "
        f"short={100.0 * totals['short_duration'] / speech:.3f}% "
        f"({total_short_turns} turns)"
    )
    lines.append("")
    lines.append("解读：")
    lines.append("- overlap% 计入 DER 评分（collar=0、不跳重叠），是重叠处理的固有难度；")
    lines.append("- >=3spk% 超出 segmentation-3.0 帧级能力（<=2 人重叠），是分割结构性下限；")
    lines.append("- short% 的 turn 闭合时被 min_segment_duration 丢弃，直接贡献 MS。")

    report = "\n".join(lines)
    print(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
