#!/usr/bin/env python3
"""重叠区分离覆盖率分析：参考 RTTM 的重叠区间 vs 导出段的分离处置结果。

用法：
    python tools/analyze_overlap_coverage.py \
        --ref datasets/aishell4-test/rttm/L_R003S01C02.rttm \
        --run-log exp/sep_export_test2/run.log \
        --manifest exp/sep_export_test2/aishell4_sep_test.segments.jsonl \
        --offset 300 --duration 120

判定：
- 从 run.log 按 chunk 顺序提取分离处置（separated / gate_fallback / both_failed）；
- 参考重叠区间（>=2 speaker 同时活跃，合并相邻）落到哪个 commit 区，
  即继承该 chunk 的处置结果；
- 同时检查该区间是否两个 speaker 都有导出段覆盖（>=50%）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load_rttm(path: str) -> list[tuple[float, float, str]]:
    turns = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            turns.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return turns


def ref_overlap_intervals(
    turns: list[tuple[float, float, str]], step: float = 0.01
) -> list[tuple[float, float, list[str]]]:
    """栅格化找 >=2 speaker 活跃的区间并合并相邻，返回 (start, end, speakers)。"""
    end_max = max(e for _, e, _ in turns)
    n = int(end_max / step) + 2
    active: list[set[str]] = [set() for _ in range(n)]
    for s, e, spk in turns:
        for i in range(int(s / step), min(n, int(e / step + 0.9999))):
            active[i].add(spk)
    intervals = []
    for i, spks in enumerate(active):
        if len(spks) >= 2:
            t = i * step
            if intervals and t - intervals[-1][1] <= 0.25:
                intervals[-1] = (intervals[-1][0], t + step, intervals[-1][2] | spks)
            else:
                intervals.append((t, t + step, set(spks)))
    return [(s, e, sorted(spks)) for s, e, spks in intervals]


def parse_run_log(path: str) -> list[dict]:
    """按 chunk 解析 commit 区与分离处置（基于结构化 JSON 事件）。

    每个 chunk 返回 {chunk, commit, disposition, events}；
    events 为 {事件名: payload}（separation 事件自带 chunk_index，
    不再需要按日志位置归属）。
    """

    from run_log_parser import iter_log_events

    commits: dict[int, tuple[float, float]] = {}
    events: dict[int, dict[str, dict]] = defaultdict(dict)
    for prefix, title, payload in iter_log_events(path):
        if prefix == "runtime" and title == "frame_decision":
            idx = int(payload["chunk_index"])
            commits[idx] = (float(payload["commit"][0]), float(payload["commit"][1]))
        elif prefix == "separation" and isinstance(payload, dict):
            idx = payload.get("chunk_index")
            if idx is not None:
                events[int(idx)][title] = payload

    chunks = []
    for idx in sorted(commits):
        chunk_events = events.get(idx, {})
        disposition = "no_overlap"
        if "pair_match" in chunk_events:
            disposition = (
                "separated"
                if chunk_events["pair_match"].get("accepted")
                else "sim_fallback"
            )
        elif "gate_fallback" in chunk_events:
            disposition = "gate_fallback"
        elif "both_tracks_failed" in chunk_events:
            disposition = "both_failed"
        chunks.append(
            {
                "chunk": idx,
                "commit": commits[idx],
                "disposition": disposition,
                "events": chunk_events,
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()

    ref = []
    for s, e, spk in load_rttm(args.ref):
        a, b = max(s, args.offset) - args.offset, min(e, args.offset + args.duration) - args.offset
        if b > a:
            ref.append((a, b, spk))
    overlaps = ref_overlap_intervals(ref)
    chunks = parse_run_log(args.run_log)
    segments = [
        json.loads(line) for line in open(args.manifest) if line.strip()
    ]

    def chunk_at(t: float):
        for c in chunks:
            if c["commit"][0] - 1e-9 <= t < c["commit"][1] - 1e-9:
                return c
        return None

    def coverage(start: float, end: float) -> dict[int, float]:
        cov: dict[int, float] = defaultdict(float)
        for seg in segments:
            ov = min(end, seg["end"]) - max(start, seg["start"])
            if ov > 0:
                cov[seg["speaker_id"]] += ov
        span = end - start
        return {spk: v / span for spk, v in cov.items()}

    print(f"参考重叠区间共 {len(overlaps)} 个：\n")
    print(f"{'区间':>17} {'时长':>5} {'参考spk':>10} {'chunk':>5} {'处置':>14} {'导出段覆盖':>22}")
    stats = defaultdict(float)
    total_dur = 0.0
    for start, end, spks in overlaps:
        c = chunk_at((start + end) / 2)
        disp = c["disposition"] if c else "out_of_range"
        idx = c["chunk"] if c else -1
        cov = coverage(start, end)
        cov_str = ", ".join(f"spk{k}:{v:.0%}" for k, v in sorted(cov.items()))
        dur = end - start
        total_dur += dur
        stats[disp] += dur
        print(f"{start:>8.2f}-{end:<8.2f} {dur:>5.2f} {str(spks):>10} {idx:>5} {disp:>14} {cov_str:>22}")

    print(f"\n重叠总时长 {total_dur:.2f}s，按处置分类：")
    for disp, dur in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {disp:>14}: {dur:6.2f}s ({dur / total_dur:.1%})")


if __name__ == "__main__":
    main()
