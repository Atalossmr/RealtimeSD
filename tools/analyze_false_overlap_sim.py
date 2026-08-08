#!/usr/bin/env python3
"""真/假重叠窗的匹配相似度分布分析（用于标定 separation_min_match_similarity）。

用法：
    python tools/analyze_false_overlap_sim.py \
        --ref datasets/aishell4-test/rttm/L_R003S01C02.rttm \
        --run-log exp/sep_export_full/run.log

判定：
- 从 run.log 的 [separation] pair_match 事件读取 2x2 匹配的候选、相似度与
  最终映射，取被选中两对中较小的相似度作为该窗的 min_sim；
- 参考 RTTM 重叠区间与 commit 区相交 >=0.1s 判为真重叠窗，否则为假重叠窗；
- 分别统计真/假重叠窗的 min_sim 分布。
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from analyze_overlap_coverage import parse_run_log


def load_rttm(path: str) -> list[tuple[float, float, str]]:
    turns = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            turns.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return turns


def overlap_intervals(turns, step=0.01):
    end_max = max(e for _, e, _ in turns)
    n = int(end_max / step) + 2
    count = [0] * n
    for s, e, _ in turns:
        for i in range(int(s / step), min(n, int(e / step + 0.9999))):
            count[i] += 1
    return [i * step for i in range(n) if count[i] >= 2]  # 重叠时刻集合


def chosen_min_sim(payload: dict) -> float | None:
    """从 pair_match 事件 payload 计算被选中分配的较小相似度。"""

    candidates = payload.get("candidates")
    sims = payload.get("sims")
    mapping = payload.get("mapping")
    if not candidates or not sims or not mapping:
        return None
    chosen = []
    for track_key in ("track0", "track1"):
        global_id = mapping.get(track_key)
        if global_id not in candidates:
            return None
        chosen.append(float(sims[track_key][candidates.index(global_id)]))
    return min(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--min-intersect", type=float, default=0.1)
    args = parser.parse_args()

    overlap_times = overlap_intervals(load_rttm(args.ref))

    def ref_overlap_duration(start: float, end: float) -> float:
        return sum(1 for t in overlap_times if start - 1e-9 <= t < end) * 0.01

    chunks = parse_run_log(args.run_log)
    rows = []
    for chunk in chunks:
        disp = chunk["disposition"]
        if disp not in {"separated", "sim_fallback", "gate_fallback", "both_failed"}:
            continue
        cs, ce = chunk["commit"]
        is_true = ref_overlap_duration(cs, ce) >= args.min_intersect
        pair_payload = chunk["events"].get("pair_match")
        min_sim = chosen_min_sim(pair_payload) if pair_payload else None
        rows.append((chunk["chunk"], cs, ce, disp, is_true, min_sim))

    buckets: dict[str, list[float]] = defaultdict(list)
    gate_false = 0
    print(f"{'chunk':>5} {'commit':>17} {'处置':>14} {'真重叠':>6} {'min_sim':>7}")
    for idx, cs, ce, disp, is_true, min_sim in rows:
        tag = "真" if is_true else "假"
        sim_str = f"{min_sim:.3f}" if min_sim is not None else "-"
        print(f"{idx:>5} {cs:>8.2f}-{ce:<8.2f} {disp:>14} {tag:>6} {sim_str:>7}")
        if disp in {"separated", "sim_fallback"} and min_sim is not None:
            buckets["true" if is_true else "false"].append(min_sim)
        elif disp == "gate_fallback" and not is_true:
            gate_false += 1

    def summarize(name: str, values: list[float]) -> None:
        if not values:
            print(f"{name}: 无样本")
            return
        values = sorted(values)
        n = len(values)
        qs = [values[int(q * (n - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        print(
            f"{name}: n={n} min={qs[0]:.3f} p25={qs[1]:.3f} "
            f"median={qs[2]:.3f} p75={qs[3]:.3f} max={qs[4]:.3f}"
        )

    print(f"\n假重叠且被能量门控拦截（gate_fallback）: {gate_false} 个")
    summarize("真重叠窗 min_sim", buckets["true"])
    summarize("假重叠窗 min_sim", buckets["false"])

    # 各候选阈值下的 拦截假重叠率 / 误伤真重叠率
    print(f"\n{'阈值':>6} {'拦假':>12} {'误伤真':>12}")
    for th in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        f = buckets["false"]
        t = buckets["true"]
        fr = sum(1 for v in f if v < th) / len(f) if f else float("nan")
        tr = sum(1 for v in t if v < th) / len(t) if t else float("nan")
        print(f"{th:>6.2f} {fr:>12.1%} {tr:>12.1%}")


if __name__ == "__main__":
    main()
