#!/usr/bin/env python3
"""分析当前管线（diarization 包，chunk 架构）run.log 中各机制命中情况与占比。

结构化事件 marker（见 diarization/utils/log.py 的 log_structured）：

- `[runtime] frame_decision:` 每个 chunk 的 local->global 分配（INFO，始终输出）
- `[runtime] current_global_speakers:` 当前全局 speaker 快照（INFO）
- `[separation] energy_gate:` 重叠窗的能量门控结果（INFO）
- `[separation] pair_match:` 2x2 匹配详情（相似度、映射、是否接受）（INFO）
- `[separation] gate_fallback:` 单路过门控的回退归属（INFO）
- `[separation] both_tracks_failed:` / `[separation] separate_too_short:`（WARNING）
- `[debug] window_summary:` chunk 级汇总（DEBUG，需 --debug）
- `[debug] new_speakers:` / `[debug] updated_speakers:` / `[debug] skipped_updates:`（DEBUG）

非结构化行（按正则识别）：
- `[streaming] finalized turns=N`、`[embeddings] saved N embeddings`、`[ahc] ...`

使用方法：
  1) 直接打印分析报告:
     python tools/analyze_run_log.py --log /path/to/run.log

  2) 同时导出 JSON 汇总:
     python tools/analyze_run_log.py --log /path/to/run.log --json-out /tmp/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from run_log_parser import basic_stats, iter_log_events


FINALIZED_TURNS_RE = re.compile(r"\[streaming\] finalized turns=(\d+)")
AHC_RE = re.compile(r"\[ahc\] observations=(\d+) clusters=(\d+)")
EMBEDDINGS_SAVED_RE = re.compile(r"\[embeddings\] saved (\d+) embeddings")


class CounterItem(TypedDict):
    name: str
    count: int
    ratio: float


class SeparationSummary(TypedDict, total=False):
    overlap_windows: int
    by_disposition: list[CounterItem]
    energy_ratio_stats: dict[str, float]
    pair_min_sim_stats: dict[str, float]
    gate_fallback_sim_stats: dict[str, float]


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _counter_with_ratio(counter: Counter[str], total: int) -> list[CounterItem]:
    """把计数器展开为含占比的列表。"""

    return [
        {"name": key, "count": int(value), "ratio": _ratio(int(value), int(total))}
        for key, value in counter.most_common()
    ]


def analyze_log(log_path: Path) -> dict:
    """扫描 run.log 并汇总机制命中统计。"""

    total_frame_decisions = 0
    commit_coverage = 0.0
    total_windows = 0
    windows_with_observation = 0

    sum_observations = 0
    sum_embedded = 0

    decision_counter: Counter[str] = Counter()
    decision_mode_counter: Counter[str] = Counter()
    similarity_by_decision: dict[str, list[float]] = {}
    skipped_reason_counter: Counter[str] = Counter()
    update_mode_counter: Counter[str] = Counter()
    update_alphas: list[float] = []

    new_speaker_count = 0
    final_speakers: list[dict] = []
    finalized_turns = 0
    embeddings_saved = 0
    ahc_files = 0
    ahc_observations = 0
    ahc_clusters = 0

    # ---- separation 事件统计 ----
    sep_disposition_counter: Counter[str] = Counter()
    sep_energy_ratios: list[float] = []
    sep_pair_min_sims: list[float] = []
    sep_gate_fallback_sims: list[float] = []

    def chosen_min_sim(payload: dict) -> float | None:
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

    with open(log_path, "r", encoding="utf-8") as file_obj:
        # 结构化事件走 iter_log_events；非结构化行用正则补充，
        # 因此先读全文再分别扫描（文件不大，run.log 一般 < 几十 MB）。
        text = file_obj.read()

    for prefix, title, payload in iter_log_events(log_path):
        if prefix == "runtime" and title == "frame_decision":
            total_frame_decisions += 1
            if isinstance(payload, dict) and "commit" in payload:
                commit = payload["commit"]
                commit_coverage += float(commit[1]) - float(commit[0])
            continue

        if prefix == "debug" and title == "window_summary":
            if not isinstance(payload, dict):
                continue
            total_windows += 1
            window_state = payload.get("window_state", {})
            if isinstance(window_state, dict):
                obs = int(window_state.get("observations", 0))
                sum_observations += obs
                sum_embedded += int(window_state.get("embedded", 0))
                if obs > 0:
                    windows_with_observation += 1
            assignment = payload.get("assignment", {})
            if isinstance(assignment, dict):
                local_assignments = assignment.get("local_assignments", [])
                if isinstance(local_assignments, list):
                    for item in local_assignments:
                        if not isinstance(item, dict):
                            continue
                        decision = str(item.get("decision", "unknown"))
                        mode = str(item.get("selection_mode", "unknown"))
                        decision_counter[decision] += 1
                        decision_mode_counter[f"{decision} | {mode}"] += 1
                        try:
                            similarity = float(item.get("similarity"))
                        except (TypeError, ValueError):
                            continue
                        if similarity >= 0.0:
                            similarity_by_decision.setdefault(decision, []).append(
                                similarity
                            )
            continue

        if prefix == "debug" and title == "skipped_updates":
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        skipped_reason_counter[str(item.get("reason", "unknown"))] += 1
            continue

        if prefix == "debug" and title == "updated_speakers":
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    update_mode_counter[str(item.get("mode", "unknown"))] += 1
                    try:
                        update_alphas.append(float(item.get("alpha")))
                    except (TypeError, ValueError):
                        pass
            continue

        if prefix == "debug" and title == "new_speakers":
            if isinstance(payload, list):
                new_speaker_count += len(payload)
            continue

        if prefix == "runtime" and title == "current_global_speakers":
            if isinstance(payload, list):
                # 每个 chunk 都会打一次快照，只保留最后一次。
                final_speakers = [item for item in payload if isinstance(item, dict)]
            continue

        if prefix == "separation" and isinstance(payload, dict):
            if title == "energy_gate":
                sep_disposition_counter["overlap_windows"] += 1
                for track in payload.get("tracks", []):
                    try:
                        sep_energy_ratios.append(float(track.get("ratio")))
                    except (TypeError, ValueError):
                        pass
            elif title == "pair_match":
                sep_disposition_counter[
                    "separated" if payload.get("accepted") else "sim_fallback"
                ] += 1
                min_sim = chosen_min_sim(payload)
                if min_sim is not None:
                    sep_pair_min_sims.append(min_sim)
            elif title == "gate_fallback":
                sep_disposition_counter["gate_fallback"] += 1
                try:
                    sep_gate_fallback_sims.append(float(payload.get("similarity")))
                except (TypeError, ValueError):
                    pass
            elif title == "both_tracks_failed":
                sep_disposition_counter["both_tracks_failed"] += 1
            elif title == "separate_too_short":
                sep_disposition_counter["separate_too_short"] += 1
            continue

    # ---- 非结构化行 ----
    for line in text.splitlines():
        turns_match = FINALIZED_TURNS_RE.search(line)
        if turns_match:
            finalized_turns += int(turns_match.group(1))
            continue
        ahc_match = AHC_RE.search(line)
        if ahc_match:
            ahc_files += 1
            ahc_observations += int(ahc_match.group(1))
            ahc_clusters += int(ahc_match.group(2))
            continue
        saved_match = EMBEDDINGS_SAVED_RE.search(line)
        if saved_match:
            embeddings_saved += int(saved_match.group(1))
            continue

    total_assignments = int(sum(decision_counter.values()))
    total_skips = int(sum(skipped_reason_counter.values()))
    total_updates = int(sum(update_mode_counter.values()))

    speakers: dict = {"new_speakers": int(new_speaker_count)}
    if final_speakers:
        counts = [float(item.get("count", 0)) for item in final_speakers]
        speakers["final_global_speakers"] = len(final_speakers)
        speakers["final_observation_counts"] = basic_stats(counts)

    streaming: dict = {}
    if finalized_turns:
        streaming["finalized_turns"] = int(finalized_turns)
    if embeddings_saved:
        streaming["embeddings_saved"] = int(embeddings_saved)

    ahc: dict = {}
    if ahc_files:
        ahc = {
            "files": ahc_files,
            "observations": int(ahc_observations),
            "clusters": int(ahc_clusters),
        }

    separation: dict = {}
    if sep_disposition_counter:
        overlap_windows = sep_disposition_counter.pop("overlap_windows")
        total_events = int(sum(sep_disposition_counter.values()))
        separation = {
            "overlap_windows": int(overlap_windows),
            "by_disposition": _counter_with_ratio(
                sep_disposition_counter, total_events
            ),
            "energy_ratio_stats": basic_stats(sep_energy_ratios),
            "pair_min_sim_stats": basic_stats(sep_pair_min_sims),
            "gate_fallback_sim_stats": basic_stats(sep_gate_fallback_sims),
        }

    return {
        "log_path": str(log_path),
        "frames": {
            "frame_decisions": total_frame_decisions,
            "commit_coverage_seconds": round(commit_coverage, 3),
            "window_summaries": total_windows,
            "windows_with_observation": windows_with_observation,
        },
        "observation": {
            "sum_observations": sum_observations,
            "sum_embedded": sum_embedded,
        },
        "assignment": {
            "total": total_assignments,
            "by_decision": _counter_with_ratio(decision_counter, total_assignments),
            "by_decision_and_mode": _counter_with_ratio(
                decision_mode_counter, total_assignments
            ),
            "similarity_stats_by_decision": {
                decision: basic_stats(values)
                for decision, values in sorted(similarity_by_decision.items())
            },
        },
        "updates": {
            "total": total_updates,
            "by_mode": _counter_with_ratio(update_mode_counter, total_updates),
            "alpha_stats": basic_stats(update_alphas),
        },
        "skips": {
            "total": total_skips,
            "by_reason": _counter_with_ratio(skipped_reason_counter, total_skips),
        },
        "speakers": speakers,
        "streaming": streaming,
        "ahc": ahc,
        "separation": separation,
    }


def _print_stats(title: str, stats: dict[str, float]) -> None:
    if not stats:
        return
    print(
        f"  {title}: min={stats['min']:.4f} mean={stats['mean']:.4f} "
        f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} max={stats['max']:.4f}"
    )


def _print_report(summary: dict) -> None:
    """以可读文本打印分析结果。"""

    frames = summary["frames"]
    observation = summary["observation"]
    assignment = summary["assignment"]
    updates = summary["updates"]
    skips = summary["skips"]
    speakers = summary.get("speakers", {})
    streaming = summary.get("streaming", {})
    ahc = summary.get("ahc", {})
    separation = summary.get("separation", {})

    print("== Run Log Analysis ==")
    print(f"log_path: {summary['log_path']}")
    print(
        f"frames: decisions={frames['frame_decisions']} "
        f"commit_coverage={frames['commit_coverage_seconds']}s "
        f"windows={frames['window_summaries']} "
        f"windows_with_observation={frames['windows_with_observation']}"
    )

    print(
        f"observations: total={observation['sum_observations']} "
        f"embedded={observation['sum_embedded']}"
    )

    print("\n-- assignment.by_decision --")
    print(f"total={assignment['total']}")
    for item in assignment["by_decision"]:
        print(f"{item['name']}: {item['count']} ({100.0 * float(item['ratio']):.2f}%)")
    for decision, stats in assignment["similarity_stats_by_decision"].items():
        _print_stats(f"similarity[{decision}]", stats)

    print("\n-- updates --")
    print(f"total={updates['total']}")
    for item in updates["by_mode"]:
        print(f"{item['name']}: {item['count']} ({100.0 * float(item['ratio']):.2f}%)")
    _print_stats("alpha", updates["alpha_stats"])

    print("\n-- skips.by_reason --")
    print(f"total={skips['total']}")
    for item in skips["by_reason"]:
        print(f"{item['name']}: {item['count']} ({100.0 * float(item['ratio']):.2f}%)")

    print("\n-- speakers --")
    print(f"new_speakers={speakers.get('new_speakers', 0)}")
    if "final_global_speakers" in speakers:
        print(f"final_global_speakers={speakers['final_global_speakers']}")
        _print_stats("observation_count", speakers["final_observation_counts"])

    print("\n-- streaming --")
    if "finalized_turns" in streaming:
        print(f"finalized_turns={streaming['finalized_turns']}")
    if "embeddings_saved" in streaming:
        print(f"embeddings_saved={streaming['embeddings_saved']}")
    if not streaming:
        print("(no streaming events)")

    if ahc:
        print("\n-- ahc --")
        print(
            f"files={ahc['files']} observations={ahc['observations']} "
            f"clusters={ahc['clusters']}"
        )

    if separation:
        print("\n-- separation --")
        print(f"overlap_windows={separation['overlap_windows']}")
        for item in separation["by_disposition"]:
            print(
                f"{item['name']}: {item['count']} "
                f"({100.0 * float(item['ratio']):.2f}%)"
            )
        _print_stats("energy_ratio", separation["energy_ratio_stats"])
        _print_stats("pair_min_sim", separation["pair_min_sim_stats"])
        _print_stats("gate_fallback_sim", separation["gate_fallback_sim_stats"])


def main() -> None:
    """解析参数并执行日志分析。"""

    parser = argparse.ArgumentParser(description="Analyze realtime pipeline run.log")
    parser.add_argument("--log", required=True, help="Path to run.log")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output JSON file path for machine-readable summary",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    summary = analyze_log(log_path)
    _print_report(summary)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)
        print(f"\nJSON summary written to: {output_path}")


if __name__ == "__main__":
    main()
