#!/usr/bin/env python3
"""分析当前管线（diarization 包，chunk 架构）run.log 中各机制命中情况与占比。

当前管线的 marker（见 diarization/pipeline.py 与 diarization/cluster/）：

- `[runtime] frame_decision:` 每个 chunk 的 local->global 分配（INFO，始终输出）
- `[debug] window_summary:` chunk 级汇总（DEBUG，需 --debug）
- `[debug] new_speakers:` / `[debug] updated_speakers:` / `[debug] skipped_updates:`（DEBUG）
- `[runtime] current_global_speakers:` 当前全局 speaker 快照（INFO）
- `[streaming] finalized turns=N` writer 收尾（INFO）
- `[ahc] observations=N clusters=M ...` AHC 离线后端聚类结果（INFO）

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
from typing import Iterator, TypedDict, TextIO


FINALIZED_TURNS_RE = re.compile(r"\[streaming\] finalized turns=(\d+)")
AHC_RE = re.compile(r"\[ahc\] observations=(\d+) clusters=(\d+)")


class CounterItem(TypedDict):
    name: str
    count: int
    ratio: float


class FramesSummary(TypedDict):
    frame_decisions: int
    window_summaries: int
    windows_with_observation: int


class ObservationSummary(TypedDict):
    sum_observations: int
    sum_embedded: int


class AssignmentSummary(TypedDict):
    total: int
    by_decision: list[CounterItem]
    by_decision_and_mode: list[CounterItem]
    similarity_stats_by_decision: dict[str, dict[str, float]]


class UpdatesSummary(TypedDict):
    total: int
    by_mode: list[CounterItem]
    alpha_stats: dict[str, float]


class SkipsSummary(TypedDict):
    total: int
    by_reason: list[CounterItem]


class SpeakersSummary(TypedDict, total=False):
    new_speakers: int
    final_global_speakers: int
    final_observation_counts: dict[str, float]


class StreamingSummary(TypedDict, total=False):
    finalized_turns: int
    embeddings_saved: int


class AhcSummary(TypedDict, total=False):
    files: int
    observations: int
    clusters: int


class RunLogSummary(TypedDict):
    log_path: str
    frames: FramesSummary
    observation: ObservationSummary
    assignment: AssignmentSummary
    updates: UpdatesSummary
    skips: SkipsSummary
    speakers: SpeakersSummary
    streaming: StreamingSummary
    ahc: AhcSummary


def _iter_lines(file_obj: TextIO) -> Iterator[str]:
    """按行迭代文本文件内容。"""

    for line in file_obj:
        yield line


def _read_json_block(lines: Iterator[str]) -> object | None:
    """从日志流中读取 marker 后的 JSON 块。"""

    first = None
    for line in lines:
        if line.strip():
            first = line
            break
    if first is None:
        return None

    first_stripped = first.lstrip()
    if not first_stripped or first_stripped[0] not in "[{":
        return None

    buffer = [first]
    stack: list[str] = []
    in_string = False
    escape = False

    def feed(text: str) -> None:
        nonlocal in_string, escape
        for ch in text:
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    continue
                left = stack[-1]
                if (left == "{" and ch == "}") or (left == "[" and ch == "]"):
                    stack.pop()

    feed(first)
    while stack:
        try:
            nxt = next(lines)
        except StopIteration:
            break
        buffer.append(nxt)
        feed(nxt)

    try:
        return json.loads("".join(buffer))
    except json.JSONDecodeError:
        return None


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _counter_with_ratio(counter: Counter[str], total: int) -> list[CounterItem]:
    """把计数器展开为含占比的列表。"""

    items: list[CounterItem] = []
    for key, value in counter.most_common():
        items.append(
            {
                "name": key,
                "count": int(value),
                "ratio": _ratio(int(value), int(total)),
            }
        )
    return items


def _basic_stats(values: list[float]) -> dict[str, float]:
    """计算一组数值的基础统计量（min/mean/p50/p90/max）。"""

    if not values:
        return {}
    sorted_values = sorted(values)
    n = len(sorted_values)

    def q(p: float) -> float:
        idx = min(n - 1, int((n - 1) * p))
        return float(sorted_values[idx])

    return {
        "min": float(sorted_values[0]),
        "mean": float(sum(sorted_values) / n),
        "p50": q(0.50),
        "p90": q(0.90),
        "max": float(sorted_values[-1]),
    }


def analyze_log(log_path: Path) -> RunLogSummary:
    """扫描 run.log 并汇总机制命中统计。"""

    total_frame_decisions = 0
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

    with open(log_path, "r", encoding="utf-8") as file_obj:
        lines = _iter_lines(file_obj)
        for line in lines:
            if "[runtime] frame_decision:" in line:
                total_frame_decisions += 1
                _ = _read_json_block(lines)
                continue

            if "[debug] window_summary:" in line:
                payload = _read_json_block(lines)
                if not isinstance(payload, dict):
                    continue
                total_windows += 1

                window_state = payload.get("window_state", {})
                if isinstance(window_state, dict):
                    obs = int(window_state.get("observations", 0))
                    embedded = int(window_state.get("embedded", 0))
                    sum_observations += obs
                    sum_embedded += embedded
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
                                similarity_by_decision.setdefault(
                                    decision, []
                                ).append(similarity)
                continue

            if "[debug] skipped_updates:" in line:
                payload = _read_json_block(lines)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            reason = str(item.get("reason", "unknown"))
                            skipped_reason_counter[reason] += 1
                continue

            if "[debug] updated_speakers:" in line:
                payload = _read_json_block(lines)
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        mode = str(item.get("mode", "unknown"))
                        update_mode_counter[mode] += 1
                        try:
                            update_alphas.append(float(item.get("alpha")))
                        except (TypeError, ValueError):
                            pass
                continue

            if "[debug] new_speakers:" in line:
                payload = _read_json_block(lines)
                if isinstance(payload, list):
                    new_speaker_count += len(payload)
                continue

            if "[runtime] current_global_speakers:" in line:
                payload = _read_json_block(lines)
                if isinstance(payload, list):
                    # 每个 chunk 都会打一次快照，只保留最后一次。
                    final_speakers = [
                        item for item in payload if isinstance(item, dict)
                    ]
                continue

            turns_match = FINALIZED_TURNS_RE.search(line)
            if turns_match:
                # 每个输入文件 finalize 一次，跨文件累加。
                finalized_turns += int(turns_match.group(1))
                continue

            ahc_match = AHC_RE.search(line)
            if ahc_match:
                ahc_files += 1
                ahc_observations += int(ahc_match.group(1))
                ahc_clusters += int(ahc_match.group(2))
                continue

            if "[embeddings] saved" in line:
                saved_match = re.search(r"saved (\d+) embeddings", line)
                if saved_match:
                    embeddings_saved += int(saved_match.group(1))
                continue

    total_assignments = int(sum(decision_counter.values()))
    total_skips = int(sum(skipped_reason_counter.values()))
    total_updates = int(sum(update_mode_counter.values()))

    speakers: SpeakersSummary = {"new_speakers": int(new_speaker_count)}
    if final_speakers:
        counts = [float(item.get("count", 0)) for item in final_speakers]
        speakers["final_global_speakers"] = len(final_speakers)
        speakers["final_observation_counts"] = _basic_stats(counts)

    streaming: StreamingSummary = {}
    if finalized_turns:
        streaming["finalized_turns"] = int(finalized_turns)
    if embeddings_saved:
        streaming["embeddings_saved"] = int(embeddings_saved)

    ahc: AhcSummary = {}
    if ahc_files:
        ahc = {
            "files": ahc_files,
            "observations": int(ahc_observations),
            "clusters": int(ahc_clusters),
        }

    return {
        "log_path": str(log_path),
        "frames": {
            "frame_decisions": total_frame_decisions,
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
                decision: _basic_stats(values)
                for decision, values in sorted(similarity_by_decision.items())
            },
        },
        "updates": {
            "total": total_updates,
            "by_mode": _counter_with_ratio(update_mode_counter, total_updates),
            "alpha_stats": _basic_stats(update_alphas),
        },
        "skips": {
            "total": total_skips,
            "by_reason": _counter_with_ratio(skipped_reason_counter, total_skips),
        },
        "speakers": speakers,
        "streaming": streaming,
        "ahc": ahc,
    }


def _print_stats(title: str, stats: dict[str, float]) -> None:
    if not stats:
        return
    print(
        f"  {title}: min={stats['min']:.4f} mean={stats['mean']:.4f} "
        f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} max={stats['max']:.4f}"
    )


def _print_report(summary: RunLogSummary) -> None:
    """以可读文本打印分析结果。"""

    frames = summary["frames"]
    observation = summary["observation"]
    assignment = summary["assignment"]
    updates = summary["updates"]
    skips = summary["skips"]
    speakers = summary.get("speakers", {})
    streaming = summary.get("streaming", {})
    ahc = summary.get("ahc", {})

    print("== Run Log Analysis ==")
    print(f"log_path: {summary['log_path']}")
    print(
        f"frames: decisions={frames['frame_decisions']} windows={frames['window_summaries']} "
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
