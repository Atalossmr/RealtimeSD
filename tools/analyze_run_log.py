#!/usr/bin/env python3
"""分析实时推理 run.log 中各机制命中情况与占比。

使用方法：
  1) 直接打印分析报告:
     python tools/analyze_run_log.py --log /path/to/run.log

  2) 同时导出 JSON 汇总:
     python tools/analyze_run_log.py --log /path/to/run.log --json-out /tmp/run_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, TypedDict, TextIO


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


class UpdatesSummary(TypedDict):
    total: int
    weak_updates: int
    weak_update_ratio: float
    by_mode: list[CounterItem]


class SkipsSummary(TypedDict):
    total: int
    by_reason: list[CounterItem]


class StreamingEventsSummary(TypedDict, total=False):
    merge_events: int
    speaker_became_stable: int
    unstable_finalize_speakers: int


class RunLogSummary(TypedDict):
    log_path: str
    frames: FramesSummary
    observation: ObservationSummary
    assignment: AssignmentSummary
    updates: UpdatesSummary
    skips: SkipsSummary
    streaming_events: StreamingEventsSummary


def _iter_lines(file_obj: TextIO) -> Iterator[str]:
    """功能：按行迭代文本文件内容。"""

    for line in file_obj:
        yield line


def _read_json_block(lines: Iterator[str]) -> object | None:
    """功能：从日志流中读取 marker 后的 JSON 块。"""

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
    """功能：安全计算比例。"""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _counter_with_ratio(counter: Counter[str], total: int) -> list[CounterItem]:
    """功能：把计数器展开为含占比的列表。"""

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


def analyze_log(log_path: Path) -> RunLogSummary:
    """功能：扫描 run.log 并汇总机制命中统计。"""

    total_frame_decisions = 0
    total_windows = 0
    windows_with_observation = 0

    sum_observations = 0
    sum_embedded = 0

    decision_counter: Counter[str] = Counter()
    decision_mode_counter: Counter[str] = Counter()
    skipped_reason_counter: Counter[str] = Counter()
    update_mode_counter: Counter[str] = Counter()

    merge_event_count = 0
    speaker_became_stable_count = 0
    unstable_finalize_speaker_count = 0

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
                        if isinstance(item, dict):
                            mode = str(item.get("mode", "unknown"))
                            update_mode_counter[mode] += 1
                continue

            if "[streaming] merge_event" in line:
                merge_event_count += 1
                continue

            if "became stable; flushing" in line:
                speaker_became_stable_count += 1
                continue

            if "unstable speaker" in line and "cached RTTM turns at finalize" in line:
                unstable_finalize_speaker_count += 1
                continue

    total_assignments = int(sum(decision_counter.values()))
    total_skips = int(sum(skipped_reason_counter.values()))
    total_updates = int(sum(update_mode_counter.values()))

    weak_updates = sum(
        count for mode, count in update_mode_counter.items() if mode.endswith("_weak")
    )

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
        },
        "updates": {
            "total": total_updates,
            "weak_updates": int(weak_updates),
            "weak_update_ratio": _ratio(int(weak_updates), total_updates),
            "by_mode": _counter_with_ratio(update_mode_counter, total_updates),
        },
        "skips": {
            "total": total_skips,
            "by_reason": _counter_with_ratio(skipped_reason_counter, total_skips),
        },
        "streaming_events": {
            "merge_events": int(merge_event_count),
            "speaker_became_stable": int(speaker_became_stable_count),
            "unstable_finalize_speakers": int(unstable_finalize_speaker_count),
        },
    }


def _print_report(summary: RunLogSummary) -> None:
    """功能：以可读文本打印分析结果。"""

    frames = summary["frames"]
    observation = summary["observation"]
    assignment = summary["assignment"]
    updates = summary["updates"]
    skips = summary["skips"]
    streaming_events = summary.get("streaming_events", {})

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
    for item in assignment["by_decision"]:
        print(f"{item['name']}: {item['count']} ({100.0 * float(item['ratio']):.2f}%)")

    print("\n-- updates.by_mode --")
    print(
        f"total={updates['total']} weak_updates={updates['weak_updates']} "
        f"weak_ratio={100.0 * float(updates['weak_update_ratio']):.2f}%"
    )
    for item in updates["by_mode"]:
        print(f"{item['name']}: {item['count']} ({100.0 * float(item['ratio']):.2f}%)")

    print("\n-- skips.by_reason --")
    print(f"total={skips['total']}")
    for item in skips["by_reason"]:
        print(f"{item['name']}: {item['count']} ({100.0 * float(item['ratio']):.2f}%)")

    print("\n-- streaming_events --")
    print(f"merge_events={streaming_events.get('merge_events', 0)}")
    print(f"speaker_became_stable={streaming_events.get('speaker_became_stable', 0)}")
    print(
        f"unstable_finalize_speakers={streaming_events.get('unstable_finalize_speakers', 0)}"
    )


def main() -> None:
    """功能：解析参数并执行日志分析。"""

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
