#!/usr/bin/env python3
"""运行日志事件统计：按模块 / 事件 / 级别聚合计数。

解析 `setup_logger` 输出格式（`YYYY-MM-DD HH:MM:SS,mmm [LEVEL] message`）的日志
文件（run.log / transcribe.log / extract / cluster 日志均可），统计：

- 时间范围与日志行数（多行 JSON 负载等续行单独计数）；
- 每个模块（消息开头的 `[tag]`，无 tag 归入 general）下每种事件的
  INFO/WARNING/ERROR 次数——事件名取冒号前前缀（或前几个词），数字归一为 #
  （如 `[merge] speaker 0 -> 3` 归并为 `speaker # -> #`）；
- WARNING/ERROR 明细（末尾列出，便于直接看异常）。

用法：

    python3 tools/analyze_log.py exp/common/default/run.log
    python3 tools/analyze_log.py exp/common/default          # 目录 → 其中所有 *.log
    python3 tools/analyze_log.py run.log transcribe.log --top 20
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# 行首：时间戳 + [LEVEL] + 消息体。
_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} \[(\w+)\] (.*)$"
)
# 消息体开头的模块 tag，如 [asr] / [runtime] / [separation]。
_TAG_RE = re.compile(r"^\[([a-z_]+)\]\s*(.*)$")
# 数字（含小数、负数）归一占位。
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

_NO_TAG = "general"


def _event_key(message: str) -> str:
    """从消息体提取归一化的事件名：冒号前缀优先，否则前 5 个词；数字归一。"""

    colon = message.find(":")
    if 0 < colon <= 40:
        key = message[:colon]
    else:
        key = " ".join(message.split()[:5])
    return _NUM_RE.sub("#", key)[:60]


def analyze_file(path: Path) -> dict:
    stats = {
        "path": path,
        "total_lines": 0,
        "continuation_lines": 0,
        "first_ts": None,
        "last_ts": None,
        # module -> event -> level -> count
        "events": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        "issues": [],  # (ts, level, module, message)
    }
    with open(path, encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            stats["total_lines"] += 1
            match = _LINE_RE.match(line.rstrip("\n"))
            if not match:
                stats["continuation_lines"] += 1
                continue
            ts, level, message = match.groups()
            if stats["first_ts"] is None:
                stats["first_ts"] = ts
            stats["last_ts"] = ts
            tag_match = _TAG_RE.match(message)
            if tag_match:
                module, body = tag_match.groups()
            else:
                module, body = _NO_TAG, message
            stats["events"][module][_event_key(body)][level] += 1
            if level in ("WARNING", "ERROR", "CRITICAL"):
                stats["issues"].append((ts, level, module, message))
    return stats


def report(stats: dict, top: int) -> str:
    lines = [
        f"== {stats['path']}",
        f"   lines={stats['total_lines']} "
        f"(continuation={stats['continuation_lines']})",
        f"   span: {stats['first_ts']} -> {stats['last_ts']}",
    ]
    for module in sorted(stats["events"], key=lambda m: (m != _NO_TAG, m)):
        events = stats["events"][module]
        total = sum(sum(levels.values()) for levels in events.values())
        lines.append(f"   [{module}] events={total}")
        ranked = sorted(
            events.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        )
        for event, levels in ranked[:top]:
            level_str = " ".join(
                f"{level}={count}" for level, count in sorted(levels.items())
            )
            lines.append(f"      {sum(levels.values()):>6}  {event:<50} {level_str}")
        if len(ranked) > top:
            lines.append(f"      ... and {len(ranked) - top} more event type(s)")
    if stats["issues"]:
        lines.append(f"   WARNING/ERROR ({len(stats['issues'])}):")
        for ts, level, module, message in stats["issues"]:
            lines.append(f"      {ts} [{level}] {message}")
    return "\n".join(lines)


def _collect_paths(raw_paths: list[str]) -> list[Path]:
    paths = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.log")))
        else:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="运行日志事件统计（模块 × 事件 × 级别）")
    parser.add_argument("logs", nargs="+", help="日志文件或包含 *.log 的目录（可多个）")
    parser.add_argument("--top", type=int, default=15, help="每模块最多展示的事件种类数")
    args = parser.parse_args()

    paths = _collect_paths(args.logs)
    if not paths:
        print("no log files found", file=sys.stderr)
        return 1
    for path in paths:
        if not path.is_file():
            print(f"skip (not a file): {path}", file=sys.stderr)
            continue
        print(report(analyze_file(path), args.top))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
