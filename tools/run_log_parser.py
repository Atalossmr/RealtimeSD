#!/usr/bin/env python3
"""run.log 结构化事件解析器（tools/ 下各分析脚本的共享入口）。

日志格式约定（见 diarization/utils/log.py 的 log_structured）：

    2026-08-08 12:00:00,000 [INFO] [prefix] title:
    { ... indent=2 的 JSON payload ... }

`iter_log_events` 逐事件产出 (prefix, title, payload)，非结构化行
（纯文本日志、RTTM 行等）自动跳过。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional


# 行尾的事件 marker：`[prefix] title:`（前缀时间戳与级别不进入匹配）。
MARKER_RE = re.compile(r"\[(?P<prefix>[a-zA-Z]+)\] (?P<title>[a-zA-Z_]+):\s*$")


def _read_json_block(lines: Iterator[str]) -> Optional[object]:
    """从日志行迭代器中读取 marker 之后的 JSON 块（容忍嵌套与字符串内的括号）。"""

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


def iter_log_events(
    log_path: str | Path,
) -> Iterator[tuple[str, str, object]]:
    """逐事件解析 run.log，产出 (prefix, title, payload)。"""

    with open(log_path, "r", encoding="utf-8") as file_obj:
        lines: Iterator[str] = iter(file_obj)
        for line in lines:
            marker = MARKER_RE.search(line)
            if marker is None:
                continue
            payload = _read_json_block(lines)
            if payload is None:
                continue
            yield marker.group("prefix"), marker.group("title"), payload


def basic_stats(values: list[float]) -> dict[str, float]:
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


__all__ = ["iter_log_events", "basic_stats", "MARKER_RE"]
