"""路径与文件收集工具。"""

from __future__ import annotations

import os
from pathlib import Path


def collect_audio_paths(input_path: str) -> list[str]:
    """收集待处理音频路径。

    支持三种输入形式：
    - 单个音频文件；
    - 音频目录；
    - 文本清单文件，每行一个音频路径。
    """

    path = Path(input_path)
    if path.is_dir():
        items: list[str] = []
        for ext in ("*.wav", "*.mp3", "*.flac"):
            items.extend(str(p) for p in sorted(path.rglob(ext)))
        return items
    if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac"}:
        return [str(path)]
    if path.is_file():
        with open(path, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def ensure_parent_dir(path: str) -> None:
    """确保目标文件的父目录存在。"""

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


__all__ = ["collect_audio_paths", "ensure_parent_dir"]
