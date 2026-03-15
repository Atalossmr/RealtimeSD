"""在线说话人分离的通用工具函数。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio


logger = logging.getLogger(__name__)


def setup_logger(verbose: bool) -> None:
    """初始化日志系统。

    这里故意保持全局 `basicConfig` 方式，原因是这个项目当前主要通过 CLI 单进程运行，
    这种形式最直观，也方便 shell 脚本直接收集 stdout/stderr。
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def resolve_device(device: str) -> torch.device:
    """把用户配置的设备字符串解析成 `torch.device`。"""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """对 numpy 向量做 L2 单位化。

    当前聚类逻辑大量依赖余弦相似度，因此把向量保持为单位范数能让后续点积更稳定。
    """

    denom = np.linalg.norm(vec)
    if denom <= 0:
        return vec
    return vec / denom


def resample_waveform_if_needed(
    waveform: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    """必要时把音频重采样到目标采样率。"""

    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


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
