"""设备解析工具（diarization / tools 共用单一来源）。"""

from __future__ import annotations

import torch


def resolve_device_name(device: str) -> str:
    """把用户配置的设备字符串解析成规范设备名（"auto" → 显式序号）。

    "auto" 解析为 "cuda:0"（而非 "cuda"）：显式钉 0 号卡；仓库内无人调用
    torch.cuda.set_device，与默认设备语义等价。

    Args:
        device: 用户配置的设备字符串，"auto" 表示自动选择。

    Returns:
        规范设备名，如 "cuda:0" 或 "cpu"。
    """

    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def resolve_device(device: str) -> torch.device:
    """把用户配置的设备字符串解析成 `torch.device`。

    Args:
        device: 用户配置的设备字符串，"auto" 表示自动选择。

    Returns:
        对应的 torch.device。
    """

    return torch.device(resolve_device_name(device))


__all__ = ["resolve_device", "resolve_device_name"]
