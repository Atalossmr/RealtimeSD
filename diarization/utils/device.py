"""设备解析工具。"""

from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """把用户配置的设备字符串解析成 `torch.device`。"""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


__all__ = ["resolve_device"]
