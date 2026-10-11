"""数值工具。"""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """对 numpy 向量做 L2 单位化。

    当前聚类逻辑大量依赖余弦相似度，因此把向量保持为单位范数能让后续点积更稳定。

    Args:
        vec: 输入向量。

    Returns:
        单位化后的向量；零向量原样返回。
    """

    denom = np.linalg.norm(vec)
    if denom <= 0:
        return vec
    return vec / denom


__all__ = ["l2_normalize"]
