"""聚类后端（clusterer）接口。

设计要点：

- embedding 提取（extract/）与聚类分配（clusterer）解耦，
  后端通过 `build_clusterer(config)`（见 backends/__init__.py）按 YAML 配置插拔；
- 流式后端（deferred=False）：`assign_chunk` 立即返回最终 local->global 映射，
  调用方逐 chunk 写出 RTTM；
- 离线后端（deferred=True）：`assign_chunk` 只缓冲 observations，
  `finalize()` 统一聚类并返回逐 chunk 的映射，调用方在音频结束后
  用同一 writer 逻辑重放帧级输出。

新增聚类方法：在本包 backends/ 下新建模块实现本接口，
并在 backends/__init__.py 的 `build_clusterer` 注册即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from diarization.contracts import ChunkDebugInfo, ChunkObservation

if TYPE_CHECKING:
    import numpy as np


def resolve_merged(global_id: int, merged_into: dict[int, int]) -> int:
    """沿 merged_into 链解析到最终幸存 id（幸存 id 之后也可能再被并）。

    Args:
        global_id: 待解析的 speaker id。
        merged_into: 流式 merge 链（absorbed id -> survivor id）。

    Returns:
        最终幸存的 speaker id；不在链上时原样返回。
    """

    seen: set[int] = set()
    while global_id in merged_into and global_id not in seen:
        seen.add(global_id)
        global_id = merged_into[global_id]
    return global_id


class BaseChunkClusterer(ABC):
    """chunk 级 local->global 分配后端接口。

    Attributes:
        deferred: False 表示 assign_chunk 立即返回最终 id；True 表示缓冲到
            finalize 统一分配。
        output_tag: 输出 RTTM 文件名后缀（<stem>.<output_tag>.rttm），各后端必须覆盖。
        centroids: L2 归一化的簇质心，供 refined 级做 post-merge 相似度计算。
        counts: 各 speaker 的样本计数（口径由后端自定：streaming 为有效
            centroid 更新次数，ahc 为簇内 observation 数）。
        merged_into: 流式 merge 链（absorbed -> survivor；无流式 merge 的后端恒为空）。
    """

    # False：assign_chunk 立即返回最终 id；True：缓冲到 finalize 统一分配。
    deferred: bool = False
    # 输出 RTTM 文件名后缀：<stem>.<output_tag>.rttm，各后端必须覆盖。
    output_tag: str

    # ---- refined 级（post_process.RefinedRTTMWriter）读取的合并状态 ----
    # 每个后端都必须维护以下三个属性，否则 refined 级退化为 raw 的拷贝
    # （不做 merge 历史修正与小样本合并，sidecar 无计数/uncertain 信息）。
    # L2 归一化的簇质心（post-merge 的相似度计算）。
    centroids: dict[int, np.ndarray] = {}
    # 各 speaker 的样本计数（post_merge_min_count 判定与 uncertain 标记；
    # 口径由后端自定：streaming 为有效 centroid 更新次数，ahc 为簇内
    # observation 数）。
    counts: dict[int, int] = {}
    # 流式 merge 链（absorbed -> survivor；无流式 merge 的后端恒为空）。
    merged_into: dict[int, int] = {}

    @abstractmethod
    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[Optional[dict[int, int]], ChunkDebugInfo]:
        """处理一个 chunk 的 observations。

        Args:
            observations: 本 chunk 的一组 local track 观测（含 embedding 等）。

        Returns:
            (local_to_global, debug_info)；流式后端立即返回最终映射，
            离线后端返回 (None, debug_info)，映射由 finalize() 统一给出。
        """

    def finalize(self) -> list[dict[int, int]]:
        """离线后端在音频结束后统一分配，返回逐 chunk 的 local->global 列表。

        Returns:
            与 assign_chunk 收到的 chunk 一一对应的 local->global 映射列表。

        Raises:
            NotImplementedError: 非 deferred 后端无需 finalize。
        """

        raise NotImplementedError(
            f"{type(self).__name__} 不是 deferred 后端，无需 finalize"
        )


__all__ = ["BaseChunkClusterer", "resolve_merged"]
