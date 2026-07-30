"""聚类后端（assigner）接口。

设计要点：

- embedding 提取（extract/）与聚类分配（assigner）解耦，
  后端通过 `build_assigner(config)`（见 backends/__init__.py）按 YAML 配置插拔；
- 流式后端（deferred=False）：`assign_chunk` 立即返回最终 local->global 映射，
  调用方逐 chunk 写出 RTTM；
- 离线后端（deferred=True）：`assign_chunk` 只缓冲 observations，
  `finalize()` 统一聚类并返回逐 chunk 的映射，调用方在音频结束后
  用同一 writer 逻辑重放帧级输出。

新增聚类方法：在本包 backends/ 下新建模块实现本接口，
并在 backends/__init__.py 的 `build_assigner` 注册即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..schema import ChunkDebugInfo, ChunkObservation


class BaseChunkAssigner(ABC):
    """chunk 级 local->global 分配后端接口。"""

    # False：assign_chunk 立即返回最终 id；True：缓冲到 finalize 统一分配。
    deferred: bool = False
    # 输出 RTTM 文件名后缀：<stem>.<output_tag>.rttm。
    output_tag: str = "streaming"

    @abstractmethod
    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[Optional[dict[int, int]], ChunkDebugInfo]:
        """处理一个 chunk 的 observations。

        流式后端返回 (local_to_global, debug_info)；
        离线后端返回 (None, debug_info)，映射由 finalize() 统一给出。
        """

    def finalize(self) -> list[dict[int, int]]:
        """离线后端在音频结束后统一分配，返回逐 chunk 的 local->global 列表。"""

        raise NotImplementedError(
            f"{type(self).__name__} 不是 deferred 后端，无需 finalize"
        )


__all__ = ["BaseChunkAssigner"]
