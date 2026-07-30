"""聚类消费循环：ChunkArtifacts 序列 -> assigner 分配 -> writer 输出 RTTM。

端到端流程（diarization/pipeline.py）与聚类 CLI（cluster/app.py）共用的唯一消费实现，
保证流式与离线（deferred）后端在两条路径上行为一致。
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

from ..schema import ChunkArtifacts, ChunkDebugInfo
from .assigners import BaseChunkAssigner
from .rttm_writer import AppendOnlyRTTMWriter


logger = logging.getLogger(__name__)

# 每 chunk 处理完后的可选回调（日志、统计等）。
ChunkHook = Callable[
    [ChunkArtifacts, Optional[dict[int, int]], ChunkDebugInfo, int], None
]


def run_clustering(
    artifacts: Iterable[ChunkArtifacts],
    assigner: BaseChunkAssigner,
    writer: AppendOnlyRTTMWriter,
    chunk_hook: Optional[ChunkHook] = None,
) -> None:
    """消费 chunk 序列并输出 RTTM（含 writer.finalize）。

    流式后端：逐 chunk 分配后即时写出帧级结果并闭合沉默 turn；
    deferred（离线）后端：逐 chunk 暂存帧参数，finalize 统一聚类后按序重放。
    """

    pending: list[ChunkArtifacts] = []
    for chunk in artifacts:
        local_to_global, debug_info = assigner.assign_chunk(chunk.observations)
        if assigner.deferred:
            pending.append(chunk)
            emitted_frames = 0
        else:
            emitted_frames = writer.consume_chunk(
                chunk.seg_scores,
                chunk.frame_step,
                chunk.chunk_start,
                chunk.commit_start,
                chunk.commit_end,
                local_to_global,
            )
            writer.close_inactive(chunk.commit_end)

        if chunk_hook is not None:
            chunk_hook(chunk, local_to_global, debug_info, emitted_frames)

    if assigner.deferred:
        assignments = assigner.finalize()
        if len(assignments) != len(pending):
            raise RuntimeError(
                f"assigner.finalize 返回 {len(assignments)} 个 chunk 的分配，"
                f"与暂存的 {len(pending)} 个不一致"
            )
        for chunk, local_to_global in zip(pending, assignments):
            writer.consume_chunk(
                chunk.seg_scores,
                chunk.frame_step,
                chunk.chunk_start,
                chunk.commit_start,
                chunk.commit_end,
                local_to_global,
            )
            writer.close_inactive(chunk.commit_end)

    writer.finalize()


__all__ = ["run_clustering", "ChunkHook"]
