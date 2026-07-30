"""聚类消费循环：ChunkArtifacts 序列 -> assigner 分配 -> writer 输出 RTTM。

端到端流程（diarization/pipeline.py）与聚类 CLI（cluster/app.py）共用的唯一消费实现，
保证流式与离线（deferred）后端在两条路径上行为一致。
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

from ..schema import ChunkArtifacts, ChunkDebugInfo
from .base import BaseChunkAssigner
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
        # 流式后端立即返回最终 local->global；deferred 后端返回 None 并内部缓冲。
        local_to_global, debug_info = assigner.assign_chunk(chunk.observations)
        if assigner.deferred:
            # 离线后端此刻还没有标签：整段帧参数暂存，待 finalize 后统一重放。
            pending.append(chunk)
            emitted_frames = 0
        else:
            # 只写提交区 [commit_start, commit_end) 的帧；
            # 随后闭合已确认沉默的 open turn（提前写出，降低尾段延迟）。
            emitted_frames = writer.consume_chunk(
                chunk.seg_scores,
                chunk.frame_step,
                chunk.chunk_start,
                chunk.commit_start,
                chunk.commit_end,
                local_to_global,
            )
            writer.close_inactive(chunk.commit_end)

        # hook 在分配之后、输出之后触发；deferred 模式下 emitted_frames 恒为 0
        # （真正的写出发生在下方重放阶段，不再回调）。
        if chunk_hook is not None:
            chunk_hook(chunk, local_to_global, debug_info, emitted_frames)

    if assigner.deferred:
        # 音频结束：离线后端统一聚类，返回与暂存 chunk 一一对应的分配列表。
        assignments = assigner.finalize()
        if len(assignments) != len(pending):
            raise RuntimeError(
                f"assigner.finalize 返回 {len(assignments)} 个 chunk 的分配，"
                f"与暂存的 {len(pending)} 个不一致"
            )
        # 按原始 chunk 顺序重放：与流式路径完全相同的 writer 调用序列，
        # 因此两种后端的输出只取决于分配结果，时间线构造逻辑一致。
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

    # 闭合残余 open turn 并追加 id 映射表（纯追加，零重写）。
    writer.finalize()


__all__ = ["run_clustering", "ChunkHook"]
