"""聚类消费循环：ChunkArtifacts 序列 -> assigner 分配 -> writer 输出 raw RTTM。

端到端流程（diarization/pipeline.py）与聚类 CLI（cluster/app.py）共用的唯一消费实现，
保证流式与离线（deferred）后端在两条路径上行为一致。

- 流式后端：逐 chunk 分配后即时写出帧级结果并闭合沉默 turn，零缓冲直通；
- deferred（离线）后端：逐 chunk 暂存帧参数，finalize 统一聚类后按序重放；
- refined 级（可选 refiner 参数，仅流式后端）：每个 chunk 写出后按最新合并
  状态整体重生成 refined RTTM 与 speakers.json sidecar（merge 历史行修正 +
  uncertain 标记随时长累积刷新），EOF 时 final 刷新叠加小样本强制合并。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from ..schema import ChunkArtifacts, ChunkDebugInfo
from .base import BaseChunkAssigner
from .rttm_writer import AppendOnlyRTTMWriter

if TYPE_CHECKING:
    from .post_merge import RefinedRTTMWriter


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
    refiner: Optional[RefinedRTTMWriter] = None,
) -> None:
    """消费 chunk 序列并输出 RTTM（含 writer.finalize）。

    流式后端：逐 chunk 分配后即时写出帧级结果并闭合沉默 turn。
    deferred（离线）后端：逐 chunk 暂存帧参数，finalize 统一聚类后按序重放。

    refiner（可选，仅流式后端）：RefinedRTTMWriter，逐 chunk 按最新合并
    状态重生成 refined RTTM 与 speaker 状态 sidecar，EOF 时 final 刷新
    叠加小样本合并。

    中途异常时 writer.finalize() 仍会在 finally 中执行（闭合残余 open
    turn、追加 id 映射表），refiner 的 final 刷新则只在正常跑完时执行。
    """

    pending: list[ChunkArtifacts] = []
    completed = False
    try:
        for chunk in artifacts:
            # 流式后端立即返回最终 local->global；deferred 后端返回 None 并内部缓冲。
            local_to_global, debug_info = assigner.assign_chunk(chunk.observations)
            if assigner.deferred:
                # 离线后端此刻还没有标签：整段帧参数暂存，待 finalize 后统一重放。
                pending.append(chunk)
                # hook 在分配之后触发；deferred 模式下 emitted_frames 恒为 0
                # （真正的写出发生在下方重放阶段，不再回调）。
                if chunk_hook is not None:
                    chunk_hook(chunk, None, debug_info, 0)
                continue

            assert local_to_global is not None
            emitted_frames = writer.consume_chunk(
                chunk.seg_scores,
                chunk.frame_step,
                chunk.chunk_start,
                chunk.commit_start,
                chunk.commit_end,
                local_to_global,
            )
            writer.close_inactive(chunk.commit_end)
            # refined 级逐 chunk 重生成：merge 事件的历史行修正即时生效，
            # sidecar 的 speaker 时长/uncertain 标记也随时长累积刷新。
            if refiner is not None:
                refiner.refresh()
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
        completed = True
    finally:
        # 中途异常也要收尾：闭合残余 open turn 并追加 id 映射表（纯追加，
        # 零重写），否则已写出的 turn 不闭合、输出不可用于评估。
        writer.finalize()

        # EOF：refined 级最终刷新，叠加小样本强制合并（post-merge）。
        # 仅在正常跑完时执行：异常路径下避免 final 刷新出错掩盖原始异常，
        # 此时 refined 保持最后一次逐 chunk 刷新的状态，仍然可用。
        if refiner is not None and completed:
            refiner.refresh(final=True)


__all__ = ["run_clustering", "ChunkHook"]
