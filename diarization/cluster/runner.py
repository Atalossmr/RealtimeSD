"""聚类消费循环：ChunkArtifacts 序列 -> assigner 分配 -> writer 输出 raw RTTM。

端到端流程（diarization/pipeline.py）与聚类 CLI（cluster/app.py）共用的唯一消费实现，
保证流式与离线（deferred）后端在两条路径上行为一致。

流式后端的 new-speaker hold（config.new_speaker_hold_chunks > 0 时启用）：

- 某 chunk 新建了 global speaker 时进入 hold：该 chunk 及后续 chunk 的
  分配结果先缓存，不喂 writer / chunk_hook（RTTM 与 exporter 同步等待）；
- 缓刑中的 speaker 全部被 merge 掉，或存活满 hold_chunks 时，缓存的若干
  chunk 经 merged_into 链式重映射后按原序一起输出——false split 的帧在
  写出前即归属幸存 speaker，已写出的行仍不受影响（append-only 不变）；
- 无新 speaker 的普通 chunk 零延迟直通；EOF 时强制 flush。

refined 级（可选 refiner 参数，仅流式后端）：每个 chunk 写出后按最新合并
状态整体重生成 refined RTTM 与 speakers.json sidecar（merge 历史行修正 +
uncertain 标记随时长累积刷新），EOF 时 final 刷新叠加小样本强制合并。
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
    refiner=None,
) -> None:
    """消费 chunk 序列并输出 RTTM（含 writer.finalize）。

    流式后端：逐 chunk 分配后即时写出帧级结果并闭合沉默 turn；
    开启 new-speaker hold 时，新建 speaker 触发的缓存区延迟到 merge
    判定尘埃落定后统一输出（映射经 merged_into 重映射）。
    deferred（离线）后端：逐 chunk 暂存帧参数，finalize 统一聚类后按序重放。

    refiner（可选，仅流式后端）：RefinedRTTMWriter，逐 chunk 按最新合并
    状态重生成 refined RTTM 与 speaker 状态 sidecar，EOF 时 final 刷新
    叠加小样本合并。
    """

    # ---- new-speaker hold 状态（仅流式后端） ----
    hold_window = 0
    merged_into: dict[int, int] = {}
    if not assigner.deferred:
        config = getattr(assigner, "config", None)
        hold_window = max(
            0, int(getattr(config, "new_speaker_hold_chunks", 0) or 0)
        )
        merged_into = getattr(assigner, "merged_into", {})
    # 缓存的 (chunk, local_to_global, debug_info)，保持原始顺序。
    held: list[tuple[ChunkArtifacts, dict[int, int], ChunkDebugInfo]] = []
    hold_chunks_left = 0
    # 缓刑中的 new speaker id 集：被 merge 即移出；清空或超时即整体 flush。
    probationary: set[int] = set()

    def resolve(global_id: int) -> int:
        # 沿 merged_into 链解析到最终幸存 id（幸存 id 之后也可能再被并）。
        seen: set[int] = set()
        while global_id in merged_into and global_id not in seen:
            seen.add(global_id)
            global_id = merged_into[global_id]
        return global_id

    def emit(
        chunk: ChunkArtifacts,
        local_to_global: dict[int, int],
        debug_info: ChunkDebugInfo,
    ) -> None:
        """写出一个 chunk 的提交区（映射先经 merged_into 重映射）并触发 hook。"""

        remapped = {
            int(local_idx): resolve(global_id)
            for local_idx, global_id in local_to_global.items()
        }
        emitted_frames = writer.consume_chunk(
            chunk.seg_scores,
            chunk.frame_step,
            chunk.chunk_start,
            chunk.commit_start,
            chunk.commit_end,
            remapped,
        )
        writer.close_inactive(chunk.commit_end)
        # refined 级逐 chunk 重生成：merge 事件的历史行修正即时生效，
        # sidecar 的 speaker 时长/uncertain 标记也随时长累积刷新。
        if refiner is not None:
            refiner.refresh()
        if chunk_hook is not None:
            chunk_hook(chunk, remapped, debug_info, emitted_frames)

    def flush_held() -> None:
        """hold 结束：缓存的 chunk 按原序一起输出。"""

        nonlocal hold_chunks_left
        if held:
            logger.info(
                "[hold] flushing %d buffered chunk(s) (chunk %d..%d)",
                len(held),
                held[0][0].chunk_index,
                held[-1][0].chunk_index,
            )
        for chunk, local_to_global, debug_info in held:
            emit(chunk, local_to_global, debug_info)
        held.clear()
        probationary.clear()
        hold_chunks_left = 0

    pending: list[ChunkArtifacts] = []
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
        if hold_window <= 0:
            # hold 关闭：零缓冲直通。
            emit(chunk, local_to_global, debug_info)
            continue

        new_ids = {rec["global"] for rec in debug_info["new_speakers"]}
        if hold_chunks_left <= 0 and not new_ids:
            # 无新 speaker 的普通 chunk：零延迟直通。
            emit(chunk, local_to_global, debug_info)
            continue

        # 新建 speaker 触发 hold：窗口以第一个新 speaker 为锚，不再延长，
        # 保证最大额外延迟为 hold_window 个 chunk。
        if hold_chunks_left <= 0:
            hold_chunks_left = hold_window
        probationary |= new_ids
        held.append((chunk, local_to_global, debug_info))
        # 本 chunk 内已被 merge 的（含新建当 chunk 即被并）立即移出考察集。
        probationary = {gid for gid in probationary if gid not in merged_into}
        hold_chunks_left -= 1
        if not probationary or hold_chunks_left <= 0:
            flush_held()

    # EOF：hold 缓存兜底输出（缓刑未满也按当前映射定案）。
    flush_held()

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

    # EOF：refined 级最终刷新，叠加小样本强制合并（post-merge）。
    if refiner is not None:
        refiner.refresh(final=True)


__all__ = ["run_clustering", "ChunkHook"]
