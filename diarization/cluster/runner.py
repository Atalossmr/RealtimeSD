"""聚类消费循环：ChunkArtifacts 序列 -> clusterer 分配 -> writer 输出 raw RTTM。

端到端流程（run.py 的 `run_diarization`）与聚类 CLI（cluster/app.py）共用的唯一消费实现，
保证流式与离线（deferred）后端在两条路径上行为一致。

- 流式后端：逐 chunk 分配后即时写出帧级结果并闭合沉默 turn，零缓冲直通；
- deferred（离线）后端：逐 chunk 暂存帧参数，finalize 统一聚类后按序重放；
- refined 级（可选 refiner 参数，两后端通用）：流式后端按需整体重生成
  refined RTTM 与 speakers.json sidecar——本 chunk 发生 merge 事件
  （merged_into 有新增）或距上次刷新满固定 chunk 数时刷新（merge 历史行
  修正 + uncertain 标记随计数累积刷新）；deferred 后端在 EOF 重放完成后
  刷新；两者均在 EOF 的 final 刷新叠加小样本强制合并。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from common.contracts import ChunkArtifacts, ChunkDebugInfo
from .base import BaseChunkClusterer
from .rttm_writer import AppendOnlyRTTMWriter

if TYPE_CHECKING:
    from .post_process import RefinedRTTMWriter


logger = logging.getLogger(__name__)

# 流式 refined 级的兜底刷新间隔（chunk 数）：无 merge 事件时按此周期刷新，
# 避免每 chunk 全量重写 refined RTTM / speakers.json（raw 文件随音频时长
# 线性增长，逐 chunk 全量重写是 O(n²) I/O，小时级音频不可接受）。
# 默认 chunk_duration=10s 下约合每 60s 刷新一次。
REFINER_REFRESH_INTERVAL_CHUNKS = 6

# 每 chunk 处理完后的可选回调（日志、统计等）。
ChunkHook = Callable[
    [ChunkArtifacts, Optional[dict[int, int]], ChunkDebugInfo, int], None
]


def run_clustering(
    artifacts: Iterable[ChunkArtifacts],
    clusterer: BaseChunkClusterer,
    writer: AppendOnlyRTTMWriter,
    chunk_hook: Optional[ChunkHook] = None,
    refiner: Optional[RefinedRTTMWriter] = None,
) -> None:
    """消费 chunk 序列并输出 RTTM（含 writer.finalize）。

    流式后端：逐 chunk 分配后即时写出帧级结果并闭合沉默 turn。
    deferred（离线）后端：逐 chunk 暂存帧参数，finalize 统一聚类后按序重放。

    中途异常时 writer.finalize() 仍会在 finally 中执行（闭合残余 open
    turn），refiner 的 final 刷新则只在正常跑完时执行。

    Args:
        artifacts: ChunkArtifacts 序列（逐 chunk）。
        clusterer: 聚类后端；deferred=False 时 assign_chunk 立即返回最终
            local->global 映射，deferred=True 时只缓冲、由 finalize 统一给出。
        writer: raw 级 RTTM 写出器。
        chunk_hook: 可选的每 chunk 回调（日志、统计等）；deferred 模式下
            分配阶段触发、emitted_frames 恒为 0，重放阶段不再回调。
        refiner: 可选的 RefinedRTTMWriter（两后端通用）。流式后端按需刷新——
            本 chunk 发生 merge 事件（merged_into 有新增）或距上次刷新满
            REFINER_REFRESH_INTERVAL_CHUNKS 个 chunk 时，按最新合并状态重生成
            refined RTTM 与 speaker 状态 sidecar；deferred 后端在 EOF 重放完成
            后刷新。EOF 的 final 刷新叠加小样本合并。

    Raises:
        RuntimeError: deferred 后端 finalize 返回的分配数与暂存 chunk 数不一致。
    """

    pending: list[ChunkArtifacts] = []
    completed = False
    # refined 级刷新状态：上次刷新时的流式 merge 数与之后经过的 chunk 数。
    last_merge_count = 0
    chunks_since_refresh = 0
    try:
        for chunk in artifacts:
            # 流式后端立即返回最终 local->global；deferred 后端返回 None 并内部缓冲。
            local_to_global, debug_info = clusterer.assign_chunk(chunk.observations)
            if clusterer.deferred:
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
            # refined 级按需重生成：本 chunk 发生 merge 事件（merged_into
            # 有新增）时立即刷新，历史行修正即时生效；否则按固定 chunk 间隔
            # 兜底刷新，让 sidecar 的 speaker 时长/uncertain 标记随时长累积
            # 更新，同时避免逐 chunk 全量重写的 O(n²) I/O。
            if refiner is not None:
                merge_count = len(clusterer.merged_into)
                chunks_since_refresh += 1
                if (
                    merge_count != last_merge_count
                    or chunks_since_refresh >= REFINER_REFRESH_INTERVAL_CHUNKS
                ):
                    refiner.refresh()
                    last_merge_count = merge_count
                    chunks_since_refresh = 0
            if chunk_hook is not None:
                chunk_hook(chunk, local_to_global, debug_info, emitted_frames)

        if clusterer.deferred:
            # 音频结束：离线后端统一聚类，返回与暂存 chunk 一一对应的分配列表。
            assignments = clusterer.finalize()
            if len(assignments) != len(pending):
                raise RuntimeError(
                    f"clusterer.finalize 返回 {len(assignments)} 个 chunk 的分配，"
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
        # 中途异常也要收尾：闭合残余 open turn（纯追加，零重写），
        # 否则已写出的 turn 不闭合、输出不可用于评估。
        writer.finalize()

        # EOF：refined 级最终刷新，叠加小样本强制合并（post-merge）。
        # 仅在正常跑完时执行：异常路径下避免 final 刷新出错掩盖原始异常，
        # 此时 refined 保持最后一次按需刷新的状态，仍然可用。
        if refiner is not None and completed:
            refiner.refresh(final=True)


__all__ = ["run_clustering", "ChunkHook"]
