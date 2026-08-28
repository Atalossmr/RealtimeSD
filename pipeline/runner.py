"""diarization 执行器：组合 stage1（extract+cluster+RTTM）与 stage2（separation 导出）。

DiarizationRunner 是管线的组合核心：持有 ChunkExtractor（模型只加载一次，
可连续处理多条音频），每条音频独立构建 clusterer / writer / exporter
（状态不跨文件），separation 的 StreamingSegmentExporter 以 chunk hook
形式挂接在聚类循环上。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from common.speakerlab.utils.fileio import load_audio

from common.config import ChunkPipelineConfig
from common.contracts import ChunkArtifacts, ChunkDebugInfo, ChunkObservation
from common.utils import log_structured
from diarization.cluster.backends import build_clusterer
from diarization.cluster.post_process import RefinedRTTMWriter
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter
from diarization.cluster.runner import run_clustering
from diarization.extract.extractor import ChunkExtractor

logger = logging.getLogger(__name__)


def _log_debug_chunk(
    *,
    chunk_index: int,
    chunk_start: float,
    commit_start: float,
    commit_end: float,
    seg_scores: np.ndarray,
    observations: list[ChunkObservation],
    local_to_global: dict[int, int],
    debug_info: ChunkDebugInfo,
    emitted_frames: int,
) -> None:
    """输出 chunk 级调试信息"""

    debug_summary = {
        "chunk_index": int(chunk_index),
        "chunk": {
            "start": round(float(chunk_start), 3),
            "commit_start": round(float(commit_start), 3),
            "end": round(float(commit_end), 3),
        },
        "segmentation_summary": {
            "shape": [int(dim) for dim in seg_scores.shape],
            "min": round(float(np.min(seg_scores)), 6),
            "max": round(float(np.max(seg_scores)), 6),
            "mean": round(float(np.mean(seg_scores)), 6),
        },
        "window_state": {
            "observations": int(len(observations)),
            "embedded": int(len(observations)),
            "emitted": int(emitted_frames),
        },
        "assignment": {
            "local_to_global": {
                str(local_idx): int(global_id)
                for local_idx, global_id in sorted(local_to_global.items())
            },
            "local_assignments": debug_info["local_assignments"],
        },
        "centroids": {
            "before": int(debug_info["num_centroids_before"]),
            "after": int(debug_info["num_centroids_after"]),
        },
    }
    log_structured(logger, logging.DEBUG, "[debug]", "window_summary", debug_summary)

    for key in ("new_speakers", "updated_speakers", "skipped_updates"):
        if debug_info[key]:
            log_structured(logger, logging.DEBUG, "[debug]", key, debug_info[key])
    if debug_info["global_speakers"]:
        log_structured(
            logger,
            logging.INFO,
            "[runtime]",
            "current_global_speakers",
            debug_info["global_speakers"],
        )


class DiarizationRunner:
    """可复用的 diarization 执行器：模型加载一次，可连续处理多条音频。

    每条音频独立构建 clusterer / writer / exporter（状态不跨文件），
    仅 extractor 持有的模型（segmentation + embedding）跨文件复用。
    with_separation=False 时不创建分段音频导出器（无 segments 输出）。
    """

    def __init__(
        self,
        config: ChunkPipelineConfig,
        model_path: Optional[str] = None,
        with_separation: bool = True,
    ):
        self.config = config
        self.with_separation = with_separation
        self.extractor = ChunkExtractor(config, model_path)

    def process_file(self, audio_path: str, exp_dir: Path) -> str:
        """处理单条音频，输出 RTTM（+ 可选 segments）；返回 raw RTTM 路径。"""

        config = self.config
        config.output_dir = str(exp_dir)

        clusterer = build_clusterer(config)

        waveform = load_audio(audio_path, obj_fs=config.sample_rate)
        waveform = self.extractor.prepare_waveform(waveform, config.sample_rate)

        uri = Path(audio_path).stem
        raw_rttm_path = str(exp_dir / "rttm" / f"{uri}.{clusterer.output_tag}.rttm")
        writer = AppendOnlyRTTMWriter(
            raw_rttm_path,
            uri,
            config.min_segment_duration,
            config.segment_merge_gap,
            config.show_rttm,
        )

        # 音频片段导出：仅 streaming 后端下逐 commit 区生效。
        # 默认 WavSegmentSink 落盘 wav + manifest，转写由 ASR 跟随进程
        # 读取输出目录独立完成；TIGER 仍是按需惰性加载。
        exporter = None
        if config.separation_enabled and self.with_separation:
            if clusterer.deferred:
                logger.warning(
                    "[separation] separation 仅支持 streaming 后端，当前后端 %s 已跳过",
                    type(clusterer).__name__,
                )
            else:
                # 刻意惰性 import：separation 依赖 TIGER 等重库，仅 stage>=2 需要。
                from separation import StreamingSegmentExporter

                exporter = StreamingSegmentExporter(
                    config,
                    waveform,
                    self.extractor.embedder,
                    uri=uri,
                    output_dir=str(exp_dir / "segments"),
                )

        def chunk_log_hook(
            chunk: ChunkArtifacts,
            local_to_global: Optional[dict[int, int]],
            debug_info: ChunkDebugInfo,
            emitted_frames: int,
        ) -> None:
            """# 每 chunk 分配完成后的回调，负责日志与分段导出"""

            # 输出本身由 runner/writer 完成。deferred 后端下 local_to_global
            # 为 None、emitted_frames 为 0（重放阶段不再回调）。
            # 分段音频导出：commit 区音频段输出（重叠区经 TIGER 分离）。
            if exporter is not None and local_to_global:
                exporter.handle_chunk(chunk, local_to_global)
            log_structured(
                logger,
                logging.INFO,
                "[runtime]",
                "frame_decision",
                {
                    "chunk_index": int(chunk.chunk_index),
                    "chunk_start": round(float(chunk.chunk_start), 3),
                    "commit": [
                        round(float(chunk.commit_start), 3),
                        round(float(chunk.commit_end), 3),
                    ],
                    "local_to_global": {
                        str(local_idx): int(global_id)
                        for local_idx, global_id in sorted(
                            (local_to_global or {}).items()
                        )
                    },
                },
            )
            if config.debug:
                _log_debug_chunk(
                    chunk_index=chunk.chunk_index,
                    chunk_start=chunk.chunk_start,
                    commit_start=chunk.commit_start,
                    commit_end=chunk.commit_end,
                    seg_scores=chunk.seg_scores,
                    observations=chunk.observations,
                    local_to_global=local_to_global or {},
                    debug_info=debug_info,
                    emitted_frames=emitted_frames,
                )

        # refined 级（两后端通用）：streaming 逐 chunk + merge 事件动态重生成，
        # ahc（deferred）在 EOF 重放完成后最终刷新；均在 EOF 叠加小样本合并。
        base = raw_rttm_path[: -len(f".{clusterer.output_tag}.rttm")]
        refiner = RefinedRTTMWriter(
            raw_rttm_path,
            f"{base}.refined.rttm",
            clusterer,
            min_count=config.post_merge_min_count,
            min_similarity=config.post_merge_min_similarity,
        )

        # 生成器惰性生产 chunk：streaming 后端下输出延迟 ≈ 一个 hop，
        # 不会因为组合成统一循环而引入额外的缓冲。
        try:
            run_clustering(
                self.extractor.iter_chunk_artifacts(waveform),
                clusterer,
                writer,
                chunk_hook=chunk_log_hook,
                refiner=refiner,
            )
        finally:
            # 闭合 exporter 残余的 open segment（音频结束，与 writer.finalize 同步）；
            # 中途异常也要收尾，否则已累积的段音频不落盘。
            if exporter is not None:
                exporter.finalize()
        return raw_rttm_path


def run_diarization(
    config: ChunkPipelineConfig,
    audio_path: str,
    exp_dir: Path,
    model_path: Optional[str] = None,
    with_separation: bool = True,
) -> str:
    """单条音频的 diarization 管线：extract + cluster + RTTM/segments 输出。

    供 diarization.app（stage1 独立 CLI）复用；返回 raw RTTM 路径。
    批量场景请直接用 DiarizationRunner 以复用已加载的模型。
    """

    return DiarizationRunner(config, model_path, with_separation).process_file(
        audio_path, exp_dir
    )


__all__ = ["DiarizationRunner", "run_diarization"]
