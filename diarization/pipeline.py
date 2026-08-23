"""端到端组合层：ChunkDiarizationPipeline = 提取（extract） + 聚类输出（cluster）。

每个 10s chunk 的处理流程：

1. `extract.ChunkExtractor` 切 chunk、跑 segmentation-3.0、聚合 track 并提
   ERes2NetV2 embedding，产出 ChunkArtifacts；
2. `cluster.runner.run_clustering` 消费 chunk 序列：clusterer 做 local->global
   分配（后端可插拔），流式后端逐 chunk 即时写出 raw RTTM 并由 refined 级
   逐 chunk 重生成 refined RTTM + speaker 状态 sidecar，离线后端音频结束后
   统一聚类重放；
3. 音频结束后 writer 纯追加收尾（闭合全部残余 open turn），
   全程零重写。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from speakerlab.utils.fileio import load_audio

from .cluster.backends import build_clusterer
from .cluster.runner import run_clustering
from .cluster.rttm_writer import AppendOnlyRTTMWriter
from .config import ChunkPipelineConfig
from .extract.extractor import ChunkExtractor
from .schema import ChunkArtifacts, ChunkDebugInfo, ChunkObservation
from .utils import log_structured


logger = logging.getLogger(__name__)


class ChunkDiarizationPipeline:
    """chunk 级实时说话人分离主控类（提取与聚类的端到端组合）。"""

    def __init__(
        self,
        config: ChunkPipelineConfig,
        embedding_model_path: Optional[str],
    ):
        self.config = config
        self.extractor = ChunkExtractor(config, embedding_model_path)
        self.clusterer = build_clusterer(config)

    def reset(self) -> None:
        """为新音频重置全局聚类状态。"""

        self.clusterer = build_clusterer(self.config)

    # ------------------------------------------------------------------
    # 日志辅助
    # ------------------------------------------------------------------

    def _log_debug_chunk(
        self,
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
        log_structured(
            logger, logging.DEBUG, "[debug]", "window_summary", debug_summary
        )

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

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        raw_rttm_path: str,
        uri: Optional[str] = None,
    ) -> None:
        """按 chunk 顺序处理整段波形并持续写出 RTTM。"""

        waveform = self.extractor.prepare_waveform(waveform, sample_rate)

        writer = AppendOnlyRTTMWriter(
            raw_rttm_path,
            uri or "unknown",
            self.config.min_segment_duration,
            self.config.segment_merge_gap,
            self.config.show_rttm,
        )

        # 音频片段导出：仅 streaming 后端下逐 commit 区生效。
        # 默认 WavSegmentSink 落盘 wav + manifest，转写由离线阶段
        # python -m asr.app 读取输出目录独立完成；TIGER 仍是按需惰性加载。
        exporter = None
        if self.config.separation_enabled:
            if self.clusterer.deferred:
                logger.warning(
                    "[separation] separation 仅支持 streaming 后端，当前后端 %s 已跳过",
                    type(self.clusterer).__name__,
                )
            else:
                from .separation import StreamingSegmentExporter

                exporter = StreamingSegmentExporter(
                    self.config,
                    waveform,
                    self.extractor.embedder,
                    uri=uri or "unknown",
                    output_dir=str(Path(self.config.output_dir) / "segments"),
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
            if self.config.debug:
                self._log_debug_chunk(
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
        from .cluster.post_process import RefinedRTTMWriter

        base = raw_rttm_path[: -len(f".{self.clusterer.output_tag}.rttm")]
        refiner = RefinedRTTMWriter(
            raw_rttm_path,
            f"{base}.refined.rttm",
            self.clusterer,
            min_count=self.config.post_merge_min_count,
            min_similarity=self.config.post_merge_min_similarity,
        )

        # 生成器惰性生产 chunk：streaming 后端下输出延迟 ≈ 一个 hop，
        # 不会因为组合成统一循环而引入额外的缓冲。
        try:
            run_clustering(
                self.extractor.iter_chunk_artifacts(waveform),
                self.clusterer,
                writer,
                chunk_hook=chunk_log_hook,
                refiner=refiner,
            )
        finally:
            # 闭合 exporter 残余的 open segment（音频结束，与 writer.finalize 同步）；
            # 中途异常也要收尾，否则已累积的段音频不落盘。
            if exporter is not None:
                exporter.finalize()

    def process_file(self, wav_path: str) -> str:
        """处理单个音频文件并返回生成的 RTTM 路径。"""

        self.reset()
        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)

        if self.config.output_dir is None:
            raise ValueError("output_dir must be set before processing audio")
        raw_rttm_path = str(
            Path(self.config.output_dir)
            / "rttm"
            / f"{Path(wav_path).stem}.{self.clusterer.output_tag}.rttm"
        )

        self.process_waveform(
            waveform,
            self.config.sample_rate,
            raw_rttm_path=raw_rttm_path,
            uri=Path(wav_path).stem,
        )
        return raw_rttm_path


__all__ = ["ChunkDiarizationPipeline"]
