"""端到端组合层：ChunkDiarizationPipeline = 提取（extract） + 聚类输出（cluster）。

每个 10s chunk 的处理流程：

1. `extract.ChunkExtractor` 切 chunk、跑 segmentation-3.0、聚合 track 并提
   ERes2NetV2 embedding，产出 ChunkArtifacts；
2. `cluster.runner.run_clustering` 消费 chunk 序列：assigner 做 local->global
   分配（后端可插拔），流式后端逐 chunk 即时写出，离线后端音频结束后统一
   聚类重放；
3. 音频结束后 writer 纯追加收尾，并在文件末尾以 # 注释写出内部
   global id -> RTTM speaker 映射表（全程零重写）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from speakerlab.utils.fileio import load_audio

from .cluster.assigners import build_assigner
from .cluster.runner import run_clustering
from .cluster.rttm_writer import AppendOnlyRTTMWriter
from .config import ChunkPipelineConfig
from .extract.extractor import ChunkExtractor
from .schema import ChunkArtifacts, ChunkDebugInfo, ChunkObservation


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
        self.assigner = build_assigner(config)

    def reset(self) -> None:
        """为新音频重置全局聚类状态。"""

        self.assigner = build_assigner(self.config)

    # ------------------------------------------------------------------
    # 日志辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _format_log_payload(payload: object) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _log_structured(
        self, level: int, prefix: str, title: str, payload: object
    ) -> None:
        logger.log(level, "%s %s:\n%s", prefix, title, self._format_log_payload(payload))

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
        """输出 chunk 级调试信息（字段与旧版 window_summary 对齐，便于日志分析工具复用）。"""

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
        self._log_structured(logging.DEBUG, "[debug]", "window_summary", debug_summary)

        for key in ("new_speakers", "updated_speakers", "skipped_updates"):
            if debug_info[key]:
                self._log_structured(logging.DEBUG, "[debug]", key, debug_info[key])
        if debug_info["global_speakers"]:
            self._log_structured(
                logging.INFO,
                "[runtime]",
                "current_global_speakers",
                debug_info["global_speakers"],
            )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    @staticmethod
    def _save_embeddings(
        observations: list[ChunkObservation],
        streaming_log_path: str,
    ) -> None:
        """把本文件全部 observation 的 embedding 落盘为 <stem>.embeddings.npz。"""

        if not observations:
            return
        # streaming_log_path 形如 <stem>.<tag>.rttm，去掉两段后缀换成 .embeddings.npz。
        embeddings_path = Path(streaming_log_path).with_suffix("").with_suffix(
            ".embeddings.npz"
        )
        np.savez(
            embeddings_path,
            embeddings=np.stack([obs.embedding for obs in observations]),
            local_idx=np.array([obs.local_idx for obs in observations]),
            start=np.array([obs.start for obs in observations]),
            end=np.array([obs.end for obs in observations]),
            duration=np.array([obs.duration for obs in observations]),
        )
        logger.info(
            "[embeddings] saved %d embeddings to %s",
            len(observations),
            embeddings_path,
        )

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        streaming_log_path: str,
        uri: Optional[str] = None,
    ) -> None:
        """按 chunk 顺序处理整段波形并持续写出 RTTM。"""

        waveform = self.extractor.prepare_waveform(waveform, sample_rate)

        writer = AppendOnlyRTTMWriter(
            streaming_log_path,
            uri or "unknown",
            self.config.min_segment_duration,
            self.config.streaming_merge_gap,
            self.config.show_rttm,
        )

        # save_embeddings 开启时收集全部带 embedding 的 observation。
        collected_observations: list[ChunkObservation] = []

        def chunk_log_hook(
            chunk: ChunkArtifacts,
            local_to_global: Optional[dict[int, int]],
            debug_info: ChunkDebugInfo,
            emitted_frames: int,
        ) -> None:
            if self.config.save_embeddings:
                collected_observations.extend(
                    obs for obs in chunk.observations if obs.embedding is not None
                )
            self._log_structured(
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

        run_clustering(
            self.extractor.iter_chunk_artifacts(waveform),
            self.assigner,
            writer,
            chunk_hook=chunk_log_hook,
        )

        if self.config.save_embeddings:
            self._save_embeddings(collected_observations, streaming_log_path)

    def process_file(self, wav_path: str) -> str:
        """处理单个音频文件并返回生成的 RTTM 路径。"""

        self.reset()
        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)

        if self.config.output_dir_for_streaming is None:
            raise ValueError(
                "output_dir_for_streaming must be set before processing audio"
            )
        streaming_log_path = str(
            Path(self.config.output_dir_for_streaming)
            / f"{Path(wav_path).stem}.{self.assigner.output_tag}.rttm"
        )

        self.process_waveform(
            waveform,
            self.config.sample_rate,
            streaming_log_path=streaming_log_path,
            uri=Path(wav_path).stem,
        )
        return streaming_log_path


__all__ = ["ChunkDiarizationPipeline"]
