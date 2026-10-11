"""diarization 执行器：extract + cluster + RTTM 的组合核心。

DiarizationRunner 持有提取侧模型（只加载一次，可连续处理多条音频），
每条音频独立构建 clusterer / writer（状态不跨文件）。

oracle 模式（--oracle）：用 ref RTTM 驱动的 OracleFrontend 替代
segmentation 前端，不加载 segmentation 模型，用于上限实验。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from common.speakerlab.utils.fileio import load_audio

from common.config import ChunkPipelineConfig
from common.contracts import ChunkArtifacts, ChunkDebugInfo, ChunkObservation
from common.models.embedding_infer import build_region_embedder
from common.utils import log_structured
from common.utils.audio import prepare_mono_waveform
from diarization.cluster.backends import build_clusterer
from diarization.cluster.post_process import RefinedRTTMWriter
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter
from diarization.cluster.runner import run_clustering
from diarization.extract.extractor import ChunkExtractor
from diarization.extract.oracle import oracle_chunk_artifacts
from diarization.frontend import ChunkTrackBuilder, OracleFrontend

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
    """输出 chunk 级调试信息。"""

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

    每条音频独立构建 clusterer / writer（状态不跨文件），仅提取侧持有的
    模型跨文件复用（正常模式：segmentation + embedding；oracle 模式：仅
    embedding）。

    Attributes:
        config: 管线配置。
        oracle: 是否 oracle 模式（ref RTTM 替代 segmentation 前端）。
        extractor: 正常模式的 chunk 提取器（oracle 模式为 None）。
    """

    def __init__(
        self,
        config: ChunkPipelineConfig,
        model_path: Optional[str] = None,
        oracle: bool = False,
    ):
        """加载模型并初始化执行器。

        Args:
            config: 管线配置。
            model_path: 显式模型路径覆盖。
            oracle: True 时用 OracleFrontend（无 segmentation 模型）+
                RegionEmbedder 组合替代 ChunkExtractor。
        """

        self.config = config
        self.oracle = oracle
        if oracle:
            self.extractor = None
            self.oracle_frontend = OracleFrontend(config)
            self.region_embedder = build_region_embedder(config, model_path)
            self.track_builder = ChunkTrackBuilder(config)
        else:
            self.extractor = ChunkExtractor(config, model_path)

    @staticmethod
    def _resolve_ref_rttm(ref: Optional[str], audio_path: str) -> str:
        """解析当前音频对应的参考 RTTM 路径。

        Args:
            ref: --ref 传入的路径（RTTM 文件或含 {uri}.rttm 的目录）。
            audio_path: 当前音频路径（uri 取其文件名主干）。

        Returns:
            参考 RTTM 路径。

        Raises:
            ValueError: oracle 模式未提供 --ref，或按目录解析的 RTTM 不存在。
        """

        if ref is None:
            raise ValueError("oracle 模式需要 --ref 指定参考 RTTM（文件或目录）")
        ref_path = Path(ref)
        if ref_path.is_dir():
            candidate = ref_path / f"{Path(audio_path).stem}.rttm"
            if not candidate.is_file():
                raise ValueError(f"参考 RTTM 不存在: {candidate}")
            return str(candidate)
        return str(ref_path)

    def process_file(
        self, audio_path: str, exp_dir: Path, ref: Optional[str] = None
    ) -> str:
        """处理单条音频，输出 RTTM。

        Args:
            audio_path: 音频文件路径；uri 取其文件名主干。
            exp_dir: 本次 run 的输出目录。
            ref: oracle 模式的参考 RTTM（文件或目录）；正常模式忽略。

        Returns:
            raw RTTM 路径。
        """

        config = self.config
        config.output_dir = str(exp_dir)

        clusterer = build_clusterer(config)

        waveform = load_audio(audio_path, obj_fs=config.sample_rate)
        if self.oracle:
            ref_rttm = self._resolve_ref_rttm(ref, audio_path)
            waveform = prepare_mono_waveform(
                waveform, config.sample_rate, config.sample_rate
            )
            artifacts = oracle_chunk_artifacts(
                self.oracle_frontend.iter_prepared_chunks(waveform, ref_rttm),
                waveform,
                self.region_embedder,
                self.track_builder,
            )
        else:
            waveform = self.extractor.prepare_waveform(waveform, config.sample_rate)
            artifacts = self.extractor.iter_chunk_artifacts(waveform)

        uri = Path(audio_path).stem
        raw_rttm_path = str(exp_dir / "rttm" / f"{uri}.{clusterer.output_tag}.rttm")
        writer = AppendOnlyRTTMWriter(
            raw_rttm_path,
            uri,
            config.min_segment_duration,
            config.segment_merge_gap,
            config.show_rttm,
        )

        def chunk_log_hook(
            chunk: ChunkArtifacts,
            local_to_global: Optional[dict[int, int]],
            debug_info: ChunkDebugInfo,
            emitted_frames: int,
        ) -> None:
            """每 chunk 分配完成后的回调：日志输出。

            Args:
                chunk: 本 chunk 的产物。
                local_to_global: 局部到全局的说话人映射；deferred 后端下为
                    None（重放阶段不再回调）。
                debug_info: chunk 级调试信息。
                emitted_frames: 本 chunk 已提交帧数；deferred 后端下为 0。
            """

            # 输出本身由 runner/writer 完成。deferred 后端下 local_to_global
            # 为 None、emitted_frames 为 0（重放阶段不再回调）。
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
        run_clustering(
            artifacts,
            clusterer,
            writer,
            chunk_hook=chunk_log_hook,
            refiner=refiner,
        )
        return raw_rttm_path


def run_diarization(
    config: ChunkPipelineConfig,
    audio_path: str,
    exp_dir: Path,
    model_path: Optional[str] = None,
) -> str:
    """单条音频的 diarization 管线：extract + cluster + RTTM 输出。

    供 diarization.app（独立 CLI）复用；批量场景请直接用
    DiarizationRunner 以复用已加载的模型。

    Args:
        config: 管线配置。
        audio_path: 音频文件路径。
        exp_dir: 本次 run 的输出目录。
        model_path: 显式模型路径覆盖。

    Returns:
        raw RTTM 路径。
    """

    return DiarizationRunner(config, model_path).process_file(audio_path, exp_dir)


__all__ = ["DiarizationRunner", "run_diarization"]
