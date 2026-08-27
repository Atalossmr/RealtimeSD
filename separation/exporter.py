"""流式分段音频导出器：commit 区重叠检测 -> TIGER 分离 -> 归属匹配 -> 段输出。

挂载在 chunk 管线的 chunk_hook 上（仅 streaming 后端），每个 commit 区：

1. 用 seg_scores 硬标签逐帧统计活跃 global speaker，>=2 人活跃即重叠帧；
2. 无重叠帧：各 speaker 的活跃帧直接从原始波形切片，拼成音频段输出；
3. 有重叠帧：TIGER 分离整个 commit 区（固定 2 路，见 common/models/tiger.py），在
   重叠帧区间做能量门控（音轨 RMS / 混合 RMS）：
   - 两路都过：2x2 embedding 互斥匹配（参照为本 chunk 观测 embedding），
     重叠帧用各自分到的音轨、独占帧用原始音频；
   - 一路不过：判为（可能的 OSD 误报）分离失败，记 log；过门控那路的
     embedding 匹配归属 speaker，该 speaker 的重叠帧回退用原始音频，
     包装为一个音频段；另一候选 speaker 的重叠帧不输出；
   - 两路都不过：记 log，整窗回退为纯切片路径。
4. 匹配结果的最小相似度低于 separation_min_match_similarity 时判分离质量
   不可靠，同样整窗回退为纯切片路径（宁可串音也不要坏轨/空轨进 ASR）。

段的跨 commit 区拼接与闭合推出委托 SegmentAssembler（见 assembler.py，
与 RTTM writer 的 open-turn 同构）；输出经 SegmentCallback 推出（接流式
ASR），默认 sink 落盘 wav + manifest（见 sink.py）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from common.frames import iter_commit_frames
from common.log import log_structured
from common.models import TigerSeparator
from common.pipeline_config import ChunkPipelineConfig
from common.schema import ChunkArtifacts

from .assembler import SegmentAssembler, SpeakerPiece
from .audio import AudioSource, rms_over_regions, slice_absolute
from .sink import SegmentCallback, WavSegmentSink

if TYPE_CHECKING:
    from common.models.embedding_infer import NativeERes2NetV2SegmentEmbedder


logger = logging.getLogger(__name__)

# 帧级活跃记录：(frame_start, frame_end, active_global_ids)。
FrameActivity = tuple[float, float, list[int]]


class StreamingSegmentExporter:
    """逐 commit 区导出各 speaker 的音频段（重叠区经 TIGER 分离）。"""

    def __init__(
        self,
        config: ChunkPipelineConfig,
        waveform: torch.Tensor,
        embedder: NativeERes2NetV2SegmentEmbedder,
        uri: str,
        output_dir: str,
        on_segment: Optional[SegmentCallback] = None,
    ):
        self.config = config
        self.sample_rate = int(config.sample_rate)
        # 整段波形 [1, T]，已重采样到 sample_rate 单声道（pipeline 侧 prepare 过）。
        self.waveform = waveform
        self.embedder = embedder
        self.uri = uri
        on_segment = on_segment or WavSegmentSink(output_dir, uri, self.sample_rate)
        self.assembler = SegmentAssembler(config, self.sample_rate, uri, on_segment)
        self.separator = TigerSeparator(
            config.separation_model,
            config.hf_cache_dir,
            self.sample_rate,
            embedder.device,
        )

    # ------------------------------------------------------------------
    # 帧级活跃与区间工具
    # ------------------------------------------------------------------

    def _commit_frames(
        self, chunk: ChunkArtifacts, local_to_global: dict[int, int]
    ) -> list[FrameActivity]:
        """commit 区内逐帧的活跃 global speaker（与 writer 共用帧裁剪逻辑）。"""

        return list(
            iter_commit_frames(
                chunk.seg_scores,
                chunk.frame_step,
                chunk.chunk_start,
                chunk.commit_start,
                chunk.commit_end,
                local_to_global,
            )
        )

    @staticmethod
    def _overlap_regions(
        overlap_frames: list[FrameActivity],
    ) -> list[tuple[float, float]]:
        """把重叠帧合成连续区间（时间上相邻的帧归并为一个区间）。"""

        regions: list[tuple[float, float]] = []
        for frame_start, frame_end, _ in overlap_frames:
            if regions and frame_start <= regions[-1][1] + 1e-6:
                regions[-1] = (regions[-1][0], frame_end)
            else:
                regions.append((frame_start, frame_end))
        return regions

    # ------------------------------------------------------------------
    # 归属匹配
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(a @ b)

    def _reference_embeddings(
        self, chunk: ChunkArtifacts, local_to_global: dict[int, int]
    ) -> dict[int, np.ndarray]:
        """各 global speaker 的参照 embedding：候选 speaker 用本 chunk 的观测。

        按 clusterer 契约，local_to_global 中的 slot 必有带 embedding 的
        observation（提不出 embedding 的 slot 不会进入分配），因此候选
        一定能被 observation 覆盖。
        """

        refs: dict[int, np.ndarray] = {}
        for obs in chunk.observations:
            global_id = local_to_global.get(obs.local_idx)
            if global_id is not None and obs.embedding is not None:
                refs[int(global_id)] = obs.embedding
        return refs

    def _match_pair(
        self,
        chunk_index: int,
        track_embeddings: list[np.ndarray],
        candidates: list[int],
        refs: dict[int, np.ndarray],
    ) -> Optional[dict[int, int]]:
        """2x2 互斥匹配：返回 track_idx -> global_id；总相似度过低返回 None。"""

        g0, g1 = candidates
        if g0 not in refs or g1 not in refs:
            return None
        s00 = self._cosine(track_embeddings[0], refs[g0])
        s01 = self._cosine(track_embeddings[0], refs[g1])
        s10 = self._cosine(track_embeddings[1], refs[g0])
        s11 = self._cosine(track_embeddings[1], refs[g1])
        if s00 + s11 >= s01 + s10:
            mapping, pair_sims = {0: g0, 1: g1}, (s00, s11)
        else:
            mapping, pair_sims = {0: g1, 1: g0}, (s01, s10)
        accepted = min(pair_sims) >= self.config.separation_min_match_similarity
        log_structured(
            logger,
            logging.INFO,
            "[separation]",
            "pair_match",
            {
                "chunk_index": int(chunk_index),
                "candidates": [int(g0), int(g1)],
                "sims": {
                    "track0": [round(s00, 4), round(s01, 4)],
                    "track1": [round(s10, 4), round(s11, 4)],
                },
                "mapping": {"track0": int(mapping[0]), "track1": int(mapping[1])},
                "min_sim": round(min(pair_sims), 4),
                "threshold": float(self.config.separation_min_match_similarity),
                "accepted": bool(accepted),
            },
        )
        if not accepted:
            return None
        return mapping

    # ------------------------------------------------------------------
    # 帧片归集（段组装见 assembler.SegmentAssembler）
    # ------------------------------------------------------------------

    def _collect_pieces(
        self,
        frames: list[FrameActivity],
        overlap_frame_ids: set[int],
        track_of: dict[int, AudioSource],
        drop_overlap_for: set[int],
    ) -> dict[int, list[SpeakerPiece]]:
        """按 speaker 归集帧片来源。

        track_of: global_id -> 分离音轨来源（仅匹配成功的重叠 pair）；
        重叠帧上无音轨的 speaker 用原始音频（串音兜底），
        drop_overlap_for 中的 speaker 重叠帧直接丢弃（门控失败的落选方）。
        """

        pieces: dict[int, list[SpeakerPiece]] = {}
        mix_source: AudioSource = (self.waveform, 0.0)
        for frame_idx, (frame_start, frame_end, active) in enumerate(frames):
            is_overlap = frame_idx in overlap_frame_ids
            for global_id in active:
                if is_overlap and global_id in drop_overlap_for:
                    continue
                source = (
                    track_of[global_id]
                    if is_overlap and global_id in track_of
                    else mix_source
                )
                pieces.setdefault(global_id, []).append((frame_start, frame_end, source))
        return pieces

    def _fallback_all_mix(
        self, frames: list[FrameActivity], overlap_frame_ids: set[int]
    ) -> None:
        """整窗回退：重叠帧不做分离，所有 speaker 一律用原始音频切片。"""

        self.assembler.feed_pieces(
            self._collect_pieces(frames, overlap_frame_ids, {}, set())
        )

    # ------------------------------------------------------------------
    # 主入口（chunk_hook 调用）
    # ------------------------------------------------------------------

    def handle_chunk(
        self, chunk: ChunkArtifacts, local_to_global: dict[int, int]
    ) -> None:
        """处理一个 chunk 的 commit 区：检测重叠、按需分离、逐 speaker 归集音频。

        音频经 open-segment 管线跨 commit 区拼接；每次归集后闭合已确认沉默
        的段（与 writer.close_inactive 同步 lowering 输出延迟）。
        """

        if not local_to_global:
            return
        frames = self._commit_frames(chunk, local_to_global)
        if not frames:
            return
        self._dispatch_chunk(chunk, local_to_global, frames)
        self.assembler.close_inactive(chunk.commit_end)

    def finalize(self) -> None:
        """音频结束：闭合所有残余 open segment。"""

        self.assembler.finalize()

    def _dispatch_chunk(
        self,
        chunk: ChunkArtifacts,
        local_to_global: dict[int, int],
        frames: list[FrameActivity],
    ) -> None:
        """单个 commit 区的重叠检测、分离与帧片归集（不含段闭合）。"""

        overlap_frame_ids = {
            frame_idx
            for frame_idx, (_, _, active) in enumerate(frames)
            if len(active) >= 2
        }
        if not overlap_frame_ids:
            self.assembler.feed_pieces(self._collect_pieces(frames, set(), {}, set()))
            return

        # ---- 有重叠：TIGER 分离整个 commit 区 ----
        overlap_frames = [frames[i] for i in sorted(overlap_frame_ids)]
        mix = slice_absolute(
            self.waveform, 0.0, chunk.commit_start, chunk.commit_end, self.sample_rate
        )
        separated = self.separator.separate(mix) if mix is not None else None
        if separated is None:
            log_structured(
                logger,
                logging.WARNING,
                "[separation]",
                "separate_too_short",
                {
                    "chunk_index": int(chunk.chunk_index),
                    "commit": [
                        round(float(chunk.commit_start), 3),
                        round(float(chunk.commit_end), 3),
                    ],
                },
            )
            self._fallback_all_mix(frames, overlap_frame_ids)
            return
        mix, tracks = separated
        track_sources: list[AudioSource] = [
            (tracks[idx : idx + 1], chunk.commit_start) for idx in range(tracks.shape[0])
        ]

        # ---- 能量门控：重叠帧区间上 音轨RMS / 混合RMS ----
        regions = self._overlap_regions(overlap_frames)
        mix_rms = rms_over_regions(mix, chunk.commit_start, regions, self.sample_rate)
        passed: list[int] = []
        gate_tracks: list[dict] = []
        for track_idx, (track, origin) in enumerate(track_sources):
            ratio = (
                rms_over_regions(track, origin, regions, self.sample_rate) / mix_rms
            )
            track_passed = ratio >= self.config.separation_energy_ratio
            if track_passed:
                passed.append(track_idx)
            gate_tracks.append(
                {
                    "track": int(track_idx),
                    "ratio": round(ratio, 4),
                    "passed": bool(track_passed),
                }
            )
        log_structured(
            logger,
            logging.INFO,
            "[separation]",
            "energy_gate",
            {
                "chunk_index": int(chunk.chunk_index),
                "commit": [
                    round(float(chunk.commit_start), 3),
                    round(float(chunk.commit_end), 3),
                ],
                "mix_rms": round(mix_rms, 6),
                "tracks": gate_tracks,
                "threshold": float(self.config.separation_energy_ratio),
            },
        )

        # ---- 候选 speaker：重叠帧上活跃次数最多的前 2 个 global ----
        counts: dict[int, int] = {}
        for _, _, active in overlap_frames:
            for global_id in active:
                counts[global_id] = counts.get(global_id, 0) + 1
        candidates = sorted(counts, key=lambda g: (-counts[g], g))[:2]

        refs = self._reference_embeddings(chunk, local_to_global)

        if len(candidates) < 2:
            # 重叠帧上的 slot 映射到同一个 global（或无候选）：等同无重叠。
            self._fallback_all_mix(frames, overlap_frame_ids)
            return

        if len(passed) == 2:
            embeddings = self.embedder.embed_segments(
                [track_sources[idx][0] for idx in passed]
            )
            mapping = self._match_pair(chunk.chunk_index, embeddings, candidates, refs)
            if mapping is None:
                self._fallback_all_mix(frames, overlap_frame_ids)
                return
            track_of = {
                global_id: track_sources[track_idx]
                for track_idx, global_id in mapping.items()
            }
            pieces = self._collect_pieces(frames, overlap_frame_ids, track_of, set())
        elif len(passed) == 1:
            # 一路不过门控：疑似 OSD 误报，过门控音轨匹配归属，重叠帧回退原始音频。
            winner = passed[0]
            embedding = self.embedder.embed_segment(track_sources[winner][0])
            sims = {
                g: self._cosine(embedding, refs[g]) for g in candidates if g in refs
            }
            if not sims:
                self._fallback_all_mix(frames, overlap_frame_ids)
                return
            winner_global = max(sims, key=sims.get)
            log_structured(
                logger,
                logging.INFO,
                "[separation]",
                "gate_fallback",
                {
                    "chunk_index": int(chunk.chunk_index),
                    "winner_track": int(winner),
                    "assigned_global": int(winner_global),
                    "similarity": round(sims[winner_global], 4),
                    "candidates": [int(g) for g in candidates],
                },
            )
            loser = {g for g in candidates if g != winner_global}
            pieces = self._collect_pieces(frames, overlap_frame_ids, {}, loser)
        else:
            # 两路都不过：混合有能量但两路皆空属异常情况，记 log 后整窗回退。
            log_structured(
                logger,
                logging.WARNING,
                "[separation]",
                "both_tracks_failed",
                {
                    "chunk_index": int(chunk.chunk_index),
                    "commit": [
                        round(float(chunk.commit_start), 3),
                        round(float(chunk.commit_end), 3),
                    ],
                },
            )
            self._fallback_all_mix(frames, overlap_frame_ids)
            return

        self.assembler.feed_pieces(pieces)


__all__ = ["StreamingSegmentExporter", "FrameActivity"]
