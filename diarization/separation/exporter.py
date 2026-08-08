"""流式分段音频导出器：commit 区重叠检测 -> TIGER 分离 -> 归属匹配 -> 段输出。

挂载在 chunk 管线的 chunk_hook 上（仅 streaming 后端），每个 commit 区：

1. 用 seg_scores 硬标签逐帧统计活跃 global speaker，>=2 人活跃即重叠帧；
2. 无重叠帧：各 speaker 的活跃帧直接从原始波形切片，拼成音频段输出；
3. 有重叠帧：TIGER 分离整个 commit 区（固定 2 路），在重叠帧区间做
   能量门控（音轨 RMS / 混合 RMS）：
   - 两路都过：2x2 embedding 互斥匹配（参照由 separation_match_reference
     决定：centroid 全局质心 / observation 本 chunk 观测），重叠帧用各自
     分到的音轨、独占帧用原始音频；
   - 一路不过：判为（可能的 OSD 误报）分离失败，记 log；过门控那路的
     embedding 匹配归属 speaker，该 speaker 的重叠帧回退用原始音频，
     包装为一个音频段；另一候选 speaker 的重叠帧不输出；
   - 两路都不过：记 log，整窗回退为纯切片路径。
4. 匹配结果的最小相似度低于 separation_min_match_similarity 时判分离质量
   不可靠，同样整窗回退为纯切片路径（宁可串音也不要坏轨/空轨进 ASR）。

输出的音频段经 SegmentCallback 推出（接流式 ASR），默认 sink 落盘 wav + manifest。
段组织方式与 RTTM writer 的 open-turn 同构：每个 speaker 维护一个 open
segment，帧片按 streaming_merge_gap 拆成连续 run 喂入，相邻 run 间隔
不超过 merge_gap 时续接，超过或 EOF 时闭合推出（min_segment_duration
在闭合时对总时长判定）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio

from ..schema import ChunkArtifacts
from ..utils import log_structured


logger = logging.getLogger(__name__)

# ASR 侧接收音频段的回调签名：(uri, speaker_id, start, end, waveform[1, T])。
SegmentCallback = Callable[[str, int, float, float, torch.Tensor], None]

# 帧级活跃记录：(frame_start, frame_end, active_global_ids)。
FrameActivity = tuple[float, float, list[int]]

# 音频来源：(波形 [1, T], 该波形零点对应的绝对时刻)。
AudioSource = tuple[torch.Tensor, float]

# 段内一片音频：(frame_start, frame_end, source)。
SpeakerPiece = tuple[float, float, AudioSource]


@dataclass
class _OpenSegment:
    """单个 speaker 未闭合的音频段（驻留内存，闭合前可跨 commit 区续接）。"""

    start: float
    end: float
    audios: list[torch.Tensor] = field(default_factory=list)


class WavSegmentSink:
    """默认段输出 sink：wav 落盘 + manifest.jsonl 追加。"""

    def __init__(self, output_dir: str, uri: str, sample_rate: int):
        self.sample_rate = sample_rate
        self.segments_dir = Path(output_dir) / "segments" / uri
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(output_dir) / f"{uri}.segments.jsonl"

    def __call__(
        self,
        uri: str,
        speaker_id: int,
        start: float,
        end: float,
        waveform: torch.Tensor,
    ) -> None:
        filename = f"spk{speaker_id}_{start:.3f}_{end:.3f}.wav"
        path = self.segments_dir / filename
        torchaudio.save(str(path), waveform.cpu(), self.sample_rate)
        with open(self.manifest_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(
                json.dumps(
                    {
                        "uri": uri,
                        "speaker_id": int(speaker_id),
                        "start": round(float(start), 3),
                        "end": round(float(end), 3),
                        "path": str(path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class StreamingSegmentExporter:
    """逐 commit 区导出各 speaker 的音频段（重叠区经 TIGER 分离）。"""

    def __init__(
        self,
        config,
        waveform: torch.Tensor,
        embedder,
        assigner,
        uri: str,
        output_dir: str,
        on_segment: Optional[SegmentCallback] = None,
    ):
        self.config = config
        self.sample_rate = int(config.sample_rate)
        # 整段波形 [1, T]，已重采样到 sample_rate 单声道（pipeline 侧 prepare 过）。
        self.waveform = waveform
        self.embedder = embedder
        self.assigner = assigner
        self.uri = uri
        self.on_segment: SegmentCallback = on_segment or WavSegmentSink(
            output_dir, uri, self.sample_rate
        )
        # TIGER 惰性加载：全程无重叠时不付出模型加载开销。
        self._tiger = None
        # 每个 speaker 未闭合的音频段（open-segment 管线，跨 commit 区拼接）。
        self._open_segments: dict[int, _OpenSegment] = {}

    # ------------------------------------------------------------------
    # TIGER 分离
    # ------------------------------------------------------------------

    def _load_tiger(self) -> None:
        import look2hear.models

        logger.info(
            "[separation] loading TIGER model %s", self.config.separation_model
        )
        model = look2hear.models.TIGER.from_pretrained(
            self.config.separation_model, cache_dir=self.config.hf_cache_dir
        )
        model.to(self.embedder.device)
        model.eval()
        self._tiger = model

    def _separate(
        self, commit_start: float, commit_end: float
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """分离 commit 区音频，返回 (混合波形, 分离音轨 [2, T])，长度已对齐。"""

        if self._tiger is None:
            self._load_tiger()
        mix = self._slice_absolute(self.waveform, 0.0, commit_start, commit_end)
        if mix is None or mix.shape[1] < self.sample_rate // 4:
            return None
        audio_input = mix.unsqueeze(0).to(self.embedder.device)  # [1, 1, T]
        with torch.inference_mode():
            ests = self._tiger(audio_input)  # [1, num_spk, T]
        tracks = ests.squeeze(0).cpu()
        if tracks.ndim == 1:
            tracks = tracks.unsqueeze(0)
        aligned = min(tracks.shape[-1], mix.shape[-1])
        return mix[:, :aligned], tracks[:, :aligned]

    # ------------------------------------------------------------------
    # 帧级活跃与切片工具
    # ------------------------------------------------------------------

    def _slice_absolute(
        self,
        source: torch.Tensor,
        origin: float,
        start: float,
        end: float,
    ) -> Optional[torch.Tensor]:
        """从 source（零点对应绝对时刻 origin）裁出 [start, end)，越界截断。"""

        total = source.shape[1]
        start_sample = max(
            0, min(int(round((start - origin) * self.sample_rate)), total)
        )
        end_sample = max(
            start_sample, min(int(round((end - origin) * self.sample_rate)), total)
        )
        if end_sample <= start_sample:
            return None
        return source[:, start_sample:end_sample]

    def _commit_frames(
        self, chunk: ChunkArtifacts, local_to_global: dict[int, int]
    ) -> list[FrameActivity]:
        """commit 区内逐帧的活跃 global speaker（与 writer 的帧裁剪逻辑一致）。"""

        frames: list[FrameActivity] = []
        seg_scores = chunk.seg_scores
        for frame_idx in range(seg_scores.shape[0]):
            frame_start = chunk.chunk_start + frame_idx * chunk.frame_step
            frame_end = frame_start + chunk.frame_step
            if frame_end <= chunk.commit_start + 1e-9:
                continue
            if frame_start >= chunk.commit_end - 1e-9:
                break
            frame_start = max(frame_start, chunk.commit_start)
            frame_end = min(frame_end, chunk.commit_end)
            frame_scores = seg_scores[frame_idx]
            active = sorted(
                {
                    int(local_to_global[local_idx])
                    for local_idx in range(len(frame_scores))
                    if frame_scores[local_idx] > 0.0
                    and local_idx in local_to_global
                }
            )
            frames.append((frame_start, frame_end, active))
        return frames

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
    # 能量门控与归属匹配
    # ------------------------------------------------------------------

    @staticmethod
    def _rms(waveform: torch.Tensor) -> float:
        if waveform.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.mean(waveform.float() ** 2))) + 1e-12

    def _rms_over_regions(
        self,
        source: torch.Tensor,
        origin: float,
        regions: list[tuple[float, float]],
    ) -> float:
        pieces = [
            piece
            for start, end in regions
            if (piece := self._slice_absolute(source, origin, start, end)) is not None
        ]
        if not pieces:
            return 0.0
        return self._rms(torch.cat(pieces, dim=1))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(a @ b)

    def _reference_embeddings(
        self, chunk: ChunkArtifacts, local_to_global: dict[int, int]
    ) -> dict[int, np.ndarray]:
        """各 global speaker 的参照 embedding。

        两种模式（separation_match_reference）：
        - observation（默认）：候选 speaker 用本 chunk 的观测 embedding。
          按 assigner 契约，local_to_global 中的 slot 必有带 embedding 的
          observation（提不出 embedding 的 slot 不会进入分配），因此候选
          一定能被 observation 覆盖；centroid 层仅是防御性填充，正常不会用到。
        - centroid：一律用全局质心，不读本 chunk 观测。
        """

        refs: dict[int, np.ndarray] = {}
        centroids = getattr(self.assigner, "centroids", {}) or {}
        for global_id, centroid in centroids.items():
            refs[int(global_id)] = np.asarray(centroid, dtype=np.float32)
        if self.config.separation_match_reference != "observation":
            return refs
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
    # 段组装与输出（open-segment 管线：与 RTTM writer 的 open-turn 同构）
    # ------------------------------------------------------------------

    def _feed_speaker_pieces(self, speaker_id: int, pieces: list[SpeakerPiece]) -> None:
        """把单个 speaker 一个 commit 区内的帧片送入 open-segment 管线。

        帧片按 merge_gap 拆成连续 run 再逐 run 喂入：帧间隔超过
        streaming_merge_gap 即断开（与 writer 逐帧喂入的合并判定等价，
        不会因窗级合并吞掉窗内较长的沉默）。音频只含活跃帧，run 内帧间
        的小于 merge_gap 的沉默不进入音频。
        """

        if not pieces:
            return
        runs: list[list[SpeakerPiece]] = [[pieces[0]]]
        for piece in pieces[1:]:
            if piece[0] - runs[-1][-1][1] <= self.config.streaming_merge_gap:
                runs[-1].append(piece)
            else:
                runs.append([piece])
        for run in runs:
            audio_pieces = [
                audio
                for start, end, (source, origin) in run
                if (audio := self._slice_absolute(source, origin, start, end))
                is not None
            ]
            if not audio_pieces:
                continue
            audio = torch.cat(audio_pieces, dim=1)
            self._feed_segment(speaker_id, run[0][0], run[-1][1], audio)

    def _feed_segment(
        self, speaker_id: int, start: float, end: float, audio: torch.Tensor
    ) -> None:
        """把一段音频送入 speaker 的 open segment（闭合前持续拼接，跨窗生效）。"""

        open_seg = self._open_segments.get(speaker_id)
        if open_seg is not None:
            forward_gap = start - open_seg.end
            # 间隔不超过 merge_gap 视为同一段，直接续接（与 writer 合并语义一致）。
            if forward_gap <= self.config.streaming_merge_gap:
                open_seg.end = max(open_seg.end, end)
                open_seg.audios.append(audio)
                return
        # 间隔过大：先闭合旧段再开新段。
        self._close_segment(speaker_id)
        self._open_segments[speaker_id] = _OpenSegment(
            start=float(start), end=float(end), audios=[audio]
        )

    def _close_segment(self, speaker_id: int) -> None:
        """闭合 speaker 的 open segment：拼接全部音频并推出。"""

        open_seg = self._open_segments.pop(speaker_id, None)
        if open_seg is None:
            return
        audio = torch.cat(open_seg.audios, dim=1)
        if audio.shape[1] / self.sample_rate < self.config.min_segment_duration:
            return
        self.on_segment(self.uri, speaker_id, open_seg.start, open_seg.end, audio)

    def _close_inactive(self, commit_end: float) -> None:
        """闭合已确认沉默的 open segment（与 writer.close_inactive 同语义）。

        提交区无缝拼接：一旦 commit_end - seg.end > merge_gap，该段未来不可能
        再被延长，提前闭合以降低输出延迟。
        """

        for speaker_id, open_seg in list(self._open_segments.items()):
            if commit_end - open_seg.end > self.config.streaming_merge_gap:
                self._close_segment(speaker_id)

    def finalize(self) -> None:
        """音频结束：闭合所有残余 open segment。"""

        for speaker_id in list(self._open_segments.keys()):
            self._close_segment(speaker_id)

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
        self._close_inactive(chunk.commit_end)

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
            pieces = self._collect_pieces(frames, set(), {}, set())
            for global_id, speaker_pieces in sorted(pieces.items()):
                self._feed_speaker_pieces(global_id, speaker_pieces)
            return

        # ---- 有重叠：TIGER 分离整个 commit 区 ----
        overlap_frames = [frames[i] for i in sorted(overlap_frame_ids)]
        separated = self._separate(chunk.commit_start, chunk.commit_end)
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
            pieces = self._collect_pieces(frames, overlap_frame_ids, {}, set())
            for global_id, speaker_pieces in sorted(pieces.items()):
                self._feed_speaker_pieces(global_id, speaker_pieces)
            return
        mix, tracks = separated
        track_sources: list[AudioSource] = [
            (tracks[idx : idx + 1], chunk.commit_start) for idx in range(tracks.shape[0])
        ]

        # ---- 能量门控：重叠帧区间上 音轨RMS / 混合RMS ----
        regions = self._overlap_regions(overlap_frames)
        mix_rms = self._rms_over_regions(mix, chunk.commit_start, regions)
        passed: list[int] = []
        gate_tracks: list[dict] = []
        for track_idx, (track, origin) in enumerate(track_sources):
            ratio = self._rms_over_regions(track, origin, regions) / mix_rms
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

        def fallback_all_mix() -> None:
            pieces = self._collect_pieces(frames, overlap_frame_ids, {}, set())
            for global_id, speaker_pieces in sorted(pieces.items()):
                self._feed_speaker_pieces(global_id, speaker_pieces)

        if len(candidates) < 2:
            # 重叠帧上的 slot 映射到同一个 global（或无候选）：等同无重叠。
            fallback_all_mix()
            return

        if len(passed) == 2:
            embeddings = self.embedder.embed_segments(
                [track_sources[idx][0] for idx in passed]
            )
            mapping = self._match_pair(chunk.chunk_index, embeddings, candidates, refs)
            if mapping is None:
                fallback_all_mix()
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
                fallback_all_mix()
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
            fallback_all_mix()
            return

        for global_id, speaker_pieces in sorted(pieces.items()):
            self._feed_speaker_pieces(global_id, speaker_pieces)


__all__ = ["StreamingSegmentExporter", "SegmentCallback", "WavSegmentSink"]
