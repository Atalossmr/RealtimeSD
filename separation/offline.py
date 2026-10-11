"""离线 RTTM 驱动的分离切分：整段音频 + RTTM -> 各 speaker 的纯净音频段。

处理流程：

1. 解析 RTTM（rttm.load_rttm），speaker label 按字典序映射为 int
   speaker id（0 起，稳定可复现），全时间线按 separation_frame_step
   生成帧级活跃记录（rttm.build_frame_activities）；
2. 参照 embedding：每 speaker 取全时间线上仅其一人活跃的独唱帧区间，
   短于 min_segment_duration_for_embedding 的丢弃、长于
   max_segment_duration_for_embedding 的从起点截断，拼接后提 embedding
   作为归属匹配参照；无独唱区间的 speaker 记 warning（匹配缺参照会
   自然回退原始音频）；
3. 重叠帧（>=2 人活跃）按相邻间隔 <= segment_merge_gap 归并为分离窗口，
   窗口边界向两侧扩展到相邻窗口的中点（首窗到 0、末窗到音频末尾），
   保证 TIGER 输入有上下文且窗口间无缝覆盖全时间线；
4. 逐窗口处理：solo 帧直接用原始波形切片；重叠帧经 TIGER 分离 ->
   能量门控 -> 2x2 / 单路归属匹配，任一步失败整窗回退原始音频
   （宁可串音也不要坏轨/空轨）；
5. 各窗口帧片按时间顺序、按 hop_duration 跨度分批喂 SegmentAssembler
   拼接闭合（分批保证 max_segment_duration 的强制闭合检查生效，见
   `_feed_pieces`），默认 sink 落盘 wav + manifest（sink.py）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from common.config import ChunkPipelineConfig
from common.models import TigerSeparator
from common.utils import log_structured

from .assembler import SegmentAssembler, SpeakerPiece
from .attribution import (
    FrameActivity,
    best_match,
    energy_gate,
    match_pair,
    overlap_regions,
)
from .rttm import build_frame_activities, load_rttm
from .sink import WavSegmentSink
from .utils import AudioSource, slice_absolute

if TYPE_CHECKING:
    from common.models.embedding_infer import NativeERes2NetV2SegmentEmbedder


logger = logging.getLogger(__name__)


def _solo_regions(
    frames: list[FrameActivity], speaker_id: int
) -> list[tuple[float, float]]:
    """该 speaker 的独唱区间：仅其一人活跃的连续帧归并。

    Args:
        frames: 全时间线逐帧活跃记录（按时间有序）。
        speaker_id: 目标 speaker id。

    Returns:
        归并后的 (start, end) 独唱区间列表。
    """

    regions: list[tuple[float, float]] = []
    for frame_start, frame_end, active in frames:
        if active == [speaker_id]:
            if regions and frame_start <= regions[-1][1] + 1e-6:
                regions[-1] = (regions[-1][0], frame_end)
            else:
                regions.append((frame_start, frame_end))
    return regions


def _reference_embeddings(
    config: ChunkPipelineConfig,
    waveform: torch.Tensor,
    frames: list[FrameActivity],
    speaker_ids: list[int],
    embedder: NativeERes2NetV2SegmentEmbedder,
    sample_rate: int,
) -> dict[int, np.ndarray]:
    """各 speaker 的参照 embedding：独唱帧区间拼接后提 embedding。

    独唱区间短于 min_segment_duration_for_embedding 的丢弃、长于
    max_segment_duration_for_embedding 的从起点截断（同 track_builder /
    oracle_segments 的截断方向）。

    Args:
        config: 管线配置。
        waveform: 整段波形 [1, T]。
        frames: 全时间线逐帧活跃记录。
        speaker_ids: 全部 speaker id。
        embedder: 段级 speaker embedding 提取器。
        sample_rate: 采样率。

    Returns:
        speaker id -> 参照 embedding；无独唱区间的 speaker 记 warning 并缺席。
    """

    refs: dict[int, np.ndarray] = {}
    for speaker_id in speaker_ids:
        pieces: list[torch.Tensor] = []
        for start, end in _solo_regions(frames, speaker_id):
            if end - start < config.min_segment_duration_for_embedding:
                continue
            embed_end = min(end, start + config.max_segment_duration_for_embedding)
            piece = slice_absolute(waveform, 0.0, start, embed_end, sample_rate)
            if piece is not None:
                pieces.append(piece)
        if not pieces:
            logger.warning(
                "[separation] speaker %d 无独唱区间，缺少参照 embedding"
                "（归属匹配将回退原始音频）",
                speaker_id,
            )
            continue
        refs[speaker_id] = embedder.embed_segment(torch.cat(pieces, dim=1))
    return refs


def _build_windows(
    frames: list[FrameActivity], merge_gap: float, t_end: float
) -> list[tuple[float, float, list[FrameActivity]]]:
    """重叠帧归并为分离窗口，边界扩展到相邻窗口中点以无缝覆盖全时间线。

    相邻重叠帧间隔 <= merge_gap 归并为同一窗口；窗口时间边界向两侧扩展到
    相邻窗口的中点（首窗到 0、末窗到 t_end），两窗口之间的 solo 帧按
    frame_start 归属中点所在侧的窗口。

    Args:
        frames: 全时间线逐帧活跃记录（按时间有序）。
        merge_gap: 重叠帧归并的最大相邻间隔（秒）。
        t_end: 时间线终点（秒，音频时长）。

    Returns:
        [(window_start, window_end, window_frames)]，按时间有序；无重叠帧时
        返回覆盖 [0, t_end) 的单窗口（纯切片路径，无需加载 TIGER）。
    """

    if not frames:
        return []
    overlap_ids = [
        frame_idx for frame_idx, (_, _, active) in enumerate(frames) if len(active) >= 2
    ]
    if not overlap_ids:
        return [(0.0, t_end, frames)]

    groups: list[list[int]] = [[overlap_ids[0]]]
    for frame_idx in overlap_ids[1:]:
        prev_idx = groups[-1][-1]
        if frames[frame_idx][0] - frames[prev_idx][1] <= merge_gap:
            groups[-1].append(frame_idx)
        else:
            groups.append([frame_idx])

    spans = [(frames[group[0]][0], frames[group[-1]][1]) for group in groups]
    boundaries: list[tuple[float, float]] = []
    for idx, (start, end) in enumerate(spans):
        win_start = 0.0 if idx == 0 else (spans[idx - 1][1] + start) / 2.0
        win_end = t_end if idx == len(spans) - 1 else (end + spans[idx + 1][0]) / 2.0
        boundaries.append((win_start, win_end))

    windows: list[tuple[float, float, list[FrameActivity]]] = []
    frame_cursor = 0
    for win_start, win_end in boundaries:
        first = frame_cursor
        while frame_cursor < len(frames) and frames[frame_cursor][0] < win_end:
            frame_cursor += 1
        windows.append((win_start, win_end, frames[first:frame_cursor]))
    return windows


def _collect_pieces(
    frames: list[FrameActivity],
    overlap_frame_ids: set[int],
    track_of: dict[int, AudioSource],
    drop_overlap_for: set[int],
    mix_source: AudioSource,
) -> dict[int, list[SpeakerPiece]]:
    """按 speaker 归集帧片来源。

    Args:
        frames: 窗口内逐帧的活跃记录。
        overlap_frame_ids: 重叠帧在 frames 中的下标集合。
        track_of: speaker_id -> 分离音轨来源（仅匹配成功的重叠 pair）。
        drop_overlap_for: 重叠帧直接丢弃的 speaker 集合（门控失败的落选方）。
        mix_source: 原始混合音频来源（(整段波形, 0.0)）。

    Returns:
        speaker_id -> 帧片列表。重叠帧上无音轨且不在丢弃集合的 speaker 用
        原始音频（串音兜底），其余帧一律用原始波形切片。
    """

    pieces: dict[int, list[SpeakerPiece]] = {}
    for frame_idx, (frame_start, frame_end, active) in enumerate(frames):
        is_overlap = frame_idx in overlap_frame_ids
        for speaker_id in active:
            if is_overlap and speaker_id in drop_overlap_for:
                continue
            source = (
                track_of[speaker_id]
                if is_overlap and speaker_id in track_of
                else mix_source
            )
            pieces.setdefault(speaker_id, []).append((frame_start, frame_end, source))
    return pieces


def _feed_pieces(
    assembler: SegmentAssembler,
    pieces: dict[int, list[SpeakerPiece]],
    batch_duration: float,
) -> None:
    """按时间有界的批次喂入帧片，保证 max_segment_duration 强制闭合生效。

    SegmentAssembler 的段长上限检查发生在 run 续接到已有 open segment 时；
    单次喂入跨度过长的帧片（如无重叠音频只有一个大窗口）会整段成为一个
    run 直接开新段、绕过上限检查。按 batch_duration 网格把帧片拆成小批
    顺序喂入，跨批次的续接即可触发上限闭合。

    Args:
        assembler: 段组装器。
        pieces: speaker_id -> 帧片列表（按时间有序）。
        batch_duration: 喂入批次的时间跨度（秒），对齐到绝对时刻网格。
    """

    batches: dict[int, dict[int, list[SpeakerPiece]]] = {}
    for speaker_id, speaker_pieces in pieces.items():
        for piece in speaker_pieces:
            batch_idx = int(piece[0] // batch_duration)
            batches.setdefault(batch_idx, {}).setdefault(speaker_id, []).append(piece)
    for batch_idx in sorted(batches):
        assembler.feed_pieces(batches[batch_idx])


def separate_by_rttm(
    config: ChunkPipelineConfig,
    waveform: torch.Tensor,
    rttm_path: str | Path,
    embedder: NativeERes2NetV2SegmentEmbedder,
    uri: str,
    output_dir: str,
) -> str:
    """按 RTTM 把整段含混叠音频拆成各 speaker 的纯净音频段。

    处理流程见模块 docstring；任一步分离/匹配失败整窗回退原始音频
    （宁可串音也不要坏轨/空轨）。

    Args:
        config: 管线配置。
        waveform: 整段波形 [1, T]，已重采样到 config.sample_rate 单声道。
        rttm_path: 驱动切分的 RTTM 路径（如 refined RTTM）。
        embedder: 段级 speaker embedding 提取器（参照与归属匹配用）。
        uri: 音频标识（决定 wav 子目录与 manifest 文件名）。
        output_dir: segments 输出根目录。

    Returns:
        manifest.jsonl 路径（{output_dir}/{uri}.segments.jsonl）。
    """

    sample_rate = int(config.sample_rate)
    turns_by_label = load_rttm(rttm_path)
    # speaker label 按字典序映射为 int id（0 起），与 RTTM 行序无关、稳定可复现。
    labels = sorted(turns_by_label)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    turns_by_id = {label_to_id[label]: turns_by_label[label] for label in labels}
    logger.info(
        "[separation] uri=%s speakers=%s",
        uri,
        {label_to_id[label]: label for label in labels},
    )

    t_end = waveform.shape[1] / sample_rate
    frames = build_frame_activities(
        turns_by_id, float(config.separation_frame_step), t_end
    )
    refs = _reference_embeddings(
        config, waveform, frames, sorted(turns_by_id), embedder, sample_rate
    )
    windows = _build_windows(frames, float(config.segment_merge_gap), t_end)

    sink = WavSegmentSink(output_dir, uri, sample_rate)
    assembler = SegmentAssembler(config, sample_rate, uri, sink)
    mix_source: AudioSource = (waveform, 0.0)
    # 喂入批次跨度取 hop_duration（复用管线滑窗步长作批次网格）。
    batch_duration = float(config.hop_duration)
    # TIGER 惰性构建：仅存在含重叠帧的窗口时才需要（无重叠音频不加载模型；
    # TigerSeparator 内部也是首次 separate 才加载权重）。
    separator: Optional[TigerSeparator] = None

    for window_index, (win_start, win_end, window_frames) in enumerate(windows):
        if not window_frames:
            continue
        overlap_frame_ids = {
            frame_idx
            for frame_idx, (_, _, active) in enumerate(window_frames)
            if len(active) >= 2
        }
        if not overlap_frame_ids:
            _feed_pieces(
                assembler,
                _collect_pieces(window_frames, set(), {}, set(), mix_source),
                batch_duration,
            )
            continue

        if separator is None:
            separator = TigerSeparator(
                config.separation_model,
                config.hf_cache_dir,
                sample_rate,
                embedder.device,
            )
        pieces = _dispatch_window(
            config,
            separator,
            embedder,
            refs,
            window_index,
            win_start,
            win_end,
            window_frames,
            overlap_frame_ids,
            mix_source,
            sample_rate,
        )
        _feed_pieces(assembler, pieces, batch_duration)

    assembler.finalize()
    logger.info(
        "[separation] uri=%s windows=%d manifest=%s",
        uri,
        len(windows),
        sink.manifest_path,
    )
    return str(sink.manifest_path)


def _dispatch_window(
    config: ChunkPipelineConfig,
    separator: TigerSeparator,
    embedder: NativeERes2NetV2SegmentEmbedder,
    refs: dict[int, np.ndarray],
    window_index: int,
    win_start: float,
    win_end: float,
    window_frames: list[FrameActivity],
    overlap_frame_ids: set[int],
    mix_source: AudioSource,
    sample_rate: int,
) -> dict[int, list[SpeakerPiece]]:
    """单个分离窗口的 TIGER 分离、能量门控与归属匹配（失败整窗回退原始音频）。

    Args:
        config: 管线配置。
        separator: TIGER 分离器。
        embedder: 段级 speaker embedding 提取器。
        refs: speaker id -> 参照 embedding。
        window_index: 本窗口序号（仅用于日志）。
        win_start: 窗口起始绝对时刻（秒）。
        win_end: 窗口结束绝对时刻（秒）。
        window_frames: 窗口内逐帧活跃记录。
        overlap_frame_ids: 重叠帧在 window_frames 中的下标集合（非空）。
        mix_source: 原始混合音频来源（(整段波形, 0.0)）。
        sample_rate: 采样率。

    Returns:
        speaker_id -> 帧片列表。
    """

    def fallback_all_mix() -> dict[int, list[SpeakerPiece]]:
        """整窗回退：重叠帧不做分离，所有 speaker 一律用原始音频切片。"""
        return _collect_pieces(window_frames, overlap_frame_ids, {}, set(), mix_source)

    overlap_frames = [window_frames[i] for i in sorted(overlap_frame_ids)]
    mix = slice_absolute(mix_source[0], 0.0, win_start, win_end, sample_rate)
    separated = separator.separate(mix) if mix is not None else None
    if separated is None:
        log_structured(
            logger,
            logging.WARNING,
            "[separation]",
            "separate_too_short",
            {
                "window_index": int(window_index),
                "window": [round(win_start, 3), round(win_end, 3)],
            },
        )
        return fallback_all_mix()
    mix, tracks = separated
    track_sources: list[AudioSource] = [
        (tracks[idx : idx + 1], win_start) for idx in range(tracks.shape[0])
    ]

    # ---- 能量门控：重叠帧区间上 音轨RMS / 混合RMS ----
    regions = overlap_regions(overlap_frames)
    passed = energy_gate(
        mix,
        track_sources,
        regions,
        sample_rate,
        config.separation_energy_ratio,
        mix_origin=win_start,
        window=(win_start, win_end),
        window_index=window_index,
    )

    # ---- 候选 speaker：重叠帧上活跃次数最多的前 2 个 ----
    counts: dict[int, int] = {}
    for _, _, active in overlap_frames:
        for speaker_id in active:
            counts[speaker_id] = counts.get(speaker_id, 0) + 1
    candidates = sorted(counts, key=lambda g: (-counts[g], g))[:2]
    if len(candidates) < 2:
        # 重叠帧上活跃的是同一个 speaker（或无候选）：等同无重叠。
        return fallback_all_mix()

    if len(passed) == 2:
        embeddings = embedder.embed_segments([track_sources[idx][0] for idx in passed])
        mapping = match_pair(
            window_index,
            embeddings,
            candidates,
            refs,
            config.separation_min_match_similarity,
        )
        if mapping is None:
            return fallback_all_mix()
        track_of = {
            speaker_id: track_sources[track_idx]
            for track_idx, speaker_id in mapping.items()
        }
        return _collect_pieces(
            window_frames, overlap_frame_ids, track_of, set(), mix_source
        )
    if len(passed) == 1:
        # 一路不过门控：疑似重叠误报，过门控音轨匹配归属，重叠帧回退原始音频。
        winner = passed[0]
        embedding = embedder.embed_segment(track_sources[winner][0])
        best = best_match(embedding, candidates, refs)
        if best is None:
            return fallback_all_mix()
        winner_speaker, winner_sim = best
        log_structured(
            logger,
            logging.INFO,
            "[separation]",
            "gate_fallback",
            {
                "window_index": int(window_index),
                "winner_track": int(winner),
                "assigned_speaker": int(winner_speaker),
                "similarity": round(winner_sim, 4),
                "candidates": [int(g) for g in candidates],
            },
        )
        loser = {g for g in candidates if g != winner_speaker}
        return _collect_pieces(window_frames, overlap_frame_ids, {}, loser, mix_source)
    # 两路都不过：混合有能量但两路皆空属异常情况，记 log 后整窗回退。
    log_structured(
        logger,
        logging.WARNING,
        "[separation]",
        "both_tracks_failed",
        {
            "window_index": int(window_index),
            "window": [round(win_start, 3), round(win_end, 3)],
        },
    )
    return fallback_all_mix()


__all__ = ["separate_by_rttm"]
