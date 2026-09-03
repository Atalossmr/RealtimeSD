"""open-segment 组装管线：帧片 -> 续接/闭合 -> 段推出（与 RTTM writer 的 open-turn 同构）。

每个 speaker 维护一个未闭合的 open segment（驻留内存，闭合前可跨 commit 区
续接）：帧片按 segment_merge_gap 拆成连续 run 喂入，相邻 run 间隔不超过
merge_gap 时续接，超过或 EOF 时闭合推出（min_segment_duration 在闭合时对
总时长判定）；段时长达到 max_segment_duration 时强制闭合并另开新段续接，
为内存驻留与 ASR 输出延迟提供上界。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from common.config import ChunkPipelineConfig

from .utils import AudioSource, slice_absolute
from .sink import SegmentCallback

# 段内一片音频：(frame_start, frame_end, source)。
SpeakerPiece = tuple[float, float, AudioSource]


@dataclass
class _OpenSegment:
    """单个 speaker 未闭合的音频段（驻留内存，闭合前可跨 commit 区续接）。"""

    start: float
    end: float
    audios: list[torch.Tensor] = field(default_factory=list)


class SegmentAssembler:
    """各 speaker 音频段的跨 commit 区拼接与闭合推出。"""

    def __init__(
        self,
        config: ChunkPipelineConfig,
        sample_rate: int,
        uri: str,
        on_segment: SegmentCallback,
    ):
        self.config = config
        self.sample_rate = sample_rate
        self.uri = uri
        self.on_segment = on_segment
        # 每个 speaker 未闭合的音频段（open-segment 管线，跨 commit 区拼接）。
        self._open_segments: dict[int, _OpenSegment] = {}

    def feed_pieces(self, pieces: dict[int, list[SpeakerPiece]]) -> None:
        """把归集好的帧片逐 speaker 送入 open-segment 管线。"""

        for global_id, speaker_pieces in sorted(pieces.items()):
            self._feed_speaker_pieces(global_id, speaker_pieces)

    def _feed_speaker_pieces(
        self, speaker_id: int, pieces: list[SpeakerPiece]
    ) -> None:
        """把单个 speaker 一个 commit 区内的帧片送入 open-segment 管线。

        帧片按 merge_gap 拆成连续 run 再逐 run 喂入：帧间隔超过
        segment_merge_gap 即断开（与 writer 逐帧喂入的合并判定等价，
        不会因窗级合并吞掉窗内较长的沉默）。音频只含活跃帧，run 内帧间
        的小于 merge_gap 的沉默不进入音频。
        """

        if not pieces:
            return
        runs: list[list[SpeakerPiece]] = [[pieces[0]]]
        for piece in pieces[1:]:
            if piece[0] - runs[-1][-1][1] <= self.config.segment_merge_gap:
                runs[-1].append(piece)
            else:
                runs.append([piece])
        for run in runs:
            audio_pieces = [
                audio
                for start, end, (source, origin) in run
                if (
                    audio := slice_absolute(
                        source, origin, start, end, self.sample_rate
                    )
                )
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
            if forward_gap <= self.config.segment_merge_gap:
                new_end = max(open_seg.end, end)
                # 段时长达到上限：先闭合当前段再以本 run 另开新段续接，避免
                # 连续讲话时 open segment 无界增长（内存驻留与 ASR 输出延迟
                # 失去上界）。切点落在 run 边界、人为切在语音中间，不依赖
                # 对端沉默；闭合/新开复用既有路径，帧数据不丢不重。
                if new_end - open_seg.start >= self.config.max_segment_duration:
                    self._close_segment(speaker_id)
                    self._open_segments[speaker_id] = _OpenSegment(
                        start=float(start), end=float(end), audios=[audio]
                    )
                    return
                open_seg.end = new_end
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

    def close_inactive(self, commit_end: float) -> None:
        """闭合已确认沉默的 open segment（与 writer.close_inactive 同语义）。

        提交区无缝拼接：一旦 commit_end - seg.end > merge_gap，该段未来不可能
        再被延长，提前闭合以降低输出延迟。
        """

        for speaker_id, open_seg in list(self._open_segments.items()):
            if commit_end - open_seg.end > self.config.segment_merge_gap:
                self._close_segment(speaker_id)

    def finalize(self) -> None:
        """音频结束：闭合所有残余 open segment。"""

        for speaker_id in list(self._open_segments.keys()):
            self._close_segment(speaker_id)
