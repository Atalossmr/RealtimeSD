"""append-only 的 chunk 级 RTTM 写出器。

与 `pipeline/streaming/writer.py` 的差异：

- 输入是整个 chunk 的帧级多标签分数 + local->global 映射，而非 0.5s 帧决策；
- 无 pending turn、无 stable 判定、无 merge 补写与区间差集；
- 已提交内容流式期间不回改；身份修正只在 `finalize` 时按 redirect_map
  对内存中的 turn 做一次终局 remap 并整文件重写。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .schema import SpeakerTurn
from ..utils import ensure_parent_dir


logger = logging.getLogger(__name__)


class AppendOnlyRTTMWriter:
    """chunk 粒度的流式 RTTM 输出器。"""

    def __init__(
        self,
        output_path: str,
        uri: str,
        min_segment_duration: float,
        merge_gap: float,
        show_rttm: bool = False,
    ):
        self.output_path = output_path
        self.uri = uri
        self.min_segment_duration = max(0.0, float(min_segment_duration))
        self.merge_gap = max(0.0, float(merge_gap))
        self.show_rttm = bool(show_rttm)

        # 每个 global speaker 当前未闭合的 turn。
        self._open_turns: dict[int, SpeakerTurn] = {}
        # 已闭合并写出的 turn（内存留存，供终局 remap）。
        self._closed_turns: list[SpeakerTurn] = []
        # 内部 global id -> 输出 speaker 编号（按首次写出顺序）。
        self._output_ids: dict[int, int] = {}
        self._next_output_id = 0

        ensure_parent_dir(self.output_path)
        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# chunk streaming RTTM for {self.uri}\n")

    # ------------------------------------------------------------------
    # 流式写出
    # ------------------------------------------------------------------

    def _output_id(self, speaker_id: int) -> int:
        if speaker_id not in self._output_ids:
            self._output_ids[speaker_id] = self._next_output_id
            self._next_output_id += 1
        return self._output_ids[speaker_id]

    def _format_line(self, turn: SpeakerTurn, output_id: int) -> str:
        duration = max(0.0, float(turn.end - turn.start))
        return (
            f"SPEAKER {self.uri} 0 {turn.start:.3f} {duration:.3f} "
            f"<NA> <NA> {int(output_id)} <NA> <NA>"
        )

    def _append_line(self, line: str) -> None:
        with open(self.output_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(line + "\n")
        if self.show_rttm:
            print(line)

    def _close_turn(self, speaker_id: int) -> None:
        turn = self._open_turns.pop(speaker_id, None)
        if turn is None:
            return
        if turn.end - turn.start < self.min_segment_duration:
            return
        self._closed_turns.append(turn)
        self._append_line(self._format_line(turn, self._output_id(speaker_id)))

    def _feed_frame(
        self,
        speaker_id: int,
        frame_start: float,
        frame_end: float,
    ) -> None:
        open_turn = self._open_turns.get(speaker_id)
        if open_turn is not None and frame_start - open_turn.end <= self.merge_gap:
            open_turn.end = max(open_turn.end, frame_end)
            return
        self._close_turn(speaker_id)
        self._open_turns[speaker_id] = SpeakerTurn(
            start=float(frame_start), end=float(frame_end), speaker_id=int(speaker_id)
        )

    def consume_chunk(
        self,
        seg_scores: np.ndarray,
        frame_step: float,
        chunk_start: float,
        commit_start: float,
        commit_end: float,
        local_to_global: dict[int, int],
    ) -> int:
        """消费一个 chunk 的帧级结果（仅 [commit_start, commit_end) 提交区），返回写入的帧数。"""

        if seg_scores.size == 0 or not local_to_global:
            return 0

        frame_step = max(1e-6, float(frame_step))
        emitted = 0
        num_frames = seg_scores.shape[0]
        for frame_idx in range(num_frames):
            frame_start = chunk_start + frame_idx * frame_step
            frame_end = frame_start + frame_step
            if frame_end <= commit_start + 1e-9:
                continue
            if frame_start >= commit_end - 1e-9:
                break
            frame_start = max(frame_start, commit_start)
            frame_end = min(frame_end, commit_end)

            frame_scores = seg_scores[frame_idx]
            active_globals = sorted(
                {
                    int(local_to_global[local_idx])
                    for local_idx in range(len(frame_scores))
                    if frame_scores[local_idx] > 0.0
                    and local_idx in local_to_global
                }
            )
            for speaker_id in active_globals:
                self._feed_frame(speaker_id, frame_start, frame_end)
            emitted += 1
        return emitted

    # ------------------------------------------------------------------
    # 终局 remap
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_redirect(redirect_map: dict[int, int], speaker_id: int) -> int:
        """沿重定向链解析最终 id（吸收只指向 confirmed，链长至多为 1，这里做通用压缩）。"""

        seen: set[int] = set()
        current = int(speaker_id)
        while current in redirect_map and current not in seen:
            seen.add(current)
            current = int(redirect_map[current])
        return current

    def finalize(self, redirect_map: Optional[dict[int, int]] = None) -> None:
        """闭合所有活跃 turn，应用身份重定向并整文件重写一次。"""

        for speaker_id in list(self._open_turns.keys()):
            self._close_turn(speaker_id)

        turns = list(self._closed_turns)
        if redirect_map:
            for turn in turns:
                turn.speaker_id = self._resolve_redirect(
                    redirect_map, turn.speaker_id
                )

        # 终局编号：按 turn 首次出现的时间顺序重映射为连续 id。
        ordered = sorted(turns, key=lambda turn: (turn.start, turn.speaker_id))
        final_ids: dict[int, int] = {}
        for turn in ordered:
            if turn.speaker_id not in final_ids:
                final_ids[turn.speaker_id] = len(final_ids)

        # remap 后同一输出 speaker 的相邻 turn 可能变成可合并的，做一次拼接。
        merged: list[tuple[float, float, int]] = []
        for turn in ordered:
            output_id = final_ids[turn.speaker_id]
            if (
                merged
                and merged[-1][2] == output_id
                and turn.start - merged[-1][1] <= self.merge_gap
            ):
                prev = merged[-1]
                merged[-1] = (prev[0], max(prev[1], turn.end), output_id)
            else:
                merged.append((float(turn.start), float(turn.end), output_id))

        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# chunk streaming RTTM for {self.uri}\n")
            for start, end, output_id in merged:
                duration = max(0.0, end - start)
                file_obj.write(
                    f"SPEAKER {self.uri} 0 {start:.3f} {duration:.3f} "
                    f"<NA> <NA> {int(output_id)} <NA> <NA>\n"
                )

        logger.info(
            "[streaming] finalized turns=%d remapped=%s",
            len(merged),
            bool(redirect_map),
        )


__all__ = ["AppendOnlyRTTMWriter"]
