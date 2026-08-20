"""零重写的 chunk 级纯流式 RTTM 写出器（raw 级输出）。

语义：

- 所有 speaker 身份在分配时一次定案（无试用期、无吸收），帧判定后
  即时进入各自的 open-turn 管线，没有任何延迟缓冲；
- 每个 speaker 维护一个未闭合的 open turn（驻留内存），仅在间隔超过
  merge_gap 或 EOF 时闭合写出——所有拼接都发生在写出之前；
- 每一行一旦写出即为最终，全程 append-only，`finalize` 也只追加不重写；
- merge 事件对历史行的修正由 refined 级（cluster/post_merge.py）承担：
  raw 文件保持原样，refined 按需整体重生成。

RTTM 行内的 speaker 字段直接写 assigner 的 global id，无额外编号映射。
"""

from __future__ import annotations

import logging

import numpy as np

from ..schema import SpeakerTurn
from ..utils import ensure_parent_dir, iter_commit_frames


logger = logging.getLogger(__name__)


class AppendOnlyRTTMWriter:
    """chunk 粒度的流式 RTTM 输出器（零重写）。"""

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

        # 每个 speaker 当前未闭合的 turn（活跃记录，驻留内存，写出前可持续拼接）。
        self._open_turns: dict[int, SpeakerTurn] = {}
        self._written_turns = 0

        ensure_parent_dir(self.output_path)
        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# chunk raw RTTM for {self.uri}\n")

    # ------------------------------------------------------------------
    # 流式写出
    # ------------------------------------------------------------------

    def _format_line(self, turn: SpeakerTurn) -> str:
        duration = max(0.0, float(turn.end - turn.start))
        return (
            f"SPEAKER {self.uri} 0 {turn.start:.3f} {duration:.3f} "
            f"<NA> <NA> {int(turn.speaker_id)} <NA> <NA>"
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
        self._append_line(self._format_line(turn))
        self._written_turns += 1

    def _feed_turn(self, speaker_id: int, start: float, end: float) -> None:
        """把一段 [start, end) 送入 speaker 的 open-turn 管线（写出前拼接）。"""

        open_turn = self._open_turns.get(speaker_id)
        if open_turn is not None:
            forward_gap = start - open_turn.end
            # 间隔不超过 merge_gap 视为同一 turn（跨 chunk 生效），直接合并。
            if forward_gap <= self.merge_gap:
                open_turn.end = max(open_turn.end, end)
                return
        # 间隔过大：先闭合旧 turn 再开新 turn。
        self._close_turn(speaker_id)
        self._open_turns[speaker_id] = SpeakerTurn(
            start=float(start), end=float(end), speaker_id=int(speaker_id)
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
        """消费一个 chunk 的帧级结果（仅 [commit_start, commit_end) 提交区），返回消费的帧数。

        所有已分配 global id 的帧即时进入 open-turn 管线，无延迟缓冲。
        """

        if seg_scores.size == 0 or not local_to_global:
            return 0

        emitted = 0
        for frame_start, frame_end, active_globals in iter_commit_frames(
            seg_scores,
            frame_step,
            chunk_start,
            commit_start,
            commit_end,
            local_to_global,
        ):
            for speaker_id in active_globals:
                self._feed_turn(speaker_id, frame_start, frame_end)
            emitted += 1
        return emitted

    def close_inactive(self, commit_end: float) -> int:
        """闭合已确认沉默的 open turn（chunk 末尾不再活跃者），返回闭合数。

        提交区无缝拼接：下一窗提交区起点即当前 commit_end，因此一旦
        `commit_end - turn.end > merge_gap`，该 turn 未来不可能再被延长，
        提前闭合与惰性闭合（下次开口回头闭合/EOF 闭合）产出完全相同，
        只是把写出提前，降低尾段输出延迟。
        """

        closed = 0
        for speaker_id, turn in list(self._open_turns.items()):
            if commit_end - turn.end > self.merge_gap:
                self._close_turn(speaker_id)
                closed += 1
        return closed

    # ------------------------------------------------------------------
    # 收尾（纯追加，不重写）
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """闭合所有活跃 turn，全程只追加、不重写。"""

        for speaker_id in list(self._open_turns.keys()):
            self._close_turn(speaker_id)

        logger.info(
            "[streaming] finalized turns=%d (append-only, no rewrite)",
            self._written_turns,
        )


__all__ = ["AppendOnlyRTTMWriter"]
