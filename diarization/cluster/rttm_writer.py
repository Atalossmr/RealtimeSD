"""零重写的 chunk 级纯流式 RTTM 写出器（raw 级输出）。

语义：

- 所有 speaker 身份在分配时一次定案（无试用期、无吸收），帧判定后
  即时进入各自的 open-turn 管线，没有任何延迟缓冲；
- 每个 speaker 维护一个未闭合的 open turn（驻留内存），仅在间隔超过
  merge_gap 或 EOF 时闭合写出——所有拼接都发生在写出之前；
- 每一行一旦写出即为最终，全程 append-only，`finalize` 也只追加不重写；
- merge 事件对历史行的修正由 refined 级（cluster/post_merge.py）承担：
  raw 文件保持原样，refined 按需整体重生成；
- `finalize` 还可在文件末尾以 # 注释写出内部 global id -> RTTM speaker
  编号的映射表（RTTM 编号按首次写出顺序分配）。
"""

from __future__ import annotations

import logging

import numpy as np

from ..schema import SpeakerTurn
from ..utils import ensure_parent_dir


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
        # 内部 global id -> 输出 speaker 编号（按首次写出顺序）。
        self._output_ids: dict[int, int] = {}
        self._next_output_id = 0
        self._written_turns = 0

        ensure_parent_dir(self.output_path)
        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# chunk raw RTTM for {self.uri}\n")

    @property
    def output_id_map(self) -> dict[int, int]:
        """内部 global id -> 输出 speaker 编号的实时映射（副本）。

        文件末尾的 # id 映射表只在 finalize 写出；refined 级在流式期间
        动态重生成时通过该属性取实时映射。
        """

        return dict(self._output_ids)

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
        self._append_line(self._format_line(turn, self._output_id(speaker_id)))
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

    def _feed_frame(
        self,
        speaker_id: int,
        frame_start: float,
        frame_end: float,
    ) -> None:
        self._feed_turn(speaker_id, frame_start, frame_end)

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

        frame_step = max(1e-6, float(frame_step))
        emitted = 0
        num_frames = seg_scores.shape[0]
        for frame_idx in range(num_frames):
            frame_start = chunk_start + frame_idx * frame_step
            frame_end = frame_start + frame_step
            # 只输出与提交区相交的帧；跨界帧裁剪到提交区边界。
            # 相邻 chunk 的提交区无缝拼接，因此不重复也不遗漏。
            if frame_end <= commit_start + 1e-9:
                continue
            if frame_start >= commit_end - 1e-9:
                break
            frame_start = max(frame_start, commit_start)
            frame_end = min(frame_end, commit_end)

            frame_scores = seg_scores[frame_idx]
            # segmentation-3.0 经 powerset 转硬标签（0/1），> 0.0 即活跃；
            # 未分配 global（未建 track / 未过门控）的 local slot 帧直接丢弃。
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
        """闭合所有活跃 turn，全程只追加；末尾以 # 注释写出 id 映射表。"""

        for speaker_id in list(self._open_turns.keys()):
            self._close_turn(speaker_id)

        logger.info(
            "[streaming] finalized turns=%d (append-only, no rewrite)",
            self._written_turns,
        )

        if self._output_ids:
            self._write_id_map()

    def _write_id_map(self) -> None:
        """以 # 注释写出内部 global id -> RTTM speaker 编号的映射表。

        RTTM 行内的 speaker 编号按首次写出顺序分配（见 `_output_ids`），
        因此与内部 global id 并不一致，此表用于追溯对应关系。
        """

        self._append_line("# speaker_id_map: global_id -> rttm_speaker")
        for speaker_id in sorted(self._output_ids):
            self._append_line(f"#   {speaker_id} -> {self._output_ids[speaker_id]}")


__all__ = ["AppendOnlyRTTMWriter"]
