"""零重写的 chunk 级流式 RTTM 写出器。

语义：

- confirmed speaker 的帧即时进入各自的 open-turn 管线；
- probationary（试用期）speaker 的帧先在内存 pending 缓冲中拼接，
  待身份定案（转正/被吸收）后由 `flush_speaker` 一次性追加写出；
- probationary 之间相互吸收时，被吸收者的 pending 缓冲由
  `defer_speaker` 改挂到目标名下继续等待，不提前落盘；
- 每个 speaker 维护一个未闭合的 open turn（驻留内存），仅在间隔超过
  merge_gap 或 EOF 时闭合写出——所有拼接都发生在写出之前；
- 每一行一旦写出即为最终，全程 append-only，`finalize` 也只追加不重写；
  `finalize` 还可在文件末尾以 # 注释写出 log speaker_id -> RTTM speaker
  的映射表（含被吸收者的 final id 与被丢弃者的标记）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .schema import SpeakerTurn
from .utils import ensure_parent_dir


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
        # probationary speaker 的缓冲 turn 列表（身份定案前不写出）。
        self._pending: dict[int, list[SpeakerTurn]] = {}
        # 内部 global id -> 输出 speaker 编号（按首次写出顺序）。
        self._output_ids: dict[int, int] = {}
        self._next_output_id = 0
        self._written_turns = 0

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
        self._append_line(self._format_line(turn, self._output_id(speaker_id)))
        self._written_turns += 1

    def _feed_turn(self, speaker_id: int, start: float, end: float) -> None:
        """把一段 [start, end) 送入 speaker 的 open-turn 管线（写出前拼接）。"""

        open_turn = self._open_turns.get(speaker_id)
        if open_turn is not None:
            forward_gap = start - open_turn.end
            backward_gap = open_turn.start - end
            # 双向间隔都不超过 merge_gap 视为同一 turn（跨 chunk 生效），直接合并；
            # 缓冲 flush 可能乱序到达，向过去重叠/相邻的段把起点往前拉。
            if forward_gap <= self.merge_gap and backward_gap <= self.merge_gap:
                open_turn.start = min(open_turn.start, start)
                open_turn.end = max(open_turn.end, end)
                return
        # 间隔过大（无论方向）：先闭合旧 turn 再开新 turn。
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

    def _feed_pending_frame(
        self,
        speaker_id: int,
        frame_start: float,
        frame_end: float,
    ) -> None:
        """probationary 帧进入内存缓冲，按 merge_gap 在缓冲内拼接。"""

        turns = self._pending.setdefault(int(speaker_id), [])
        if turns and frame_start - turns[-1].end <= self.merge_gap:
            turns[-1].end = max(turns[-1].end, float(frame_end))
            return
        turns.append(
            SpeakerTurn(
                start=float(frame_start),
                end=float(frame_end),
                speaker_id=int(speaker_id),
            )
        )

    def consume_chunk(
        self,
        seg_scores: np.ndarray,
        frame_step: float,
        chunk_start: float,
        commit_start: float,
        commit_end: float,
        local_to_global: dict[int, int],
        deferred_speakers: Optional[set[int]] = None,
    ) -> int:
        """消费一个 chunk 的帧级结果（仅 [commit_start, commit_end) 提交区），返回消费的帧数。

        `deferred_speakers` 中的 global id（probationary）只进内存缓冲，不写出。
        """

        if seg_scores.size == 0 or not local_to_global:
            return 0

        deferred = deferred_speakers or set()
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
                if speaker_id in deferred:
                    self._feed_pending_frame(speaker_id, frame_start, frame_end)
                else:
                    self._feed_frame(speaker_id, frame_start, frame_end)
            emitted += 1
        return emitted

    # ------------------------------------------------------------------
    # 身份定案后的缓冲 flush（仍是纯追加）
    # ------------------------------------------------------------------

    def flush_speaker(self, speaker_id: int, final_id: int) -> int:
        """把 speaker 的 pending 缓冲按时间序送入 final_id 的 open-turn 管线。

        与 final_id 当前 open turn 连续（间隔 ≤ merge_gap）的缓冲段直接拼接；
        返回 flush 的缓冲 turn 数。
        """

        turns = self._pending.pop(int(speaker_id), [])
        for turn in sorted(turns, key=lambda item: (item.start, item.end)):
            self._feed_turn(int(final_id), turn.start, turn.end)
        return len(turns)

    def defer_speaker(self, speaker_id: int, final_id: int) -> int:
        """把 speaker 的 pending 缓冲改挂到 final_id 名下（仍是 probationary）。

        用于 probationary 之间相互吸收：两路缓冲各自按时间有序，
        合并重排后按 merge_gap 重新拼接，继续等待 final_id 定案，
        不提前落盘。返回转移的缓冲 turn 数。
        """

        source_turns = self._pending.pop(int(speaker_id), [])
        if not source_turns:
            return 0
        target_turns = self._pending.pop(int(final_id), [])
        combined = sorted(
            source_turns + target_turns, key=lambda item: (item.start, item.end)
        )
        merged: list[SpeakerTurn] = []
        for turn in combined:
            turn.speaker_id = int(final_id)
            if merged and turn.start - merged[-1].end <= self.merge_gap:
                merged[-1].end = max(merged[-1].end, turn.end)
            else:
                merged.append(turn)
        self._pending[int(final_id)] = merged
        return len(source_turns)

    def drop_speaker(self, speaker_id: int) -> int:
        """丢弃 speaker 的 pending 缓冲（不输出 RTTM 行），返回丢弃的 turn 数。

        用于 output_unresolved_speakers=False 时清理结尾仍不确定的 speaker。
        """

        turns = self._pending.pop(int(speaker_id), [])
        if turns:
            logger.info(
                "[streaming] dropped %d pending turns for unresolved speaker %d",
                len(turns),
                speaker_id,
            )
        return len(turns)

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

    def finalize(self, final_id_map: Optional[dict[int, int]] = None) -> None:
        """闭合所有活跃 turn；残余 pending 按自身 id 兜底 flush。全程只追加。

        给出 `final_id_map`（log speaker_id -> 吸收链解析后的最终内部 id）时，
        在文件末尾以 # 注释写出 log speaker_id -> RTTM speaker 的映射表。
        """

        # 正常情况下 orchestrator 已在定案后 flush 全部 pending，这里仅兜底。
        for speaker_id in sorted(list(self._pending.keys())):
            flushed = self.flush_speaker(speaker_id, speaker_id)
            if flushed:
                logger.warning(
                    "[streaming] finalize: flushed %d residual pending turns for "
                    "speaker %d without resolution",
                    flushed,
                    speaker_id,
                )
        for speaker_id in list(self._open_turns.keys()):
            self._close_turn(speaker_id)

        logger.info(
            "[streaming] finalized turns=%d (append-only, no rewrite)",
            self._written_turns,
        )

        if final_id_map:
            self._write_id_map(final_id_map)

    def _write_id_map(self, final_id_map: dict[int, int]) -> None:
        """以 # 注释写出 log speaker_id -> RTTM speaker 编号的映射表。

        RTTM 行内的 speaker 编号按首次写出顺序分配（见 `_output_ids`）；
        被吸收的 id 标注其 final id，未产生任何输出行（缓冲被丢弃或
        片段全部短于 min_segment_duration）的 id 标记 <dropped>。
        """

        self._append_line("# speaker_id_map: log speaker_id -> rttm_speaker")
        for speaker_id in sorted(final_id_map):
            final_id = final_id_map[speaker_id]
            output_id = self._output_ids.get(final_id)
            target = str(output_id) if output_id is not None else "<dropped>"
            note = f" (absorbed into {final_id})" if final_id != speaker_id else ""
            self._append_line(f"#   {speaker_id} -> {target}{note}")


__all__ = ["AppendOnlyRTTMWriter"]
