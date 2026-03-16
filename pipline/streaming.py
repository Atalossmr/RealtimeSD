"""overlap 版本流式决策输出与 RTTM 写出模块。"""

from __future__ import annotations

import logging
import os

import numpy as np

from .schema import ActiveStreamingTurn, SpeakerTurn, StreamingFrameDecision


logger = logging.getLogger(__name__)


class StreamingRTTMWriter:
    """把 overlap 版本的逐帧决策持续写成 RTTM。

    和原版相比，这里最重要的差别是：
    - 不再维护单一 `pending_write_turn`；
    - 而是按 speaker 分别维护活跃 turn；
    - 因此在 overlap 场景里，两个 speaker 可以同时延展各自的 turn；
    - 写出时也不会因为某一帧刚经过就立即把一小段 RTTM 切出来。

    这能显著减少 overlap 场景下“每帧一过就写一小段 RTTM”的碎片问题。
    """

    def __init__(
        self,
        output_path: str,
        uri: str,
        min_segment_duration: float,
        flush_interval: float,
        merge_gap: float,
        delay_short_speaker_output: bool = False,
        speaker_min_total_duration_to_emit: float = 0.0,
    ):
        self.output_path = output_path
        self.uri = uri
        self.min_segment_duration = min_segment_duration
        self.flush_interval = max(0.1, flush_interval)
        self.merge_gap = max(0.0, merge_gap)
        self.delay_short_speaker_output = bool(delay_short_speaker_output)
        self.speaker_min_total_duration_to_emit = max(
            0.0, float(speaker_min_total_duration_to_emit)
        )

        # `active_turns` 以 speaker 为单位维护当前仍在延展的说话段。
        # overlap 场景里，它天然允许多个 speaker 同时处于活跃状态。
        self.active_turns: dict[int, ActiveStreamingTurn] = {}

        # 当启用“短 speaker 延迟输出”时，先把未达阈值 speaker 的 turn 缓存在内存，
        # 等累计时长超过阈值后再一次性补写此前片段。
        self.pending_turns_by_speaker: dict[int, list[SpeakerTurn]] = {}
        self.total_duration_by_speaker: dict[int, float] = {}
        self.speaker_release_state: dict[int, bool] = {}
        self.rttm_speaker_ids: dict[int, int] = {}
        self.next_rttm_speaker_id = 0

        # `written_turns` 用于记录已经写出的稳定 turn，仅用于调试和阅读，不参与匹配逻辑。
        self.written_turns: list[SpeakerTurn] = []

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# overlap streaming RTTM for {self.uri}\n")

    def _write_turn(self, turn: SpeakerTurn) -> None:
        """把一个已经足够稳定的说话段真正写入 RTTM。"""

        duration = max(0.0, turn.end - turn.start)
        if duration < self.min_segment_duration:
            return
        speaker_id = int(turn.speaker_id)
        output_speaker_id = speaker_id
        if self.delay_short_speaker_output:
            if speaker_id not in self.rttm_speaker_ids:
                self.rttm_speaker_ids[speaker_id] = self.next_rttm_speaker_id
                self.next_rttm_speaker_id += 1
            output_speaker_id = self.rttm_speaker_ids[speaker_id]
        with open(self.output_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(
                f"SPEAKER {self.uri} 0 {turn.start:.3f} {duration:.3f} <NA> <NA> {output_speaker_id} <NA> <NA>\n"
            )
        self.written_turns.append(turn)

    def _speaker_is_released(self, speaker_id: int) -> bool:
        if not self.delay_short_speaker_output:
            return True
        return bool(self.speaker_release_state.get(int(speaker_id), False))

    def _turn_to_rttm_line(self, turn: SpeakerTurn) -> str:
        duration = max(0.0, float(turn.end - turn.start))
        return (
            f"SPEAKER {self.uri} 0 {turn.start:.3f} {duration:.3f} "
            f"<NA> <NA> {turn.speaker_id} <NA> <NA>"
        )

    def _release_speaker_if_ready(self, speaker_id: int) -> None:
        speaker_id = int(speaker_id)
        if not self.delay_short_speaker_output:
            return
        if self.speaker_release_state.get(speaker_id, False):
            return
        total_duration = float(self.total_duration_by_speaker.get(speaker_id, 0.0))
        if total_duration + 1e-6 < self.speaker_min_total_duration_to_emit:
            return

        pending_turns = list(self.pending_turns_by_speaker.get(speaker_id, []))
        for turn in pending_turns:
            self._write_turn(turn)
        self.pending_turns_by_speaker.pop(speaker_id, None)
        self.speaker_release_state[speaker_id] = True
        logger.info(
            "[streaming] speaker %s reached emission threshold %.3fs with total %.3fs; released %d cached RTTM turns as rttm speaker %s",
            speaker_id,
            self.speaker_min_total_duration_to_emit,
            total_duration,
            len(pending_turns),
            self.rttm_speaker_ids.get(speaker_id, speaker_id),
        )

    def _record_turn(self, turn: SpeakerTurn, *, count_towards_total: bool) -> None:
        speaker_id = int(turn.speaker_id)
        duration = max(0.0, float(turn.end - turn.start))
        if duration < self.min_segment_duration:
            return

        if count_towards_total:
            self.total_duration_by_speaker[speaker_id] = (
                float(self.total_duration_by_speaker.get(speaker_id, 0.0)) + duration
            )

        if self._speaker_is_released(speaker_id):
            self._write_turn(turn)
            return

        self.pending_turns_by_speaker.setdefault(speaker_id, []).append(turn)
        self._release_speaker_if_ready(speaker_id)

    def _flush_confirmed_prefix(self, speaker_id: int, stable_until: float) -> None:
        """只把某个 speaker 已确认稳定的前缀部分写出去。

        这里仍保留“稳定前缀”思路：
        - 一个活跃 turn 不会在每一帧都整段重写；
        - 只有当它的前半段已经足够稳定时，才把前缀写入 RTTM；
        - overlap 场景里，每个 speaker 都独立做这个动作。
        """

        state = self.active_turns.get(speaker_id)
        if state is None:
            return

        candidate_end = min(state.end, stable_until)
        flush_start = state.flushed_until
        duration = candidate_end - flush_start
        if duration + 1e-6 < self.flush_interval:
            return

        self._record_turn(
            SpeakerTurn(
                start=flush_start,
                end=candidate_end,
                speaker_id=speaker_id,
            ),
            count_towards_total=True,
        )
        state.flushed_until = candidate_end

    def _close_turn(self, speaker_id: int, force: bool = False) -> None:
        """关闭某个 speaker 的当前活跃 turn。

        当一个 speaker 在当前帧集合里不再活跃时：
        - 先尽量把稳定前缀刷出去；
        - 剩余尾段如果足够长，再在关闭时一次性写出；
        - overlap 场景中，这能避免每一帧都把短尾巴写成 RTTM 碎片。
        """

        state = self.active_turns.pop(speaker_id, None)
        if state is None:
            return

        remaining_start = state.flushed_until
        remaining_end = state.end
        remaining_duration = remaining_end - remaining_start
        if force or remaining_duration + 1e-6 >= self.min_segment_duration:
            self._record_turn(
                SpeakerTurn(
                    start=remaining_start,
                    end=remaining_end,
                    speaker_id=speaker_id,
                ),
                count_towards_total=True,
            )

    def _extend_or_start_turn(
        self,
        speaker_id: int,
        seg_start: float,
        seg_end: float,
    ) -> None:
        """延长已有 turn，或为该 speaker 新开一个 turn。"""

        if speaker_id in self.active_turns:
            state = self.active_turns[speaker_id]
            if seg_start <= state.end + self.merge_gap + 1e-6:
                state.end = max(state.end, seg_end)
                return

            # 如果新片段和已有 turn 已经断开太远，则先关闭旧 turn，再开新 turn。
            self._close_turn(speaker_id, force=True)

        self.active_turns[speaker_id] = ActiveStreamingTurn(
            start=seg_start,
            end=seg_end,
            flushed_until=seg_start,
        )

    def consume(
        self,
        decisions: list[StreamingFrameDecision],
        frame_step: float,
        stable_until: float,
    ) -> None:
        """消费新一批逐帧决策。

        overlap 版本的重点是：
        - `decision.speakers` 可以同时包含多个 speaker；
        - 每个 speaker 的 turn 独立延展；
        - 不会因为 overlap 帧每来一次就立刻写一小段 RTTM。
        """

        if not decisions:
            return

        half = frame_step / 2.0
        for decision in decisions:
            seg_start = max(0.0, float(decision.time - half))
            seg_end = float(decision.time + half)
            active_speakers = set(int(speaker_id) for speaker_id in decision.speakers)

            # 对当前帧里已经不再活跃的 speaker，先尝试刷稳定前缀，再关闭其 turn。
            to_close = [
                speaker_id
                for speaker_id in list(self.active_turns.keys())
                if speaker_id not in active_speakers
            ]
            for speaker_id in to_close:
                self._flush_confirmed_prefix(speaker_id, stable_until)
                self._close_turn(speaker_id, force=False)

            # 对当前仍然活跃的 speaker，分别延展或新开 turn。
            for speaker_id in sorted(active_speakers):
                self._extend_or_start_turn(speaker_id, seg_start, seg_end)

            # 对仍然活跃中的所有 speaker，继续尝试刷新稳定前缀。
            # 这一步是 overlap 版本避免 RTTM 碎片的关键：
            # 我们只刷“稳定前缀”，而不是每一帧都把末尾部分立刻落盘。
            for speaker_id in sorted(self.active_turns):
                self._flush_confirmed_prefix(speaker_id, stable_until)

    def finalize(self) -> None:
        """在音频结束时把尚未写出的 turn 全部刷盘。"""

        for speaker_id in sorted(list(self.active_turns.keys())):
            self._close_turn(speaker_id, force=True)
        self.active_turns.clear()

        if self.delay_short_speaker_output:
            for speaker_id in sorted(list(self.pending_turns_by_speaker.keys())):
                self._release_speaker_if_ready(speaker_id)
            for speaker_id in sorted(list(self.pending_turns_by_speaker.keys())):
                pending_turns = self.pending_turns_by_speaker.get(speaker_id, [])
                if not pending_turns:
                    continue
                total_duration = float(
                    self.total_duration_by_speaker.get(speaker_id, 0.0)
                )
                logger.info(
                    "[streaming] speaker %s stayed below emission threshold %.3fs (total %.3fs); cached RTTM-style turns before external remapping:\n%s",
                    speaker_id,
                    self.speaker_min_total_duration_to_emit,
                    total_duration,
                    "\n".join(self._turn_to_rttm_line(turn) for turn in pending_turns),
                )
            if self.rttm_speaker_ids:
                mapping_lines = [
                    f"internal speaker {speaker_id} -> RTTM speaker {rttm_speaker_id}"
                    for speaker_id, rttm_speaker_id in sorted(
                        self.rttm_speaker_ids.items(), key=lambda item: item[1]
                    )
                ]
                logger.info(
                    "[streaming] final internal-to-RTTM speaker mapping:\n%s",
                    "\n".join(mapping_lines),
                )


def quantize_decision_time(step: float, time_value: float) -> float:
    """把目标帧时间量化到统一时间桶，避免重复输出。"""

    frame_step = max(1e-6, float(step))
    bucket = int(np.floor(max(0.0, time_value) / frame_step))
    return (bucket + 0.5) * frame_step
