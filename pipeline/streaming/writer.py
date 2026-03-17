"""overlap 版本流式决策输出与 RTTM 写出模块。

本模块的核心职责是：

1) 把每个目标帧的离散说话人决策（可能包含多个 speaker）转换为更稳定的说话段（turn）；
2) 在保证 RTTM 尽量不碎片化的前提下，持续写出 RTTM；
3) 持续实时写出 RTTM，并在 merge 场景下做一致性补写。

新增约束（用于后续维护）：

- 说话人合并（merge）只允许合并不稳定（unstable）speaker；
- merge 触发 RTTM 补写时必须等待目标 speaker 的 active turn 结束；
- 任何新写出的 RTTM 段必须与已输出段做重叠检查，必要时裁掉重叠部分。
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from ..schema import (
    ActiveStreamingTurn,
    SpeakerTurn,
    StreamingFrameDecision,
)
from .merge_commit import (
    register_written_interval,
    subtract_overlaps,
)


logger = logging.getLogger(__name__)


class StreamingRTTMWriter:
    """把 overlap 版本的逐帧决策持续写成 RTTM。

    设计要点：

    - 以 speaker 为粒度维护活跃 turn（允许多个 speaker 同时 active，用于 overlap）；
    - 通过“稳定前缀 flush + 关闭时补尾段”的策略减少 RTTM 碎片；
    - 按 speaker 粒度实时写出 turn，并在 merge 场景下补写。

    额外能力：

    - 支持合并说话人（merge）后的 RTTM 补写策略；
    - 写 RTTM 时自动去除与已写 RTTM 的重叠区间（避免重复/冲突）。
    """

    def __init__(
        self,
        output_path: str,
        uri: str,
        min_segment_duration: float,
        flush_interval: float,
        merge_gap: float,
        delay_short_speaker_output: bool = True,
        show_rttm: bool = False,
        stable_speaker_ids_provider=None,
    ):
        """功能：初始化流式 RTTM 写出器。

        参数：
            output_path: RTTM 输出文件路径。
            uri: RTTM 中使用的音频标识。
            min_segment_duration: 最短写出片段时长阈值。
            flush_interval: 活跃 turn 前缀最小刷写间隔。
            merge_gap: turn 合并允许的间隔。
            delay_short_speaker_output: 是否缓存未稳定 speaker 的 turn。
            show_rttm: 是否同步打印新增 RTTM 行。
            stable_speaker_ids_provider: 返回稳定 speaker 集合的回调。
        """
        self.output_path = output_path
        self.uri = uri
        self.min_segment_duration = min_segment_duration
        self.flush_interval = max(0.1, flush_interval)
        self.merge_gap = max(0.0, merge_gap)
        self.delay_short_speaker_output = bool(delay_short_speaker_output)
        self.show_rttm = bool(show_rttm)
        self._stable_speaker_ids_provider = stable_speaker_ids_provider

        # `active_turns` 以 speaker 为单位维护当前仍在延展的说话段。
        # overlap 场景里，它天然允许多个 speaker 同时处于活跃状态。
        self.active_turns: dict[int, ActiveStreamingTurn] = {}

        self.pending_turns_by_speaker: dict[int, list[SpeakerTurn]] = {}
        self.total_duration_by_speaker: dict[int, float] = {}
        self.rttm_speaker_ids: dict[int, int] = {}
        self.next_rttm_speaker_id = 0

        # `written_turns` 用于记录已经写出的稳定 turn，仅用于调试和阅读，不参与匹配逻辑。
        self.written_turns: list[SpeakerTurn] = []

        # 记录已经写入 RTTM 的区间（按输出 speaker id 维护）。
        # 用于：
        # - merge 后补写历史 turn 时，裁剪掉与已输出 RTTM 的重叠部分；
        # - 避免任意场景下重复写出重叠 RTTM。
        self._written_intervals_by_output_speaker: dict[
            int, list[tuple[float, float]]
        ] = {}

        # speaker 合并相关状态。
        # - `_speaker_redirect` 用于把“旧 speaker id”映射到“合并后的 speaker id”（链式）。
        # - `_pending_merge_turns_by_speaker` 用于保存需要在 active 结束后补写的 RTTM turn。
        # - `_pending_merge_release` 用于标记“因 merge 达到稳定阈值，需要在 active 结束后一次性 flush 全部 RTTM”。
        self._speaker_redirect: dict[int, int] = {}
        self._pending_merge_turns_by_speaker: dict[int, list[SpeakerTurn]] = {}
        self._pending_merge_release: set[int] = set()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# overlap streaming RTTM for {self.uri}\n")

    def _format_rttm_line(
        self, turn: SpeakerTurn, output_speaker_id: int | None = None
    ) -> str:
        """功能：格式化单条 RTTM 记录文本。

        参数：
            turn: 说话段对象。
            output_speaker_id: 可选的输出 speaker id（用于重映射）。

        返回：
            一行 RTTM 文本。
        """
        duration = max(0.0, float(turn.end - turn.start))
        speaker_id = int(turn.speaker_id)
        final_speaker_id = (
            speaker_id if output_speaker_id is None else int(output_speaker_id)
        )
        return (
            f"SPEAKER {self.uri} 0 {turn.start:.3f} {duration:.3f} "
            f"<NA> <NA> {final_speaker_id} <NA> <NA>"
        )

    def _resolve_speaker_id(self, speaker_id: int) -> int:
        """解析 speaker 重定向后的最终 speaker id。

        说明：
        - merge 后，小 speaker 会被重定向到大 speaker；
        - 这里做简单路径压缩，避免长链。
        """

        speaker_id = int(speaker_id)
        parent = self._speaker_redirect.get(speaker_id)
        if parent is None:
            return speaker_id
        root = self._resolve_speaker_id(parent)
        self._speaker_redirect[speaker_id] = root
        return root

    def speaker_is_active(self, speaker_id: int) -> bool:
        """判断某个 speaker 是否处于 active turn 状态。"""

        speaker_id = self._resolve_speaker_id(int(speaker_id))
        return speaker_id in self.active_turns

    def speaker_total_duration(self, speaker_id: int) -> float:
        """返回某个 speaker 当前累计的说话时长（秒）。"""

        speaker_id = self._resolve_speaker_id(int(speaker_id))
        return float(self.total_duration_by_speaker.get(speaker_id, 0.0))

    def _is_stable_speaker(self, speaker_id: int) -> bool:
        """判断 speaker 是否稳定（由上层提供判定）。"""

        if not self.delay_short_speaker_output:
            return True
        if self._stable_speaker_ids_provider is None:
            return False
        stable_ids = self._stable_speaker_ids_provider()
        return int(self._resolve_speaker_id(int(speaker_id))) in {
            int(s) for s in stable_ids
        }

    def _turn_to_rttm_line(self, turn: SpeakerTurn) -> str:
        """把说话段转换为 RTTM 文本行（用于日志展示）。"""

        speaker_id = int(self._resolve_speaker_id(turn.speaker_id))
        output_speaker_id = int(self.rttm_speaker_ids.get(speaker_id, speaker_id))
        return self._format_rttm_line(
            SpeakerTurn(
                start=float(turn.start), end=float(turn.end), speaker_id=speaker_id
            ),
            output_speaker_id,
        )

    def _flush_pending_for_speaker_if_stable(self, speaker_id: int) -> None:
        """当 speaker 稳定且 inactive 时，刷出其缓存 turn。"""

        speaker_id = int(self._resolve_speaker_id(speaker_id))
        if not self.delay_short_speaker_output:
            return
        if not self._is_stable_speaker(speaker_id):
            return
        if self.speaker_is_active(speaker_id):
            return
        pending_turns = self.pending_turns_by_speaker.pop(speaker_id, [])
        if pending_turns:
            logger.info(
                "[streaming] speaker %s became stable; flushing %d cached RTTM turns",
                speaker_id,
                len(pending_turns),
            )
        for turn in pending_turns:
            self._write_turn(turn)

    def _subtract_overlaps(
        self,
        output_speaker_id: int,
        start: float,
        end: float,
    ) -> list[tuple[float, float]]:
        """从 [start, end) 中裁掉已写 RTTM 的重叠部分。

        返回若干个不重叠子区间（可能为空）。
        """

        # 使用独立 overlap 工具函数，便于针对“区间裁剪”做纯函数级别测试。
        intervals = self._written_intervals_by_output_speaker.get(
            int(output_speaker_id), []
        )
        return subtract_overlaps(
            intervals=intervals, start=float(start), end=float(end)
        )

    def _register_written_interval(
        self,
        output_speaker_id: int,
        start: float,
        end: float,
    ) -> None:
        """登记一个已写出的 RTTM 区间，并保持区间列表有序且不重叠。"""

        output_speaker_id = int(output_speaker_id)
        start = float(start)
        end = float(end)
        if end <= start:
            return

        intervals = list(
            self._written_intervals_by_output_speaker.get(output_speaker_id, [])
        )
        self._written_intervals_by_output_speaker[output_speaker_id] = (
            register_written_interval(intervals=intervals, start=start, end=end)
        )

    def _write_turn(self, turn: SpeakerTurn) -> None:
        """把一个已经足够稳定的说话段真正写入 RTTM（带重叠裁剪）。"""

        duration = max(0.0, turn.end - turn.start)
        if duration < self.min_segment_duration:
            return
        speaker_id = int(self._resolve_speaker_id(turn.speaker_id))
        # 恢复“global speaker -> RTTM speaker”连续映射：
        # RTTM speaker id 按首次写出顺序连续分配，避免输出 id 跳号。
        if speaker_id not in self.rttm_speaker_ids:
            self.rttm_speaker_ids[speaker_id] = self.next_rttm_speaker_id
            self.next_rttm_speaker_id += 1
        output_speaker_id = int(self.rttm_speaker_ids[speaker_id])

        # merge/补写等场景可能导致“同一 speaker 同一时间段”被再次写出。
        # 这里统一做区间差集，裁掉已输出 RTTM 的重叠部分。
        segments = self._subtract_overlaps(
            output_speaker_id, float(turn.start), float(turn.end)
        )
        if not segments:
            return

        with open(self.output_path, "a", encoding="utf-8") as file_obj:
            for seg_start, seg_end in segments:
                line = self._format_rttm_line(
                    SpeakerTurn(start=seg_start, end=seg_end, speaker_id=speaker_id),
                    output_speaker_id,
                )
                file_obj.write(f"{line}\n")
                if self.show_rttm:
                    sys.stdout.write(f"{line}\n")
                    sys.stdout.flush()
                self._register_written_interval(output_speaker_id, seg_start, seg_end)

        self.written_turns.append(
            SpeakerTurn(start=turn.start, end=turn.end, speaker_id=speaker_id)
        )

    def _record_turn(self, turn: SpeakerTurn, *, count_towards_total: bool) -> None:
        """功能：记录一个 turn，并按模式决定立即写出或缓存。

        参数：
            turn: 待记录的说话段。
            count_towards_total: 是否计入 speaker 累计时长。
        """
        speaker_id = int(self._resolve_speaker_id(turn.speaker_id))
        duration = max(0.0, float(turn.end - turn.start))

        if count_towards_total:
            self.total_duration_by_speaker[speaker_id] = (
                float(self.total_duration_by_speaker.get(speaker_id, 0.0)) + duration
            )

        if duration < self.min_segment_duration:
            return
        if self.delay_short_speaker_output and not self._is_stable_speaker(speaker_id):
            self.pending_turns_by_speaker.setdefault(speaker_id, []).append(turn)
            return
        self._write_turn(turn)

    def _flush_confirmed_prefix(self, speaker_id: int, stable_until: float) -> None:
        """只把某个 speaker 已确认稳定的前缀部分写出去。

        这里仍保留“稳定前缀”思路：
        - 一个活跃 turn 不会在每一帧都整段重写；
        - 只有当它的前半段已经足够稳定时，才把前缀写入 RTTM；
        - overlap 场景里，每个 speaker 都独立做这个动作。
        """

        speaker_id = int(self._resolve_speaker_id(speaker_id))
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

        speaker_id = int(self._resolve_speaker_id(speaker_id))
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

        # turn 关闭意味着该 speaker 当前不再 active；
        # 如果 merge 触发了“补写/释放请求”，这里是最安全的落盘时机。
        self._flush_pending_merge_outputs_if_ready(speaker_id)

    def _flush_pending_merge_outputs_if_ready(self, speaker_id: int) -> None:
        """在 speaker 非 active 时，处理 merge 触发的补写/释放请求。"""

        speaker_id = int(self._resolve_speaker_id(speaker_id))
        if self.speaker_is_active(speaker_id):
            return

        pending_turns = self._pending_merge_turns_by_speaker.pop(speaker_id, [])
        for turn in pending_turns:
            self._write_turn(turn)
        self._flush_pending_for_speaker_if_stable(speaker_id)

        if speaker_id in self._pending_merge_release:
            self._pending_merge_release.discard(speaker_id)

    def notify_speaker_became_inactive(self, speaker_id: int) -> None:
        """通知某个 speaker 已结束 active，可尝试处理 merge 补写/释放。"""

        self._flush_pending_merge_outputs_if_ready(int(speaker_id))
        self._flush_pending_for_speaker_if_stable(int(speaker_id))

    def _extend_or_start_turn(
        self,
        speaker_id: int,
        seg_start: float,
        seg_end: float,
    ) -> None:
        """延长已有 turn，或为该 speaker 新开一个 turn。"""

        speaker_id = int(self._resolve_speaker_id(speaker_id))
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
        stable_until: float,
    ) -> None:
        """消费新一批逐帧决策。

        overlap 版本的重点是：
        - `decision.speakers` 可以同时包含多个 speaker；
        - 每个 speaker 的 turn 独立延展；
        - 不会因为 overlap 帧每来一次就立刻写一小段 RTTM。

        注意：此函数仅维护 RTTM turn 状态，不再返回音轨写入事件。
        """

        if not decisions:
            return

        for decision in decisions:
            seg_start = max(0.0, float(decision.start))
            seg_end = max(seg_start, float(decision.end))
            # 对输入 speaker id 先做 merge 重定向，保证后续状态管理只使用 canonical id。
            active_speakers = set(
                int(self._resolve_speaker_id(speaker_id))
                for speaker_id in decision.speakers
            )

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

    def handle_speaker_merge(
        self,
        *,
        large_speaker_id: int,
        small_speaker_id: int,
        merge_time: float,
    ) -> None:
        """处理一次 speaker 合并事件，并按规则决定 RTTM 的补写方式。

        规则（按需求实现）：

        - 仅允许 unstable speaker 被合并：本函数假设 small_speaker_id 是 unstable。
        - 合并后若目标 speaker 仍为 unstable：不输出 RTTM（仍继续缓存）。
        - 合并后若目标 speaker 由 unstable -> stable：需要输出合并后 speaker 的“全部 RTTM”（含历史缓存）。
        - 若目标 speaker 合并前已 stable：仅输出“被合并进来的 RTTM 片段”。

        重要约束：

        - merge 触发的任何 RTTM 写出必须等待目标 speaker 的 active turn 结束；
        - 写出时会自动裁剪掉与既有 RTTM 的重叠部分。
        """

        large_id = int(self._resolve_speaker_id(large_speaker_id))
        small_id = int(self._resolve_speaker_id(small_speaker_id))
        if large_id == small_id:
            return

        logger.info(
            "[streaming] merge_event large=%s small=%s merge_time=%.3f",
            large_id,
            small_id,
            float(merge_time),
        )

        merge_time = float(merge_time)
        if merge_time < 0.0:
            merge_time = 0.0

        # 1) 如果 small 仍处于 active，则把它的未刷尾段强制截断到 merge_time。
        small_state = self.active_turns.pop(small_id, None)
        if small_state is not None:
            tail_end = min(float(small_state.end), merge_time)
            tail_start = float(small_state.flushed_until)
            if tail_end > tail_start + 1e-6:
                self._pending_merge_turns_by_speaker.setdefault(large_id, []).append(
                    SpeakerTurn(start=tail_start, end=tail_end, speaker_id=large_id)
                )

        # 2) 把 small 的待补写 RTTM turn 和累计时长合并到 large。
        #    这里的 turn 需要改写 speaker_id 为 large，保证最终输出的 speaker id 一致。
        small_pending = list(self._pending_merge_turns_by_speaker.get(small_id, []))
        if small_pending:
            self._pending_merge_turns_by_speaker.pop(small_id, None)

        # 3) 把 small 的延迟 RTTM 缓存也迁移到 large。
        #    否则 redirect 生效后，旧 key 下的缓存在 flush 阶段将不可达，导致 RTTM 丢段。
        small_delayed_pending = list(self.pending_turns_by_speaker.get(small_id, []))
        if small_delayed_pending:
            self.pending_turns_by_speaker.pop(small_id, None)

        small_total = float(self.total_duration_by_speaker.get(small_id, 0.0))
        if small_total > 0.0:
            self.total_duration_by_speaker[large_id] = (
                float(self.total_duration_by_speaker.get(large_id, 0.0)) + small_total
            )
        self.total_duration_by_speaker.pop(small_id, None)

        self.rttm_speaker_ids.pop(small_id, None)

        transferred_turns = [
            SpeakerTurn(start=float(t.start), end=float(t.end), speaker_id=large_id)
            for t in small_pending
        ]
        transferred_delayed_turns = [
            SpeakerTurn(start=float(t.start), end=float(t.end), speaker_id=large_id)
            for t in small_delayed_pending
        ]

        if transferred_turns:
            queue = self._pending_merge_turns_by_speaker.setdefault(large_id, [])
            queue.extend(transferred_turns)
            queue.sort(key=lambda turn: (float(turn.start), float(turn.end)))
        if transferred_delayed_turns:
            delayed_queue = self.pending_turns_by_speaker.setdefault(large_id, [])
            delayed_queue.extend(transferred_delayed_turns)
            delayed_queue.sort(key=lambda turn: (float(turn.start), float(turn.end)))
        if not self.speaker_is_active(large_id):
            self._flush_pending_merge_outputs_if_ready(large_id)
        self._speaker_redirect[small_id] = large_id

    def finalize(self) -> None:
        """在音频结束时把尚未写出的 turn 全部刷盘。"""

        for speaker_id in sorted(list(self.active_turns.keys())):
            self._close_turn(speaker_id, force=True)
        self.active_turns.clear()

        # 结束时，merge 触发的补写/释放请求也需要处理。
        for speaker_id in sorted(list(self._pending_merge_turns_by_speaker.keys())):
            self._flush_pending_merge_outputs_if_ready(speaker_id)
        for speaker_id in sorted(list(self._pending_merge_release)):
            self._flush_pending_merge_outputs_if_ready(speaker_id)

        for speaker_id in sorted(list(self.pending_turns_by_speaker.keys())):
            self._flush_pending_for_speaker_if_stable(speaker_id)

        # 音频结束后仍未稳定的 speaker：输出其缓存片段，便于排查误检或短时说话人。
        unstable_speakers = [
            int(sid)
            for sid, turns in self.pending_turns_by_speaker.items()
            if turns and not self._is_stable_speaker(int(sid))
        ]
        for speaker_id in sorted(unstable_speakers):
            turns = self.pending_turns_by_speaker.get(speaker_id, [])
            if not turns:
                continue
            logger.info(
                "[streaming] unstable speaker %s cached RTTM turns at finalize:\n%s",
                int(self._resolve_speaker_id(speaker_id)),
                "\n".join(self._turn_to_rttm_line(turn) for turn in turns),
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

    def speaker_export_metadata(self) -> dict[int, dict[str, int | float | bool]]:
        """返回说话人音轨导出所需的稳定性和编号信息。"""

        # 注意：merge 可能导致多个内部 speaker id 重定向到同一个 canonical id。
        # 为避免导出阶段出现“重复 speaker/错误稳定性”，这里按 canonical speaker 聚合。
        all_speakers = set(self.total_duration_by_speaker.keys())
        all_speakers.update(self.rttm_speaker_ids.keys())
        all_speakers.update(self._speaker_redirect.keys())
        all_speakers.update(self._speaker_redirect.values())

        aggregated_total: dict[int, float] = {}
        for sid in all_speakers:
            canonical = int(self._resolve_speaker_id(int(sid)))
            # total_duration_by_speaker 始终按 canonical speaker 计数；
            # 这里聚合时只读取 canonical 对应的 total，避免重复累加。
            aggregated_total[canonical] = float(
                self.total_duration_by_speaker.get(int(canonical), 0.0)
            )

        metadata: dict[int, dict[str, int | float | bool]] = {}
        for speaker_id in sorted(aggregated_total.keys()):
            is_stable = bool(self._is_stable_speaker(speaker_id))
            output_speaker_id = int(self.rttm_speaker_ids.get(speaker_id, speaker_id))
            metadata[speaker_id] = {
                "speaker_id": int(speaker_id),
                "output_speaker_id": int(output_speaker_id),
                "total_duration": float(aggregated_total.get(speaker_id, 0.0)),
                "is_stable": bool(is_stable),
            }
        return metadata


def quantize_decision_time(frame_step: float, time_value: float) -> float:
    """把目标帧时间量化到统一时间桶，避免重复输出。

    参数:
        frame_step: 量化的时间步长（通常使用 advance_step）
        time_value: 待量化的绝对时间（秒）
    """

    frame_step = max(1e-6, float(frame_step))
    bucket = int(np.floor(max(0.0, time_value) / frame_step))
    return (bucket + 0.5) * frame_step
