"""实时全局 speaker 分配与 centroid 维护模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import (
    BufferedDecisionWindow,
    GlobalSpeakerDebug,
    MergedSpeakerDebug,
    ResolvedDecisionWindow,
    SegmentObservation,
    WindowDebugInfo,
)
from .utils import l2_normalize


@dataclass
class UpdateSegmentRecord:
    """记录某个 global speaker 上一次用于更新 centroid 的片段。"""

    start: float
    end: float


class IncrementalCentroidClusterer:
    """实时全局 speaker 分配器。

    依据以下逻辑：
    - 构造 local x global 的 cost matrix；
    - 用 Hungarian algorithm 做联合分配；

    防止同时活跃的局部说话人被贴到同一个全局说话人
    """

    def __init__(
        self,
        delta_new: float,
        max_speakers: int,
        global_match_threshold: float,
        merge_threshold: float,
        sma_window: int,
        update_segment_overlap_threshold: float,
        weak_update_similarity_margin: float,
        weak_update_weight_multiplier: float,
    ):
        self.delta_new = float(delta_new)
        self.max_speakers = max(1, int(max_speakers))
        self.global_match_threshold = float(global_match_threshold)
        self.merge_threshold = float(merge_threshold)
        self.sma_window = max(1, int(sma_window))
        self.ema_alpha = 2.0 / (self.sma_window + 1.0)
        self.update_segment_overlap_threshold = float(update_segment_overlap_threshold)
        self.weak_update_similarity_margin = float(weak_update_similarity_margin)
        self.weak_update_weight_multiplier = float(weak_update_weight_multiplier)

        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        self.last_update_segments: dict[int, UpdateSegmentRecord] = {}

        self.next_speaker_id = 0
        self.window_counter = 0
        self.pending_resolved_windows: list[ResolvedDecisionWindow] = []

    def _merge_speakers_if_needed(self) -> list[MergedSpeakerDebug]:
        """在匹配前尝试合并相似 speaker，并始终把小 speaker 合并进大 speaker。"""

        merge_events: list[MergedSpeakerDebug] = []
        while len(self.centroids) >= 2:
            speaker_ids = sorted(self.centroids.keys())
            centroid_matrix = np.stack([self.centroids[sid] for sid in speaker_ids])
            similarity_matrix = np.matmul(centroid_matrix, centroid_matrix.T)
            np.fill_diagonal(similarity_matrix, -1.0)

            best_flat = int(np.argmax(similarity_matrix))
            best_similarity = float(similarity_matrix.flat[best_flat])
            if best_similarity < self.merge_threshold:
                break

            left_idx, right_idx = np.unravel_index(best_flat, similarity_matrix.shape)
            left_id = speaker_ids[left_idx]
            right_id = speaker_ids[right_idx]

            left_count = self.counts[left_id]
            right_count = self.counts[right_id]
            if left_count >= right_count:
                large_id, small_id = left_id, right_id
            else:
                large_id, small_id = right_id, left_id

            total = self.counts[large_id] + self.counts[small_id]
            merged_center = (
                self.centroids[large_id] * self.counts[large_id]
                + self.centroids[small_id] * self.counts[small_id]
            ) / float(total)
            self.centroids[large_id] = l2_normalize(merged_center)
            self.counts[large_id] = total

            large_record = self.last_update_segments.get(large_id)
            small_record = self.last_update_segments.get(small_id)
            if large_record is None and small_record is not None:
                self.last_update_segments[large_id] = small_record
            elif (
                large_record is not None
                and small_record is not None
                and small_record.end > large_record.end
            ):
                self.last_update_segments[large_id] = small_record

            del self.centroids[small_id]
            del self.counts[small_id]
            self.last_update_segments.pop(small_id, None)

            merge_events.append(
                {
                    "large": int(large_id),
                    "small": int(small_id),
                    "similarity": float(best_similarity),
                    "merged_count": int(total),
                }
            )

        return merge_events

    def _default_debug_info(self) -> WindowDebugInfo:
        # 这里返回固定结构的 TypedDict，而不是宽泛的 `dict[str, object]`。
        # 这样 Pylance 就能确认这些键实际是 list，从而消除 `.append()` / `.extend()` 误报。
        return {
            "num_centroids_before": len(self.centroids),
            "num_centroids_after": len(self.centroids),
            "assignment_cost_matrix": None,
            "local_assignments": [],
            "new_speakers": [],
            "merged_speakers": [],
            "updated_speakers": [],
            "skipped_updates": [],
            "global_speakers": [],
        }

    def current_global_speakers(self) -> list[GlobalSpeakerDebug]:
        speakers: list[GlobalSpeakerDebug] = []
        for speaker_id in sorted(self.centroids):
            centroid = self.centroids[speaker_id]
            speakers.append(
                {
                    "speaker": int(speaker_id),
                    "count": int(self.counts.get(speaker_id, 0)),
                    "dim": int(centroid.shape[0]),
                }
            )
        return speakers

    def _create_speaker(self, observation: SegmentObservation) -> int:
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        self.centroids[speaker_id] = l2_normalize(
            observation.embedding.astype(np.float32, copy=False)
        )
        self.counts[speaker_id] = 1
        self.last_update_segments[speaker_id] = UpdateSegmentRecord(
            start=float(observation.start),
            end=float(observation.end),
        )
        return speaker_id

    def _segment_overlap_ratio(
        self,
        left: UpdateSegmentRecord,
        right: UpdateSegmentRecord,
    ) -> float:
        overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
        min_duration = max(1e-6, min(left.end - left.start, right.end - right.start))
        return float(overlap / min_duration)

    def _should_skip_update(
        self,
        speaker_id: int,
        observation: SegmentObservation,
    ) -> tuple[bool, float]:
        previous = self.last_update_segments.get(speaker_id)
        if previous is None:
            return False, 0.0
        current = UpdateSegmentRecord(
            start=float(observation.start),
            end=float(observation.end),
        )
        overlap_ratio = self._segment_overlap_ratio(previous, current)
        return overlap_ratio >= self.update_segment_overlap_threshold, overlap_ratio

    def _update_speaker(
        self,
        speaker_id: int,
        observation: SegmentObservation,
        weight_multiplier: float = 1.0,
    ) -> tuple[str, float]:
        """更新对应 Global Speaker 的聚类中心 (Centroid)。

        参数:
            speaker_id: 匹配到的全局说话人 ID
            observation: 用于更新的观测片段特征
            weight_multiplier: 更新权重的乘子（用于降低 overlap fallback 等不确定场景下的更新幅度，防止身份漂移）

        更新策略:
            - 早期 (count < sma_window): 使用简单移动平均 (SMA, Simple Moving Average) 快速拉入特征
            - 后期 (count >= sma_window): 使用指数移动平均 (EMA, Exponential Moving Average) 稳定维持特征
        """
        embedding = l2_normalize(observation.embedding.astype(np.float32, copy=False))
        centroid = self.centroids[speaker_id]
        count = self.counts[speaker_id]

        if count < self.sma_window:
            alpha = 1.0 / float(count + 1)
            alpha *= weight_multiplier
            updated = (1.0 - alpha) * centroid + alpha * embedding
            mode = "sma"
            if weight_multiplier == 1.0:
                self.counts[speaker_id] = count + 1
        else:
            alpha = self.ema_alpha * weight_multiplier
            updated = (1.0 - alpha) * centroid + alpha * embedding
            mode = "ema"
            if weight_multiplier == 1.0:
                self.counts[speaker_id] = count + 1

        self.centroids[speaker_id] = l2_normalize(updated)
        self.last_update_segments[speaker_id] = UpdateSegmentRecord(
            start=float(observation.start),
            end=float(observation.end),
        )
        return mode, alpha

    def _similarity_vector(
        self,
        observation: SegmentObservation,
        global_ids: list[int],
    ) -> np.ndarray:
        if not global_ids:
            return np.zeros((0,), dtype=np.float32)
        centroid_matrix = np.stack([self.centroids[sid] for sid in global_ids])
        similarities = np.matmul(centroid_matrix, observation.embedding)
        return similarities.astype(np.float32, copy=False)

    def _build_assignment(
        self,
        observations: list[SegmentObservation],
        debug_info: WindowDebugInfo,
    ) -> dict[int, tuple[Optional[int], float]]:
        """用 cost matrix + Hungarian algorithm 计算当前窗口的联合分配。

        解决重叠语音核心问题：
        如果同一个目标时刻有两个 local slot 同时活跃（例如 A 和 B 都在说话），
        且它们都在一定程度上和同一个全局说话人 centroid 相似。
        如果采用贪心匹配，它们很可能都会被错误地归为同一个 global speaker，从而漏掉其中一人。

        使用匈牙利算法（Hungarian algorithm）可以在同一窗口内，对所有 local speaker
        和所有 global speaker 统一进行二分图最大权匹配。
        这隐式地实现了一个“Cannot-Link”约束：同一个窗口中的不同 local slot，绝对不会被分配给同一个 global speaker。

        返回值是：
        - key: `local_idx`
        - value: `(matched_global_id_or_None, similarity)`

        如果没有任何已有 global speaker，则所有 local 都返回 `(None, -1.0)`，
        后续逻辑会根据 `delta_new` 决定是否新建 speaker。
        """

        if not self.centroids:
            return {obs.local_idx: (None, -1.0) for obs in observations}

        global_ids = sorted(self.centroids.keys())
        num_locals = len(observations)
        num_globals = len(global_ids)

        similarities = np.zeros((num_locals, num_globals), dtype=np.float32)
        for row_idx, observation in enumerate(observations):
            similarities[row_idx] = self._similarity_vector(observation, global_ids)

        # Hungarian algorithm 是最小化 cost，因此这里用 `1 - similarity`。
        cost_matrix = 1.0 - similarities
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        debug_info["assignment_cost_matrix"] = {
            "global_ids": [int(speaker_id) for speaker_id in global_ids],
            "cost_matrix": cost_matrix.tolist(),
            "similarity_matrix": similarities.tolist(),
        }

        assignment: dict[int, tuple[Optional[int], float]] = {
            obs.local_idx: (None, -1.0) for obs in observations
        }
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            observation = observations[row_idx]
            matched_global = global_ids[col_idx]
            similarity = float(similarities[row_idx, col_idx])
            assignment[observation.local_idx] = (matched_global, similarity)
        return assignment

    def start_window(
        self,
        *,
        target_time: float,
        target_local_indices: list[int],
        chunk_start_time: float,
        segmentation: np.ndarray,
        absolute_centers: np.ndarray,
        observations: list[SegmentObservation],
    ) -> BufferedDecisionWindow:
        window = BufferedDecisionWindow(
            window_id=self.window_counter,
            target_time=target_time,
            target_local_indices=list(target_local_indices),
            chunk_start_time=chunk_start_time,
            segmentation=segmentation,
            absolute_centers=absolute_centers,
            observations=observations,
        )
        self.window_counter += 1
        return window

    def push_window(
        self,
        window: BufferedDecisionWindow,
        *,
        force_flush: bool = False,
        return_debug: bool = False,
    ) -> (
        list[ResolvedDecisionWindow]
        | tuple[list[ResolvedDecisionWindow], list[WindowDebugInfo]]
    ):
        resolved = self._resolve_window(window)
        self.pending_resolved_windows.append(resolved)
        if return_debug:
            return [resolved], [resolved.debug_info]
        return [resolved]

    def pop_committable_windows(
        self,
        commit_limit: float,
        force_flush: bool = False,
    ) -> list[ResolvedDecisionWindow]:
        committed: list[ResolvedDecisionWindow] = []
        remaining: list[ResolvedDecisionWindow] = []
        for resolved in self.pending_resolved_windows:
            if not force_flush and resolved.window.target_time > commit_limit + 1e-6:
                remaining.append(resolved)
                continue
            committed.append(resolved)
        self.pending_resolved_windows = remaining
        committed.sort(
            key=lambda item: (item.window.target_time, item.window.window_id)
        )
        return committed

    def _resolve_window(self, window: BufferedDecisionWindow) -> ResolvedDecisionWindow:
        debug_info = self._default_debug_info()
        local_to_global: dict[int, int] = {}

        if not window.observations:
            return ResolvedDecisionWindow(
                window=window,
                local_to_global=local_to_global,
                debug_info=debug_info,
            )

        observations_by_local: dict[int, list[SegmentObservation]] = {}
        for observation in window.observations:
            observations_by_local.setdefault(observation.local_idx, []).append(
                observation
            )

        selected_observations: list[SegmentObservation] = []
        for local_idx in sorted(observations_by_local):
            candidates = observations_by_local[local_idx]
            candidates.sort(
                key=lambda item: (
                    item.score_at_target,
                    item.mean_activity,
                    item.duration,
                ),
                reverse=True,
            )
            selected_observations.append(candidates[0])

        # overlap 版本中，speaker merge 放在匹配前执行。
        # 这样 Hungarian 看到的 cost matrix 始终对应“当前最新的 global speaker 集合”。
        merge_events = self._merge_speakers_if_needed()
        if merge_events:
            debug_info["merged_speakers"].extend(merge_events)

        assignment = self._build_assignment(selected_observations, debug_info)

        for observation in selected_observations:
            local_idx = observation.local_idx
            matched_speaker, similarity = assignment.get(local_idx, (None, -1.0))

            # 情况 1：Hungarian 给出一个已有 speaker，且相似度足够高，直接采用。
            if (
                matched_speaker is not None
                and similarity >= self.global_match_threshold
            ):
                assigned_speaker = matched_speaker
                decision = "matched"

            # 情况 2：与所有旧 speaker 都不够像，则尝试创建新 speaker。
            elif len(self.centroids) < self.max_speakers and (
                matched_speaker is None or similarity < self.delta_new
            ):
                assigned_speaker = self._create_speaker(observation)
                decision = "new"
                similarity = 1.0

            # 情况 3：Hungarian 给出了一个最接近 speaker，但没到主阈值，也没有低到一定新建。
            # overlap 版本里仍保留一个 fallback，以便在证据不足时延续旧身份。
            # 这种匹配在置信度不高时默认不更新 centroid。
            elif matched_speaker is not None:
                assigned_speaker = matched_speaker
                decision = "fallback"

            else:
                continue

            local_to_global[int(local_idx)] = int(assigned_speaker)
            debug_info["local_assignments"].append(
                {
                    "local": int(local_idx),
                    "global": int(assigned_speaker),
                    "decision": decision,
                    "similarity": float(similarity),
                    "score_at_target": float(observation.score_at_target),
                    "mean_activity": float(observation.mean_activity),
                    "speech_ratio": float(observation.speech_ratio),
                    "selection_mode": observation.selection_mode,
                    "start": float(observation.start),
                    "end": float(observation.end),
                }
            )

            if decision == "new":
                debug_info["new_speakers"].append(
                    {
                        "local": int(local_idx),
                        "global": int(assigned_speaker),
                        "start": float(observation.start),
                        "end": float(observation.end),
                    }
                )
                continue

            if not observation.allow_centroid_update:
                # [弱更新策略]
                # 当 observation 为 overlap_fallback（即在纯净无重叠区域找不到片段，只能在包含重叠的区域提取时），
                # 为了防止聚类中心被重叠的人声污染，原本是一律跳过更新的。
                # 但这里引入高置信度弱更新：如果这个片段与匹配到的 Global Speaker 极其相似，
                # 并且相似度超过 `global_match_threshold + weak_update_similarity_margin`，
                # 则说明我们很确信这就是他，此时允许以可配置的衰减权重进行轻微更新。
                # 这能够有效解决长时间争吵/重叠导致第二说话人身份轨迹偏离的问题。
                if (
                    similarity
                    > self.global_match_threshold + self.weak_update_similarity_margin
                ):
                    should_skip, overlap_ratio = self._should_skip_update(
                        assigned_speaker,
                        observation,
                    )
                    if should_skip:
                        debug_info["skipped_updates"].append(
                            {
                                "global": int(assigned_speaker),
                                "reason": "segment_overlap_during_weak_update",
                                "overlap_ratio": float(overlap_ratio),
                                "start": float(observation.start),
                                "end": float(observation.end),
                            }
                        )
                        continue

                    mode, alpha = self._update_speaker(
                        assigned_speaker,
                        observation,
                        weight_multiplier=self.weak_update_weight_multiplier,
                    )
                    debug_info["updated_speakers"].append(
                        {
                            "global": int(assigned_speaker),
                            "mode": mode + "_weak",
                            "alpha": float(alpha),
                            "start": float(observation.start),
                            "end": float(observation.end),
                        }
                    )
                else:
                    debug_info["skipped_updates"].append(
                        {
                            "global": int(assigned_speaker),
                            "reason": "overlap_fallback_observation",
                            "selection_mode": observation.selection_mode,
                            "start": float(observation.start),
                            "end": float(observation.end),
                        }
                    )
                continue

            if decision == "fallback":
                debug_info["skipped_updates"].append(
                    {
                        "global": int(assigned_speaker),
                        "reason": "matched_fallback",
                        "selection_mode": observation.selection_mode,
                        "start": float(observation.start),
                        "end": float(observation.end),
                    }
                )
                continue

            should_skip, overlap_ratio = self._should_skip_update(
                assigned_speaker,
                observation,
            )
            if should_skip:
                debug_info["skipped_updates"].append(
                    {
                        "global": int(assigned_speaker),
                        "reason": "segment_overlap",
                        "overlap_ratio": float(overlap_ratio),
                        "start": float(observation.start),
                        "end": float(observation.end),
                    }
                )
                continue

            mode, alpha = self._update_speaker(assigned_speaker, observation)
            debug_info["updated_speakers"].append(
                {
                    "global": int(assigned_speaker),
                    "mode": mode,
                    "alpha": float(alpha),
                    "start": float(observation.start),
                    "end": float(observation.end),
                }
            )

        debug_info["num_centroids_after"] = len(self.centroids)
        debug_info["global_speakers"] = self.current_global_speakers()
        return ResolvedDecisionWindow(
            window=window,
            local_to_global=local_to_global,
            debug_info=debug_info,
        )
