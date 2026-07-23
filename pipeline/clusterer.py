"""chunk 级全局 speaker 分配与 centroid 维护。

与 `pipeline/clustering/clusterer.py` 的差异：

- 处理单元是一个 chunk 的一组 local track observation，而不是 0.5s 滑窗；
- 不做 speaker merge：confirmed speaker 身份一旦建立永不改变；
- 新建 speaker 进入 probationary 试用期，累计足够语音才转正；
  试用期内若与某个 confirmed speaker 足够相似，则被吸收（absorb），
  吸收只影响后续 chunk 与定案事件（resolved_speakers），
  probationary 的输出在定案前由 writer 缓冲，定案后按 final_id flush。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ChunkPipelineConfig
from .schema import ChunkDebugInfo, ChunkObservation
from .utils import l2_normalize


class ChunkSpeakerClusterer:
    """chunk 粒度的实时全局 speaker 分配器。

    每个 chunk 内用 Hungarian algorithm 做 local->global 联合分配，
    隐式提供 cannot-link 约束：同一 chunk 的不同 local slot 不会分配给同一 global speaker。
    """

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config

        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}

        # probationary 状态： probationary 集合 + 各 speaker 累计匹配语音时长。
        self.probationary: set[int] = set()
        self.speaker_speech: dict[int, float] = {}

        # 吸收产生的 global id 重定向（定案信息的记录，供日志与追溯）。
        self.redirect_map: dict[int, int] = {}

        self.next_speaker_id = 0

    # ------------------------------------------------------------------
    # 基本状态查询
    # ------------------------------------------------------------------

    def current_speaker_ids(self) -> set[int]:
        """返回当前维护的全部 global speaker id。"""

        return set(self.centroids.keys())

    def current_global_speakers(self) -> list[dict[str, int | str]]:
        """返回当前全局说话人摘要列表（含 probationary 状态）。"""

        speakers: list[dict[str, int | str]] = []
        for speaker_id in sorted(self.centroids):
            centroid = self.centroids[speaker_id]
            speakers.append(
                {
                    "speaker": int(speaker_id),
                    "count": int(self.counts.get(speaker_id, 0)),
                    "dim": int(centroid.shape[0]),
                    "status": (
                        "probationary" if speaker_id in self.probationary else "confirmed"
                    ),
                }
            )
        return speakers

    # ------------------------------------------------------------------
    # centroid 更新（纯 SMA，逻辑平移自旧 clusterer）
    # ------------------------------------------------------------------

    def _update_speaker(
        self,
        speaker_id: int,
        observation: ChunkObservation,
    ) -> tuple[str, float]:
        """以简单移动平均（SMA）增量更新对应 global speaker 的 centroid。"""

        if observation.embedding is None:
            return "sma", 0.0
        embedding = l2_normalize(observation.embedding.astype(np.float32, copy=False))
        centroid = self.centroids[speaker_id]
        count = self.counts[speaker_id]

        alpha = 1.0 / float(count + 1)
        updated = (1.0 - alpha) * centroid + alpha * embedding
        self.counts[speaker_id] = count + 1

        self.centroids[speaker_id] = l2_normalize(updated)
        return "sma", alpha

    def _create_speaker(self, observation: ChunkObservation) -> int:
        """基于 observation 新建 probationary 全局说话人。"""

        if observation.embedding is None:
            raise ValueError("Cannot create speaker from observation without embedding")
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        self.centroids[speaker_id] = l2_normalize(
            observation.embedding.astype(np.float32, copy=False)
        )
        self.counts[speaker_id] = 1
        self.probationary.add(speaker_id)
        self.speaker_speech[speaker_id] = float(observation.duration)
        return speaker_id

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------

    def _resolve_redirect(self, speaker_id: int) -> int:
        """沿 redirect_map 解析最终 id（吸收目标恒为 confirmed，链长至多为 1，做通用压缩）。"""

        seen: set[int] = set()
        current = int(speaker_id)
        while current in self.redirect_map and current not in seen:
            seen.add(current)
            current = int(self.redirect_map[current])
        return current

    def _build_assignment(
        self,
        observations: list[ChunkObservation],
    ) -> dict[int, tuple[Optional[int], float]]:
        """用 cost matrix + Hungarian algorithm 计算本 chunk 的联合分配。"""

        if not self.centroids:
            return {obs.local_idx: (None, -1.0) for obs in observations}

        global_ids = sorted(self.centroids.keys())
        centroid_matrix = np.stack([self.centroids[sid] for sid in global_ids])

        similarities = np.zeros((len(observations), len(global_ids)), dtype=np.float32)
        for row_idx, observation in enumerate(observations):
            if observation.embedding is None:
                continue
            similarities[row_idx] = np.matmul(
                centroid_matrix, observation.embedding
            ).astype(np.float32, copy=False)

        cost_matrix = 1.0 - similarities
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment: dict[int, tuple[Optional[int], float]] = {
            obs.local_idx: (None, -1.0) for obs in observations
        }
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            observation = observations[row_idx]
            if observation.embedding is None:
                continue
            assignment[observation.local_idx] = (
                global_ids[col_idx],
                float(similarities[row_idx, col_idx]),
            )
        return assignment

    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[dict[int, int], ChunkDebugInfo]:
        """完成一个 chunk 的 local->global 分配、centroid 更新与 probation 维护。"""

        debug_info: ChunkDebugInfo = {
            "num_centroids_before": len(self.centroids),
            "num_centroids_after": len(self.centroids),
            "local_assignments": [],
            "new_speakers": [],
            "updated_speakers": [],
            "skipped_updates": [],
            "absorb_events": [],
            "resolved_speakers": [],
            "global_speakers": [],
        }
        local_to_global: dict[int, int] = {}

        embedded = [obs for obs in observations if obs.embedding is not None]
        if embedded:
            assignment = self._build_assignment(embedded)
            for observation in embedded:
                self._resolve_observation(
                    observation, assignment, local_to_global, debug_info
                )

        self._maintain_probation(debug_info)

        # 本 chunk 分配到的 speaker 可能在 probation 维护中被吸收，
        # 通过 redirect_map 把映射解析到最终 id，避免向已删除的 id 输出。
        if self.redirect_map:
            local_to_global = {
                local_idx: self._resolve_redirect(global_id)
                for local_idx, global_id in local_to_global.items()
            }

        debug_info["num_centroids_after"] = len(self.centroids)
        debug_info["global_speakers"] = self.current_global_speakers()
        return local_to_global, debug_info

    def _resolve_observation(
        self,
        observation: ChunkObservation,
        assignment: dict[int, tuple[Optional[int], float]],
        local_to_global: dict[int, int],
        debug_info: ChunkDebugInfo,
    ) -> None:
        """对单个 observation 做 matched/new/fallback 判定并更新状态。"""

        config = self.config
        local_idx = observation.local_idx
        matched_speaker, similarity = assignment.get(local_idx, (None, -1.0))

        if matched_speaker is not None and similarity >= config.global_match_threshold:
            assigned_speaker = matched_speaker
            decision = "matched"
        elif (
            len(self.centroids) < config.max_speakers
            and (matched_speaker is None or similarity < config.new_speaker_threshold)
            and (observation.duration >= config.min_segment_duration_for_new_speaker)
        ):
            assigned_speaker = self._create_speaker(observation)
            decision = "new"
        elif matched_speaker is not None:
            assigned_speaker = matched_speaker
            decision = "fallback"
        else:
            return

        local_to_global[int(local_idx)] = int(assigned_speaker)
        debug_info["local_assignments"].append(
            {
                "local": int(local_idx),
                "global": int(assigned_speaker),
                "decision": decision,
                "similarity": float(similarity),
                "start": float(observation.start),
                "end": float(observation.end),
                "selection_mode": observation.selection_mode,
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
            return

        # matched / fallback 都计入该 speaker 的累计匹配语音（用于转正判定）。
        self.speaker_speech[assigned_speaker] = self.speaker_speech.get(
            assigned_speaker, 0.0
        ) + float(observation.duration)

        if not observation.allow_centroid_update:
            # overlap_fallback 片段不参与 centroid 更新。
            debug_info["skipped_updates"].append(
                {
                    "global": int(assigned_speaker),
                    "reason": "overlap_fallback_observation",
                    "selection_mode": observation.selection_mode,
                    "start": float(observation.start),
                    "end": float(observation.end),
                }
            )
            return

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
            return

        if observation.duration < config.min_segment_duration_for_centroid_update:
            debug_info["skipped_updates"].append(
                {
                    "global": int(assigned_speaker),
                    "reason": "segment_too_short_for_update",
                    "selection_mode": observation.selection_mode,
                    "start": float(observation.start),
                    "end": float(observation.end),
                }
            )
            return

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

    # ------------------------------------------------------------------
    # probationary 维护与终局收尾
    # ------------------------------------------------------------------

    def _maintain_probation(self, debug_info: ChunkDebugInfo) -> None:
        """转正累计语音足够的 probationary，并吸收与 confirmed 过于相似的。"""

        for speaker_id in sorted(list(self.probationary)):
            if (
                self.speaker_speech.get(speaker_id, 0.0)
                >= self.config.probation_confirm_duration
            ):
                self.probationary.discard(speaker_id)
                debug_info["resolved_speakers"].append(
                    {
                        "speaker_id": int(speaker_id),
                        "final_id": int(speaker_id),
                        "resolution": "confirmed",
                    }
                )
                continue

            self._try_absorb(speaker_id, debug_info)

    def _try_absorb(
        self,
        speaker_id: int,
        debug_info: ChunkDebugInfo,
    ) -> bool:
        """若 probationary 与某 confirmed speaker 足够相似，则把它吸收掉。

        吸收只影响后续 chunk 的分配与定案事件（resolved_speakers）；
        被吸收者的缓冲输出由 writer 按 final_id flush，不回改任何已写出行。
        """

        if speaker_id not in self.centroids:
            self.probationary.discard(speaker_id)
            return False

        confirmed_ids = [
            sid for sid in self.centroids if sid not in self.probationary
        ]
        if not confirmed_ids:
            return False

        source = self.centroids[speaker_id]
        best_target: Optional[int] = None
        best_similarity = -1.0
        for target_id in confirmed_ids:
            similarity = float(np.dot(source, self.centroids[target_id]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_target = target_id

        if best_target is None or best_similarity < self.config.absorb_threshold:
            return False

        # centroid 按 counts 加权并入目标 speaker。
        source_count = self.counts.get(speaker_id, 1)
        target_count = self.counts.get(best_target, 1)
        total = source_count + target_count
        merged = (
            self.centroids[best_target] * target_count + source * source_count
        ) / float(total)
        self.centroids[best_target] = l2_normalize(merged)
        self.counts[best_target] = total
        self.speaker_speech[best_target] = self.speaker_speech.get(
            best_target, 0.0
        ) + self.speaker_speech.get(speaker_id, 0.0)

        del self.centroids[speaker_id]
        del self.counts[speaker_id]
        self.speaker_speech.pop(speaker_id, None)
        self.probationary.discard(speaker_id)
        self.redirect_map[int(speaker_id)] = int(best_target)

        debug_info["absorb_events"].append(
            {
                "absorbed": int(speaker_id),
                "into": int(best_target),
                "similarity": float(best_similarity),
            }
        )
        debug_info["resolved_speakers"].append(
            {
                "speaker_id": int(speaker_id),
                "final_id": int(best_target),
                "resolution": "absorbed",
                "similarity": float(best_similarity),
            }
        )
        return True

    def finalize_redirects(self) -> list[dict[str, int | float | str]]:
        """音频结束时收尾：定案全部残余 probationary。

        够像某 confirmed 的吸收并入（记录 redirect_map），不够像的转正保留自身 id。
        返回定案列表（元素含 speaker_id / final_id / resolution），
        供 writer 逐个 flush 缓冲输出。
        """

        debug_info: ChunkDebugInfo = {
            "num_centroids_before": len(self.centroids),
            "num_centroids_after": len(self.centroids),
            "local_assignments": [],
            "new_speakers": [],
            "updated_speakers": [],
            "skipped_updates": [],
            "absorb_events": [],
            "resolved_speakers": [],
            "global_speakers": [],
        }
        for speaker_id in sorted(list(self.probationary)):
            absorbed = self._try_absorb(speaker_id, debug_info)
            if not absorbed:
                debug_info["resolved_speakers"].append(
                    {
                        "speaker_id": int(speaker_id),
                        "final_id": int(speaker_id),
                        "resolution": "confirmed",
                    }
                )
            self.probationary.discard(speaker_id)
        return list(debug_info["resolved_speakers"])


__all__ = ["ChunkSpeakerClusterer"]
