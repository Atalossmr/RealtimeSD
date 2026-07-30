"""chunk 级全局 speaker 分配与 centroid 维护。

设计要点：

- 处理单元是一个 chunk 的一组 local track observation，而不是 0.5s 滑窗；
- 纯流式：新建 speaker 立即成为永久身份，无试用期、无吸收、无合并，
  所有分配判定一次定案，身份一旦建立永不改变；
- 无身份修正出口：false split 与 false glue 均不可修复，因此阈值策略
  应为"宁可 glue 不可 split"（new_speaker_threshold 应适当调高）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import ChunkPipelineConfig
from .assigners import BaseChunkAssigner
from ..schema import ChunkDebugInfo, ChunkObservation
from ..utils import l2_normalize


class ChunkSpeakerClusterer(BaseChunkAssigner):
    """chunk 粒度的实时全局 speaker 分配器。

    每个 chunk 内用 Hungarian algorithm 做 local->global 联合分配，
    隐式提供 cannot-link 约束：同一 chunk 的不同 local slot 不会分配给同一 global speaker。
    """

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config

        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}

        self.next_speaker_id = 0

    # ------------------------------------------------------------------
    # 基本状态查询
    # ------------------------------------------------------------------

    def current_speaker_ids(self) -> set[int]:
        """返回当前维护的全部 global speaker id。"""

        return set(self.centroids.keys())

    def current_global_speakers(self) -> list[dict[str, int]]:
        """返回当前全局说话人摘要列表。"""

        speakers: list[dict[str, int]] = []
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
        """基于 observation 新建全局说话人（立即成为永久身份）。"""

        if observation.embedding is None:
            raise ValueError("Cannot create speaker from observation without embedding")
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        self.centroids[speaker_id] = l2_normalize(
            observation.embedding.astype(np.float32, copy=False)
        )
        self.counts[speaker_id] = 1
        return speaker_id

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------

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
            # centroid 均已 L2 归一化，点积即余弦相似度。
            similarities[row_idx] = np.matmul(
                centroid_matrix, observation.embedding
            ).astype(np.float32, copy=False)

        # Hungarian 一对一联合分配：同一 chunk 的不同 local slot 不会撞同一个
        # global speaker（隐式 cannot-link）；observation 多于 centroid 时，
        # 超出的行保持未匹配，后续走 new / 放弃判定。
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
        """完成一个 chunk 的 local->global 分配与 centroid 更新。"""

        debug_info: ChunkDebugInfo = {
            "num_centroids_before": len(self.centroids),
            "num_centroids_after": len(self.centroids),
            "local_assignments": [],
            "new_speakers": [],
            "updated_speakers": [],
            "skipped_updates": [],
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

        # matched：相似度达到主阈值，直接沿用该 global speaker。
        if matched_speaker is not None and similarity >= config.global_match_threshold:
            assigned_speaker = matched_speaker
            decision = "matched"
        # new：相似度低于建簇阈值且时长足够，新建全局 speaker（立即生效）。
        elif (
            len(self.centroids) < config.max_speakers
            and (matched_speaker is None or similarity < config.new_speaker_threshold)
            and (observation.duration >= config.min_segment_duration_for_new_speaker)
        ):
            assigned_speaker = self._create_speaker(observation)
            decision = "new"
        # fallback：介于两阈值之间（或 speaker 数已满 / 时长不足）时保守沿用，
        # 不更新 centroid，避免低置信观测污染身份。
        elif matched_speaker is not None:
            assigned_speaker = matched_speaker
            decision = "fallback"
        # 无匹配且不满足建簇条件：放弃该 local slot，其帧不输出。
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


__all__ = ["ChunkSpeakerClusterer"]
