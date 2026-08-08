"""chunk 级全局 speaker 分配与 centroid 维护。

设计要点：

- 处理单元是一个 chunk 的一组 local track observation，而不是 0.5s 滑窗；
- 纯流式：新建 speaker 立即成为永久身份，无试用期、无吸收，
  所有分配判定一次定案；
- 合并（merge）：每次加入新片段后尝试合并最相似的一对 speaker
  （count 小者并入大者），由 merge_threshold 控制；合并只影响后续分配，
  已写出的 RTTM 不改，被合并者从 centroid 集中移除、不再参与后续聚类；
- false split 可由 merge 事后修复，false glue 仍不可修复，因此阈值策略
  仍为"宁可 glue 不可 split"（new_speaker_threshold 应适当调高）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...config import ChunkPipelineConfig
from ..base import BaseChunkAssigner
from ...schema import ChunkDebugInfo, ChunkObservation
from ...utils import l2_normalize


logger = logging.getLogger(__name__)


class ChunkSpeakerClusterer(BaseChunkAssigner):
    """chunk 粒度的实时全局 speaker 分配器。

    每个 chunk 内用 Hungarian algorithm 做 local->global 联合分配，
    隐式提供 cannot-link 约束：同一 chunk 的不同 local slot 不会分配给同一 global speaker。
    """

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config

        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        # 被合并 speaker id -> 幸存 speaker id（仅用于追溯，不再参与分配）。
        self.merged_into: dict[int, int] = {}
        # speaker 诞生时的 chunk 序号（merge_protect_established 的"年龄"依据）。
        self.created_at: dict[int, int] = {}
        self._chunk_counter = 0

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

        # alpha = 1/(count+1)：等权平均的增量形式，观测越多单条影响越小，
        # centroid 随时间自然稳定下来。
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
        self.created_at[speaker_id] = self._chunk_counter
        return speaker_id

    # ------------------------------------------------------------------
    # merge：最相似的一对 speaker 合并（小并入大）
    # ------------------------------------------------------------------

    def _is_probationary(self, speaker_id: int) -> bool:
        """speaker 是否仍在缓刑期（年龄 ≤ new_speaker_hold_chunks 个 chunk）。"""

        probation = max(0, int(self.config.new_speaker_hold_chunks))
        return self._chunk_counter - self.created_at.get(speaker_id, 0) <= probation

    def _pick_merge_pair(
        self, global_ids: list[int], similarities: np.ndarray
    ) -> Optional[tuple[int, int, int, int, float]]:
        """选出最相似的一对可合并 speaker。

        返回 (survivor, absorbed, similarity)；无可合并对返回 None。
        merge_protect_established 开启时，被合并一方必须仍在缓刑期
        （已存活过缓冲期的 speaker 不允许被并掉）：恰好一方在缓刑期时该方为
        absorbed（不论 count），双方都在缓刑期时沿用"小并入大"。
        """

        n = len(global_ids)
        best: Optional[tuple[int, int, float]] = None
        for row_idx in range(n):
            for col_idx in range(row_idx + 1, n):
                similarity = float(similarities[row_idx, col_idx])
                if similarity < self.config.merge_threshold:
                    continue
                id_a, id_b = global_ids[row_idx], global_ids[col_idx]
                if self.config.merge_protect_established:
                    probationary_a = self._is_probationary(id_a)
                    probationary_b = self._is_probationary(id_b)
                    if not (probationary_a or probationary_b):
                        # 双方都已存活过缓冲期：禁止合并。
                        continue
                    # 恰好一方在缓刑期时该方必为 absorbed；方向确定后按
                    # (count, -id) 规则统一求 survivor/absorbed 的优先级。
                    if probationary_a and not probationary_b:
                        key_a, key_b = (0, 0), (1, 0)  # id_a 必为 absorbed
                    elif probationary_b and not probationary_a:
                        key_a, key_b = (1, 0), (0, 0)  # id_b 必为 absorbed
                    else:
                        key_a = (1, self.counts[id_a], -id_a)
                        key_b = (1, self.counts[id_b], -id_b)
                else:
                    key_a = (1, self.counts[id_a], -id_a)
                    key_b = (1, self.counts[id_b], -id_b)
                # key 大者为 survivor；缓刑语义下 key 首元素小者必为 absorbed。
                if key_a >= key_b:
                    survivor, absorbed = id_a, id_b
                else:
                    survivor, absorbed = id_b, id_a
                if best is None or similarity > best[2]:
                    best = (survivor, absorbed, similarity)
        if best is None:
            return None
        survivor, absorbed, similarity = best
        return survivor, absorbed, similarity

    def _try_merge_speakers(
        self,
        local_to_global: dict[int, int],
        debug_info: ChunkDebugInfo,
    ) -> None:
        """若最相似的一对可合并 centroid 相似度达到 merge_threshold，则合并（每次最多一对）。

        count 小者并入大者（count 相同保留 id 较小者），centroid 按 count
        加权平均后重归一化；被合并者从 centroid 集中移除，不再参与后续分配。
        已写出的 RTTM 不受影响；本 chunk 尚未写出的分配改挂到幸存 id。
        merge_protect_established 开启时，已存活过缓冲期
        （new_speaker_hold_chunks）的 speaker 不允许被合并。
        """

        if len(self.centroids) < 2:
            return

        global_ids = sorted(self.centroids.keys())
        centroid_matrix = np.stack([self.centroids[sid] for sid in global_ids])
        # centroid 均已 L2 归一化， Gram 矩阵即两两余弦相似度。
        similarities = centroid_matrix @ centroid_matrix.T
        picked = self._pick_merge_pair(global_ids, similarities)
        if picked is None:
            return
        survivor, absorbed, best_similarity = picked

        count_survivor = self.counts[survivor]
        count_absorbed = self.counts[absorbed]
        total = count_survivor + count_absorbed
        merged = (
            count_survivor * self.centroids[survivor]
            + count_absorbed * self.centroids[absorbed]
        ) / float(total)
        self.centroids[survivor] = l2_normalize(merged.astype(np.float32, copy=False))
        self.counts[survivor] = total
        del self.centroids[absorbed]
        del self.counts[absorbed]
        del self.created_at[absorbed]
        self.merged_into[absorbed] = survivor

        # 本 chunk 的帧尚未写出：把指向被合并者的分配与调试记录改挂到幸存 id。
        for local_idx, global_id in list(local_to_global.items()):
            if global_id == absorbed:
                local_to_global[local_idx] = survivor
        for record in debug_info["local_assignments"]:
            if record["global"] == absorbed:
                record["global"] = survivor

        logger.info(
            "[merge] speaker %d -> %d (similarity=%.3f, counts=%d+%d)",
            absorbed,
            survivor,
            best_similarity,
            count_absorbed,
            count_survivor,
        )

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

        self._chunk_counter += 1
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
                # 每加入一个新片段都尝试合并最相似的一对 speaker；
                # 合并可能影响本 chunk 后续 observation 的匹配对象。
                self._try_merge_speakers(local_to_global, debug_info)

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
        # assignment 基于本 chunk 开头的 centroid 集计算；若匹配对象在本 chunk
        # 内已被 merge 掉，改挂到幸存 id（相似度沿用旧值，与幸存 centroid 近似）。
        if matched_speaker is not None:
            matched_speaker = self.merged_into.get(matched_speaker, matched_speaker)

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
