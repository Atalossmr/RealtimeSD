"""fastdiar 风格的 chunk 级在线聚类后端（中位数质心 + 连通分量并簇）。

仿 `fastdiar` 的 `OnlineClustering`（帧级在线聚类）思路，适配到本项目的
chunk/observation 级接口。与 fastdiar 机制的对应关系：

- confident 帧（自相似门控）才参与聚类/更新质心
  → observation 与所指派质心的相似度 ≥ ``fastdiar_confidence_threshold``
  才是 anchor，**只有 anchor 更新质心**；``allow_centroid_update=False``
  （overlap_fallback）的 observation 永远不是 anchor；
- 质心 = 逐维中位数（fastdiar 用 _RunningMedian 双堆）
  → 每 speaker 存 anchor embedding 列表（L2 归一化后存入），质心 = 逐维
  ``np.median`` 后 L2 归一化；列表只追加，带惰性缓存（脏标记）——observation
  规模远小于帧级，不需要双堆；
- sim ≥ clust_th 归最近簇，否则开新簇
  → 单阈值 ``fastdiar_assign_threshold``，贪心取最近质心（不用 Hungarian）；
- 新簇只能由 confident 帧创建
  → 建簇门槛：``allow_centroid_update`` 且 ``duration >=
  min_segment_duration_for_new_speaker`` 且未达 ``max_speakers``；
- ``_force_assign`` 无阈值兜底
  → 不能建簇的未匹配 observation 不丢弃，强制指派最近质心（decision
  ="forced"，不更新质心）；边界：无任何质心且不能建簇时直接建簇兜底
  （保证首个 observation 不丢）；
- merge_th >> clust_th，新簇出现时触发，连通分量整组并簇
  → 每次新建 speaker 后触发一次：质心 Gram 矩阵中相似度 >
  ``fastdiar_merge_threshold`` 的配对建边，
  ``scipy.sparse.csgraph.connected_components`` 取连通分量，≥2 成员的
  分量整组并簇（survivor = anchor 数最多者，并列取 id 小者；新质心 = 并集
  的中位数；``merged_into`` 记录全部被吸收者）。

与 fastdiar 的适配差异：

- **无 960ms 自我纠正窗口**：流式契约要求 ``assign_chunk`` 立即返回最终
  local->global 映射，observation 判定一次定案；
- **置信度为段级代理**：fastdiar 的 confidence 是帧级回看自相似
  （e_t·e_{t-delay}），本项目 observation 级没有帧序列上下文，用
  "observation 与所指派质心的相似度"作段级置信代理；
- 不做 overlap 输出（fastdiar 的 overlap 检测与本管线 segmentation 重叠
  处理重复）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from diarization.config import ChunkPipelineConfig
from diarization.contracts import ChunkDebugInfo, ChunkObservation
from diarization.utils import l2_normalize

from ..base import BaseChunkClusterer


logger = logging.getLogger(__name__)


class FastdiarChunkClusterer(BaseChunkClusterer):
    """fastdiar 风格的 chunk 粒度在线聚类器。

    贪心指派（最近质心 argmax，不做 Hungarian 联合分配），matched/new/forced
    三分支判定；质心为 anchor embedding 的逐维中位数（惰性缓存）；每次新建
    speaker 后做一次连通分量整组并簇。
    """

    # 流式输出为 raw 级（append-only）；修正后的最终输出为 refined 级
    # （见 cluster/post_process.py 的 RefinedRTTMWriter）。
    output_tag = "raw"

    def __init__(self, config: ChunkPipelineConfig):
        self.config = config

        # speaker id -> 该 speaker 的 anchor embedding 列表（L2 归一化，只追加）。
        self.anchors: dict[int, list[np.ndarray]] = {}
        # speaker id -> 中位数质心缓存；anchor 追加后置脏，下次读取时重算。
        self._centroid_cache: dict[int, np.ndarray] = {}
        self._dirty: set[int] = set()
        self.counts: dict[int, int] = {}
        # 被合并 speaker id -> 幸存 speaker id（仅用于追溯，不再参与分配）。
        self.merged_into: dict[int, int] = {}

        self.next_speaker_id = 0

    @property
    def centroids(self) -> dict[int, np.ndarray]:
        """当前全部存活 speaker 的 L2 归一化中位数质心（惰性重算）。"""

        return {sid: self._centroid(sid) for sid in sorted(self.anchors)}

    # ------------------------------------------------------------------
    # 基本状态查询
    # ------------------------------------------------------------------

    def current_speaker_ids(self) -> set[int]:
        """返回当前维护的全部 global speaker id。"""

        return set(self.anchors.keys())

    def current_global_speakers(self) -> list[dict[str, int]]:
        """返回当前全局说话人摘要列表。"""

        speakers: list[dict[str, int]] = []
        for speaker_id in sorted(self.anchors):
            centroid = self._centroid(speaker_id)
            speakers.append(
                {
                    "speaker": int(speaker_id),
                    "count": int(self.counts.get(speaker_id, 0)),
                    "dim": int(centroid.shape[0]),
                }
            )
        return speakers

    # ------------------------------------------------------------------
    # 质心（逐维中位数，惰性缓存）
    # ------------------------------------------------------------------

    def _centroid(self, speaker_id: int) -> np.ndarray:
        """返回 speaker 的 L2 归一化中位数质心；缓存脏时重算。"""

        if speaker_id in self._dirty or speaker_id not in self._centroid_cache:
            matrix = np.stack(self.anchors[speaker_id])
            median = np.median(matrix, axis=0)
            self._centroid_cache[speaker_id] = l2_normalize(
                median.astype(np.float32, copy=False)
            )
            self._dirty.discard(speaker_id)
        return self._centroid_cache[speaker_id]

    def _nearest(self, embedding: np.ndarray) -> tuple[Optional[int], float]:
        """贪心取与 embedding 余弦相似度最高的质心。

        Args:
            embedding: L2 归一化的 observation embedding。

        Returns:
            (speaker id, 余弦相似度)；无存活质心时为 (None, -1.0)。
        """

        if not self.anchors:
            return None, -1.0
        global_ids = sorted(self.anchors.keys())
        centroid_matrix = np.stack([self._centroid(sid) for sid in global_ids])
        # 质心均已 L2 归一化，点积即余弦相似度。
        similarities = centroid_matrix @ embedding
        best = int(np.argmax(similarities))
        return global_ids[best], float(similarities[best])

    def _create_speaker(self, embedding: np.ndarray) -> int:
        """以 observation 的 embedding 为首个 anchor 新建全局说话人。

        Args:
            embedding: L2 归一化的 observation embedding。

        Returns:
            新建的 global speaker id。
        """

        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        self.anchors[speaker_id] = [embedding]
        self.counts[speaker_id] = 1
        self._mark_dirty(speaker_id)
        return speaker_id

    def _mark_dirty(self, speaker_id: int) -> None:
        """把 speaker 的质心缓存置脏。"""

        self._dirty.add(speaker_id)

    def _add_anchor(self, speaker_id: int, embedding: np.ndarray) -> None:
        """向 speaker 追加 anchor embedding 并置脏质心缓存。"""

        self.anchors[speaker_id].append(embedding)
        self.counts[speaker_id] = len(self.anchors[speaker_id])
        self._mark_dirty(speaker_id)

    # ------------------------------------------------------------------
    # merge：连通分量整组并簇（新簇出现时触发）
    # ------------------------------------------------------------------

    def _merge_connected(
        self,
        local_to_global: dict[int, int],
        debug_info: ChunkDebugInfo,
    ) -> None:
        """质心相似度 > merge 阈值的连通分量整组并簇。

        survivor = 分量内 anchor 数最多者（并列取 id 小者）；新质心 = 分量内
        全部 anchor 并集的中位数；被并者从质心集中移除并记入 merged_into。
        raw RTTM 已写出行不受影响（历史行由 refined 级修正）；本 chunk 尚未
        写出的分配与调试记录改挂到幸存 id。

        Args:
            local_to_global: 本 chunk 尚未写出的分配，指向被并者的条目
                会被原地改挂到幸存 id。
            debug_info: 本 chunk 的调试记录，local_assignments 中指向被并者
                的条目同样改挂幸存 id。
        """

        if len(self.anchors) < 2:
            return

        global_ids = sorted(self.anchors.keys())
        centroid_matrix = np.stack([self._centroid(sid) for sid in global_ids])
        # 质心均已 L2 归一化，Gram 矩阵即两两余弦相似度。
        similarities = centroid_matrix @ centroid_matrix.T
        rows, cols = np.where(
            np.tril(similarities, k=-1) > self.config.fastdiar_merge_threshold
        )
        if len(rows) == 0:
            return

        n = len(global_ids)
        adjacency = csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        )
        _, labels = connected_components(adjacency, directed=False)

        groups: dict[int, list[int]] = {}
        for speaker_id, component in zip(global_ids, labels.tolist()):
            groups.setdefault(component, []).append(speaker_id)

        for group in groups.values():
            if len(group) < 2:
                continue
            # survivor = anchor 数最多者，并列取 id 小者。
            survivor = min(group, key=lambda sid: (-len(self.anchors[sid]), sid))
            absorbed = [sid for sid in group if sid != survivor]
            for sid in absorbed:
                self.anchors[survivor].extend(self.anchors[sid])
                del self.anchors[sid]
                del self.counts[sid]
                self._centroid_cache.pop(sid, None)
                self._dirty.discard(sid)
                self.merged_into[sid] = survivor
            self.counts[survivor] = len(self.anchors[survivor])
            self._mark_dirty(survivor)

            absorbed_set = set(absorbed)
            # 本 chunk 的帧尚未写出：把指向被合并者的分配与调试记录改挂到幸存 id。
            for local_idx, global_id in list(local_to_global.items()):
                if global_id in absorbed_set:
                    local_to_global[local_idx] = survivor
            for record in debug_info["local_assignments"]:
                if record["global"] in absorbed_set:
                    record["global"] = survivor

            logger.info(
                "[merge] 连通分量并簇：%s -> %d (component=%s, anchors=%d)",
                absorbed,
                survivor,
                group,
                len(self.anchors[survivor]),
            )

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------

    def assign_chunk(
        self,
        observations: list[ChunkObservation],
    ) -> tuple[dict[int, int], ChunkDebugInfo]:
        """完成一个 chunk 的 local->global 分配与质心维护。

        逐 observation（仅 embedding 非 None 者）做 matched/new/forced 判定；
        每次新建 speaker 后触发一次连通分量并簇（可能影响本 chunk 后续
        observation 的匹配对象）。

        Args:
            observations: 本 chunk 的一组 local track 观测。

        Returns:
            (local_to_global, debug_info)：local slot -> 最终 global id
            （embedding 缺失的 slot 不出现），以及本 chunk 的调试信息。
        """

        debug_info: ChunkDebugInfo = {
            "num_centroids_before": len(self.anchors),
            "num_centroids_after": len(self.anchors),
            "local_assignments": [],
            "new_speakers": [],
            "updated_speakers": [],
            "skipped_updates": [],
            "global_speakers": [],
        }
        local_to_global: dict[int, int] = {}

        for observation in observations:
            if observation.embedding is None:
                continue
            self._resolve_observation(observation, local_to_global, debug_info)

        debug_info["num_centroids_after"] = len(self.anchors)
        debug_info["global_speakers"] = self.current_global_speakers()
        return local_to_global, debug_info

    def _resolve_observation(
        self,
        observation: ChunkObservation,
        local_to_global: dict[int, int],
        debug_info: ChunkDebugInfo,
    ) -> None:
        """对单个 observation 做 matched/new/forced 判定并更新状态。

        判定规则（贪心取最近质心）：

        - matched：相似度 ≥ ``fastdiar_assign_threshold`` 归最近簇；相似度再
          达到 ``fastdiar_confidence_threshold`` 且允许更新时作为 anchor
          追加（更新质心），否则不更新；
        - new：不满足 matched 且满足建簇条件（允许 centroid 更新、时长
          足够、speaker 数未满）时新建 speaker，随后触发一次连通分量并簇；
        - forced：其余情况强制指派最近质心（不更新质心）；无任何质心且不能
          建簇时直接建簇兜底（保证首个 observation 不丢）。

        Args:
            observation: 待判定的观测（embedding 非 None）。
            local_to_global: 本 chunk 的分配结果（原地追加）。
            debug_info: 本 chunk 的调试记录（原地追加）。
        """

        config = self.config
        local_idx = observation.local_idx
        embedding = l2_normalize(
            observation.embedding.astype(np.float32, copy=False)
        )
        matched_speaker, similarity = self._nearest(embedding)

        can_create = (
            observation.allow_centroid_update
            and observation.duration >= config.min_segment_duration_for_new_speaker
            and len(self.anchors) < config.max_speakers
        )

        is_anchor = False
        if matched_speaker is not None and similarity >= config.fastdiar_assign_threshold:
            assigned_speaker = matched_speaker
            decision = "matched"
            # anchor：只有高置信的 matched 观测才更新质心（对应 fastdiar 的
            # confident 帧门控）；overlap_fallback 观测永远不是 anchor。
            is_anchor = (
                similarity >= config.fastdiar_confidence_threshold
                and observation.allow_centroid_update
            )
        elif can_create:
            assigned_speaker = self._create_speaker(embedding)
            decision = "new"
            # 新簇出现时触发一次连通分量并簇（对应 fastdiar 的
            # new_speech_cluster 触发 _merge_speech_clusters）。
            self._merge_connected(local_to_global, debug_info)
        elif matched_speaker is not None:
            # forced：不能建簇的未匹配观测不丢弃，无阈值兜底指派最近质心。
            assigned_speaker = matched_speaker
            decision = "forced"
        else:
            # 边界兜底：无任何质心且不能建簇时直接建簇（首个 observation 不丢）。
            assigned_speaker = self._create_speaker(embedding)
            decision = "new"
            self._merge_connected(local_to_global, debug_info)

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

        if is_anchor:
            self._add_anchor(assigned_speaker, embedding)
            debug_info["updated_speakers"].append(
                {
                    "global": int(assigned_speaker),
                    "mode": "anchor",
                    "start": float(observation.start),
                    "end": float(observation.end),
                }
            )
            return

        reason = (
            "below_confidence" if decision == "matched" else "forced_assignment"
        )
        debug_info["skipped_updates"].append(
            {
                "global": int(assigned_speaker),
                "reason": reason,
                "selection_mode": observation.selection_mode,
                "start": float(observation.start),
                "end": float(observation.end),
            }
        )


__all__ = ["FastdiarChunkClusterer"]
