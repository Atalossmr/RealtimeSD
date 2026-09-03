"""小样本簇强制合并后处理 + 两级输出（raw / refined）的 refined 级。

两个聚类后端的输出都分两级：

- raw 级：AppendOnlyRTTMWriter 的 append-only 输出，每行写出即最终
  （两后端统一为 `<stem>.raw.rttm`）；
- refined 级（`<stem>.refined.rttm`）：由 RefinedRTTMWriter 维护，读取
  raw 文件与 clusterer 当前合并状态整体重生成——streaming 在流式期间发生
  merge 事件时即时刷新（修正 merge 前写出的旧身份行），无 merge 时按固定
  chunk 间隔兜底刷新（时机由 cluster/runner.py 控制，避免逐 chunk 全量
  重写的 O(n²) I/O）；ahc 无流式 merge、仅在 EOF 刷新；两者都在结束时
  （final）叠加小样本强制合并。refined 是面向下游/展示的最终输出。

小样本强制合并语义（两后端共用，均在 refined 级生效）：聚类结束后，
把样本计数低于 `min_count` 的簇/speaker 整体并入质心余弦相似度最高
的达标簇；最高相似度仍低于 `min_similarity` 时保留原身份不并（避免把
"说得少但确实独立"的 speaker 错并到别人头上）。计数口径：streaming 为
有效 centroid 更新次数（counts），ahc 为簇内 observation 数。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from common.utils import l2_normalize

from .base import BaseChunkClusterer, resolve_merged


logger = logging.getLogger(__name__)


def compute_merge_map(
    centroids: dict[int, np.ndarray],
    counts: dict[int, int],
    min_count: int,
    min_similarity: float,
) -> dict[int, int]:
    """计算小样本簇的强制合并映射（absorbed id -> survivor id）。

    centroids 需已 L2 归一化；counts 为各簇样本计数，缺失按 0 计。
    计数 < min_count 的簇为 absorbed 候选，其余为可并入的 target；
    无 target（全部簇都不达标）时不做任何合并。absorbed 候选并入质心
    余弦相似度最高的 target，最高相似度仍低于 min_similarity 时保留原身份。

    Args:
        centroids: L2 归一化的簇质心（speaker id -> centroid）。
        counts: 各簇样本计数。
        min_count: 小样本阈值，计数低于该值的簇参与被并判定。
        min_similarity: 允许并入的最低余弦相似度。

    Returns:
        合并映射（absorbed id -> survivor id）；无可合并对时为空 dict。
    """

    small = sorted(sid for sid in centroids if counts.get(sid, 0) < min_count)
    small_set = set(small)
    targets = sorted(sid for sid in centroids if sid not in small_set)
    if not small or not targets:
        return {}

    target_matrix = np.stack([centroids[tid] for tid in targets])
    merge_map: dict[int, int] = {}
    for sid in small:
        # centroid 均已 L2 归一化，点积即余弦相似度。
        similarities = target_matrix @ centroids[sid]
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        if best_similarity < min_similarity:
            logger.info(
                "[post-merge] cluster %d kept (count=%d, best similarity "
                "%.3f < %.3f)",
                sid,
                counts.get(sid, 0),
                best_similarity,
                min_similarity,
            )
            continue
        survivor = targets[best_idx]
        merge_map[sid] = survivor
        logger.info(
            "[post-merge] cluster %d -> %d (count=%d < %d, similarity=%.3f)",
            sid,
            survivor,
            counts.get(sid, 0),
            min_count,
            best_similarity,
        )
    return merge_map


def cluster_stats(
    labels: np.ndarray,
    observations: list,
    embeddings: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """按簇标签汇总质心（均值重归一化）与总发声时长。

    Args:
        labels: 每条 observation 的簇标签。
        observations: 与 labels 一一对应的观测（读取其 duration）。
        embeddings: 与 labels 一一对应的 embedding。

    Returns:
        (centroids, durations)：簇标签 -> L2 归一化质心，簇标签 -> 总发声时长。
    """

    unique_labels = sorted(int(label) for label in set(labels.tolist()))
    dim = int(embeddings.shape[1])
    sums = {label: np.zeros(dim, dtype=np.float64) for label in unique_labels}
    durations = {label: 0.0 for label in unique_labels}
    for label, observation, embedding in zip(labels, observations, embeddings):
        label = int(label)
        sums[label] += embedding.astype(np.float64, copy=False)
        durations[label] += float(observation.duration)
    centroids = {
        label: l2_normalize(total.astype(np.float32, copy=False))
        for label, total in sums.items()
    }
    return centroids, durations


# ----------------------------------------------------------------------
# streaming refined 级：读取 raw RTTM，按当前合并状态整体重生成
# ----------------------------------------------------------------------


def _atomic_replace(tmp_path: str, dst_path: str) -> None:
    """os.replace 的 Windows 容错封装（退避重试短暂占用）。

    目标文件被杀毒软件/索引器等短暂占用时会抛 PermissionError
    [WinError 5]，退避重试若干次再放弃。

    Args:
        tmp_path: 已写好的临时文件路径。
        dst_path: 替换目标路径。

    Raises:
        PermissionError: 重试耗尽后目标仍被占用。
    """

    attempts, delay = 10, 0.2
    for attempt in range(attempts):
        try:
            os.replace(tmp_path, dst_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def write_refined_rttm(
    src_path: str,
    dst_path: str,
    centroids: dict[int, np.ndarray],
    merged_into: dict[int, int],
    counts: dict[int, int],
    min_count: int,
    min_similarity: float,
) -> dict[str, object]:
    """读取 raw RTTM，按当前合并状态重生成 refined RTTM。

    raw 文件不动（append-only 不变量）：先沿 merged_into 链修正流式期间被
    merge 的旧身份行，再叠加小样本强制合并（min_count=0 时跳过，按 counts
    判定）。RTTM 行内的 speaker 字段即 global id，被并 speaker 的行改挂
    幸存者的 global id。写出为临时文件后原子替换，避免读者看到半更新状态。

    Args:
        src_path: raw RTTM 路径（只读）。
        dst_path: refined RTTM 输出路径（临时文件 + 原子替换）。
        centroids: L2 归一化的簇质心（只保留在 raw 中实际出现的 speaker）。
        merged_into: 流式 merge 链（absorbed id -> survivor id）。
        counts: 各 speaker 的样本计数（小样本判定用）。
        min_count: 小样本阈值；0 表示跳过小样本合并。
        min_similarity: 允许并入的最低余弦相似度。

    Returns:
        {"merge_map": 小样本合并映射, "durations": 幸存 global id -> 总时长}，
        供 RefinedRTTMWriter 写 speaker 状态（durations 仅用于展示，不参与判定）。
    """

    speaker_lines: list[list[str]] = []
    with open(src_path, "r", encoding="utf-8") as file_obj:
        for raw in file_obj:
            line = raw.strip()
            if line.startswith("SPEAKER"):
                speaker_lines.append(line.split())

    def final_survivor(global_id: int) -> int:
        # 先解流式期间的 merge 链，再叠小样本合并映射。
        resolved = resolve_merged(global_id, merged_into)
        return merge_map.get(resolved, resolved)

    # 时长按"最终幸存 global id"累计（流式期间被 merge 的先归并到幸存者）。
    durations: dict[int, float] = {}
    for parts in speaker_lines:
        survivor = resolve_merged(int(parts[7]), merged_into)
        durations[survivor] = durations.get(survivor, 0.0) + float(parts[4])

    alive_centroids = {
        sid: centroid for sid, centroid in centroids.items() if sid in durations
    }
    merge_map = compute_merge_map(
        alive_centroids, counts, min_count, min_similarity
    )

    uri = Path(src_path).name.split(".", 1)[0]

    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(
            f"# refined RTTM for {uri} (post-merge: "
            f"min_count={min_count}, min_similarity={min_similarity})\n"
        )
        for parts in speaker_lines:
            parts[7] = str(final_survivor(int(parts[7])))
            file_obj.write(" ".join(parts) + "\n")
    _atomic_replace(tmp_path, dst_path)

    logger.info(
        "[refined] wrote refined RTTM %s (%d speaker(s) post-merged)",
        dst_path,
        len(merge_map),
    )
    return {"merge_map": merge_map, "durations": durations}


def write_speaker_status(
    status_path: str,
    *,
    uri: str,
    durations: dict[int, float],
    counts: dict[int, int],
    merged_into: dict[int, int],
    merge_map: dict[int, int],
    min_count: int,
    final: bool,
) -> None:
    """落盘 `{uri}.speakers.json`：refined 级的 speaker 状态 sidecar。

    供 viewer 等下游实时消费（与 refined RTTM 同步原子更新）：

    - speakers：每个出现过的 global speaker 的总发声时长（展示用）、样本
      计数、uncertain 标记（开启小样本阈值且当前计数未达标；流式中途会随
      计数累积自动解除）、最终归属（merged_into，含流式 merge 链与
      post-merge 两级合并）；
    - merge_events：全部合并事件（kind 区分流式 merge 与 EOF post-merge）。

    Args:
        status_path: sidecar 输出路径（临时文件 + 原子替换）。
        uri: 音频标识（写入 payload 的 uri 字段）。
        durations: 幸存 global id -> 总发声时长（被并 speaker 的数值已并入
            幸存者，自身置 null）。
        counts: 各 speaker 的样本计数（uncertain 判定用）。
        merged_into: 流式 merge 链。
        merge_map: 小样本强制合并映射（post-merge）。
        min_count: 小样本阈值（写入 payload，并用于 uncertain 判定）。
        final: 是否为 EOF 最终刷新（写入 payload 的 final 字段）。
    """

    speakers = []
    for global_id in sorted(set(durations) | set(merged_into)):
        resolved = resolve_merged(global_id, merged_into)
        survivor = merge_map.get(resolved, resolved)
        merged = survivor != global_id
        # durations/counts 按幸存者累计；被并 speaker 自身的数值已并入幸存者，
        # 置 null 避免误显示为幸存者数值。
        duration = None if merged else round(float(durations.get(global_id, 0.0)), 3)
        count = None if merged else int(counts.get(global_id, 0))
        speakers.append(
            {
                "id": int(global_id),
                "duration": duration,
                "count": count,
                "uncertain": bool(
                    not merged and min_count > 0 and (count or 0) < min_count
                ),
                "merged_into": int(survivor) if merged else None,
            }
        )
    merge_events = [
        {"absorbed": int(absorbed), "survivor": int(survivor), "kind": "merge"}
        for absorbed, survivor in sorted(merged_into.items())
    ] + [
        {"absorbed": int(absorbed), "survivor": int(survivor), "kind": "post_merge"}
        for absorbed, survivor in sorted(merge_map.items())
    ]
    payload = {
        "uri": uri,
        "final": bool(final),
        "post_merge_min_count": int(min_count),
        "speakers": speakers,
        "merge_events": merge_events,
    }
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False)
    _atomic_replace(tmp_path, status_path)


class RefinedRTTMWriter:
    """输出的第二级：raw RTTM + 当前合并状态 -> refined RTTM（两后端通用）。

    streaming：流式期间发生 merge 事件时整体重生成（修正 merge 前写出
    的旧身份行，即"动态处理合并事件"），无 merge 时按固定 chunk 间隔
    兜底刷新（刷新时机由 cluster/runner.py 控制）；ahc（deferred）：无
    流式 merge，仅在 EOF 重放完成后刷新。final=True 时叠加小样本强制合并
    （post_merge_min_count > 0 才生效，按 clusterer 的 counts 判定）。
    raw 文件全程不动。每次刷新同步更新 `{uri}.speakers.json` sidecar
    （speaker 时长 / 计数 / uncertain 标记 / 合并事件），供前端实时展示。

    Args:
        raw_path: raw 级 RTTM 路径（只读）。
        refined_path: refined 级 RTTM 输出路径；sidecar 路径由其推导
            （<stem>.refined.rttm -> <stem>.speakers.json）。
        clusterer: 聚类后端，读取其 centroids / counts / merged_into。
        min_count: 小样本阈值（仅 final 刷新时生效，中途刷新按 0 处理）。
        min_similarity: 允许并入的最低余弦相似度。
    """

    def __init__(
        self,
        raw_path: str,
        refined_path: str,
        clusterer: BaseChunkClusterer,
        min_count: int,
        min_similarity: float,
    ):
        self.raw_path = raw_path
        self.refined_path = refined_path
        # <stem>.refined.rttm -> <stem>.speakers.json
        self.status_path = refined_path[: -len(".refined.rttm")] + ".speakers.json"
        self._clusterer = clusterer
        self.min_count = int(min_count)
        self.min_similarity = float(min_similarity)

    def refresh(self, final: bool = False) -> None:
        """按当前合并状态重生成 refined RTTM 与 speaker 状态 sidecar。

        Args:
            final: True 时叠加小样本强制合并（post-merge），并在 sidecar 中
                标记 final；流式中途刷新应传 False（计数尚未累积够会误并）。
        """

        result = write_refined_rttm(
            self.raw_path,
            self.refined_path,
            centroids=self._clusterer.centroids,
            merged_into=self._clusterer.merged_into,
            counts=self._clusterer.counts,
            # 流式中途不应用小样本合并（计数尚未累积够会误并），仅 final 时启用。
            min_count=self.min_count if final else 0,
            min_similarity=self.min_similarity,
        )
        write_speaker_status(
            self.status_path,
            uri=Path(self.raw_path).name.split(".", 1)[0],
            durations=result["durations"],
            counts=self._clusterer.counts,
            merged_into=self._clusterer.merged_into,
            merge_map=result["merge_map"],
            min_count=self.min_count,
            final=final,
        )


__all__ = [
    "compute_merge_map",
    "cluster_stats",
    "write_refined_rttm",
    "write_speaker_status",
    "RefinedRTTMWriter",
]
