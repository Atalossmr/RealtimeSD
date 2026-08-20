"""小样本簇强制合并后处理 + streaming 输出的 refined 级（两级分离）。

streaming 输出分两级：

- raw 级（`<stem>.raw.rttm`）：AppendOnlyRTTMWriter 的 append-only 输出，
  每行写出即最终，流式期间的 merge 不影响已写出行；
- refined 级（`<stem>.refined.rttm`）：由 RefinedRTTMWriter 维护，读取
  raw 文件与 assigner 当前合并状态整体重生成——流式期间每次 merge 事件
  动态刷新（修正 merge 前写出的旧身份行），结束时（final）再叠加小样本
  强制合并。refined 是面向下游/展示的最终输出。

小样本强制合并语义（ahc finalize 与 streaming refined 共用）：聚类结束后，
把总发声时长低于 `min_duration` 的簇/speaker 整体并入质心余弦相似度最高
的达标簇；最高相似度仍低于 `min_similarity` 时保留原身份不并（避免把
"说得少但确实独立"的 speaker 错并到别人头上）。ahc 后端标签尚未写出，
直接在 finalize 内重映射。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils import l2_normalize


logger = logging.getLogger(__name__)


def compute_merge_map(
    centroids: dict[int, np.ndarray],
    durations: dict[int, float],
    min_duration: float,
    min_similarity: float,
) -> dict[int, int]:
    """计算小样本簇的强制合并映射（absorbed id -> survivor id）。

    centroids 需已 L2 归一化；durations 为各簇总发声时长（秒），缺失按 0 计。
    时长 < min_duration 的簇为 absorbed 候选，其余为可并入的 target；
    无 target（全部簇都不达标）时不做任何合并。
    """

    small = sorted(
        sid for sid in centroids if durations.get(sid, 0.0) < min_duration
    )
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
                "[post-merge] cluster %d kept (duration=%.2fs, best similarity "
                "%.3f < %.3f)",
                sid,
                durations.get(sid, 0.0),
                best_similarity,
                min_similarity,
            )
            continue
        survivor = targets[best_idx]
        merge_map[sid] = survivor
        logger.info(
            "[post-merge] cluster %d -> %d (duration=%.2fs < %.2fs, "
            "similarity=%.3f)",
            sid,
            survivor,
            durations.get(sid, 0.0),
            min_duration,
            best_similarity,
        )
    return merge_map


def cluster_stats(
    labels: np.ndarray,
    observations: list,
    embeddings: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """按簇标签汇总质心（均值重归一化）与总发声时长。"""

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

_ID_MAP_PREFIX = "#   "


def _resolve_merged(global_id: int, merged_into: dict[int, int]) -> int:
    """沿 merged_into 链解析到最终幸存 id（幸存 id 之后也可能再被并）。"""

    seen: set[int] = set()
    while global_id in merged_into and global_id not in seen:
        seen.add(global_id)
        global_id = merged_into[global_id]
    return global_id


def write_refined_rttm(
    src_path: str,
    dst_path: str,
    centroids: dict[int, np.ndarray],
    merged_into: dict[int, int],
    min_duration: float,
    min_similarity: float,
    global_to_output: Optional[dict[int, int]] = None,
) -> dict[str, object]:
    """读取 raw RTTM，按当前合并状态重生成 refined RTTM。

    raw 文件不动（append-only 不变量）：先沿 merged_into 链修正流式期间被
    merge 的旧身份行，再叠加小样本强制合并（min_duration=0 时跳过）。
    speaker 重编号沿用 raw RTTM 的输出编号（被并 speaker 的行改挂幸存者的
    输出编号），末尾附 refined 的 id 映射表。

    global_to_output：raw writer 的实时 global->输出编号映射；缺省时从
    raw 文件末尾的 # id 映射注释解析（仅 finalize 后可用）。
    写出为临时文件后原子替换，避免读者看到半更新状态。

    返回 {"merge_map": 小样本合并映射, "durations": 幸存 global id -> 总时长,
    "global_to_output": 输出编号映射}，供 RefinedRTTMWriter 写 speaker 状态。
    """

    speaker_lines: list[list[str]] = []
    if global_to_output is None:
        global_to_output = {}
        with open(src_path, "r", encoding="utf-8") as file_obj:
            for raw in file_obj:
                line = raw.strip()
                if line.startswith("SPEAKER"):
                    speaker_lines.append(line.split())
                elif line.startswith(_ID_MAP_PREFIX) and "->" in line:
                    left, _, right = line[len(_ID_MAP_PREFIX):].partition("->")
                    global_to_output[int(left.strip())] = int(right.strip())
    else:
        with open(src_path, "r", encoding="utf-8") as file_obj:
            for raw in file_obj:
                line = raw.strip()
                if line.startswith("SPEAKER"):
                    speaker_lines.append(line.split())

    output_to_global = {output: gid for gid, output in global_to_output.items()}

    def final_survivor(global_id: int) -> int:
        # 先解流式期间的 merge 链，再叠小样本合并映射。
        resolved = _resolve_merged(global_id, merged_into)
        return merge_map.get(resolved, resolved)

    # 时长按"最终幸存 global id"累计（流式期间被 merge 的先归并到幸存者）。
    durations: dict[int, float] = {}
    for parts in speaker_lines:
        output_id = int(parts[7])
        global_id = output_to_global.get(output_id)
        if global_id is None:
            continue
        survivor = _resolve_merged(global_id, merged_into)
        durations[survivor] = durations.get(survivor, 0.0) + float(parts[4])

    alive_centroids = {
        sid: centroid for sid, centroid in centroids.items() if sid in durations
    }
    merge_map = compute_merge_map(
        alive_centroids, durations, min_duration, min_similarity
    )

    uri = Path(src_path).name.split(".", 1)[0]

    # 幸存 id 从未写出过 raw 行（如新建当 chunk 即被并、其语音又落在 commit
    # 区外）时不在输出编号映射中：为其分配未占用的输出编号，保证 refined 的
    # 行与末尾映射表一致（raw writer 后续给该 id 分配编号时从同一计数继续，
    # 通常恰好对齐；不一致也只是下次刷新重编号，单份文件内始终自洽）。
    next_fallback_id = max(global_to_output.values(), default=-1) + 1

    def output_id_of(global_id: int) -> int:
        nonlocal next_fallback_id
        output = global_to_output.get(global_id)
        if output is None:
            output = next_fallback_id
            next_fallback_id += 1
            global_to_output[global_id] = output
        return output

    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(
            f"# refined RTTM for {uri} (post-merge: "
            f"min_duration={min_duration}, min_similarity={min_similarity})\n"
        )
        for parts in speaker_lines:
            output_id = int(parts[7])
            global_id = output_to_global.get(output_id)
            survivor = final_survivor(global_id) if global_id is not None else None
            if survivor is not None:
                parts[7] = str(output_id_of(survivor))
            file_obj.write(" ".join(parts) + "\n")
        file_obj.write("# speaker_id_map: global_id -> rttm_speaker (refined)\n")
        survivors = sorted(
            {final_survivor(global_id) for global_id in global_to_output}
        )
        for survivor in survivors:
            file_obj.write(f"#   {survivor} -> {output_id_of(survivor)}\n")
    os.replace(tmp_path, dst_path)

    logger.info(
        "[refined] wrote refined RTTM %s (%d speaker(s) post-merged)",
        dst_path,
        len(merge_map),
    )
    return {
        "merge_map": merge_map,
        "durations": durations,
        "global_to_output": global_to_output,
    }


def write_speaker_status(
    status_path: str,
    *,
    uri: str,
    durations: dict[int, float],
    merged_into: dict[int, int],
    merge_map: dict[int, int],
    global_to_output: dict[int, int],
    min_duration: float,
    final: bool,
) -> None:
    """落盘 `{uri}.speakers.json`：refined 级的 speaker 状态 sidecar。

    供 viewer 等下游实时消费（与 refined RTTM 同步原子更新）：

    - speakers：每个出现过的 global speaker 的总发声时长、uncertain 标记
      （开启小样本阈值且当前时长未达标；流式中途会随时长累积自动解除）、
      最终归属（merged_into，含流式 merge 链与 post-merge 两级合并）；
    - merge_events：全部合并事件（kind 区分流式 merge 与 EOF post-merge）。
    """

    speakers = []
    for global_id in sorted(set(global_to_output) | set(merged_into)):
        resolved = _resolve_merged(global_id, merged_into)
        survivor = merge_map.get(resolved, resolved)
        merged = survivor != global_id
        # durations 按幸存者累计；被并 speaker 自身的时长已并入幸存者，置 null
        # 避免误显示为幸存者时长。
        duration = None if merged else round(float(durations.get(global_id, 0.0)), 3)
        speakers.append(
            {
                "id": int(global_id),
                "output_id": global_to_output.get(global_id),
                "duration": duration,
                "uncertain": bool(
                    not merged and min_duration > 0 and (duration or 0.0) < min_duration
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
        "post_merge_min_speech_duration": float(min_duration),
        "speakers": speakers,
        "merge_events": merge_events,
    }
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False)
    os.replace(tmp_path, status_path)


class RefinedRTTMWriter:
    """streaming 输出的第二级：raw RTTM + 当前合并状态 -> refined RTTM。

    流式期间每当发生 merge 事件即整体重生成（修正 merge 前写出的旧身份行，
    即"动态处理合并事件"）；final=True 时额外叠加小样本强制合并
    （post_merge_min_speech_duration > 0 才生效）。raw 文件全程不动。
    每次刷新同步更新 `{uri}.speakers.json` sidecar（speaker 时长 / uncertain
    标记 / 合并事件），供前端实时展示。
    """

    def __init__(
        self,
        raw_path: str,
        refined_path: str,
        writer,
        assigner,
        min_duration: float,
        min_similarity: float,
    ):
        self.raw_path = raw_path
        self.refined_path = refined_path
        # <stem>.refined.rttm -> <stem>.speakers.json
        self.status_path = refined_path[: -len(".refined.rttm")] + ".speakers.json"
        self._writer = writer
        self._assigner = assigner
        self.min_duration = float(min_duration)
        self.min_similarity = float(min_similarity)

    def refresh(self, final: bool = False) -> None:
        """按当前合并状态重生成 refined RTTM 与 speaker 状态 sidecar。"""

        result = write_refined_rttm(
            self.raw_path,
            self.refined_path,
            centroids=getattr(self._assigner, "centroids", {}),
            merged_into=getattr(self._assigner, "merged_into", {}),
            # 流式中途不应用小样本合并（时长尚未累积够会误并），仅 final 时启用。
            min_duration=self.min_duration if final else 0.0,
            min_similarity=self.min_similarity,
            global_to_output=self._writer.output_id_map,
        )
        write_speaker_status(
            self.status_path,
            uri=Path(self.raw_path).name.split(".", 1)[0],
            durations=result["durations"],
            merged_into=getattr(self._assigner, "merged_into", {}),
            merge_map=result["merge_map"],
            global_to_output=result["global_to_output"],
            min_duration=self.min_duration,
            final=final,
        )


__all__ = [
    "compute_merge_map",
    "cluster_stats",
    "write_refined_rttm",
    "write_speaker_status",
    "RefinedRTTMWriter",
]
