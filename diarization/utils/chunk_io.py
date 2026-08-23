"""chunk artifacts 中间文件（<stem>.chunks.npz）的存取。

嵌入提取阶段（python -m diarization.extract.app）写入，聚类阶段
（python -m diarization.cluster.app）读取。
纯 numpy npz，无 pickle，可跨进程/跨机器传递。

文件布局：

- 元信息：`uri`
- 逐 chunk：`chunk_index` / `chunk_start` / `commit_start` / `commit_end` /
  `frame_step` / `seg_num_frames` / `seg_num_locals` / `obs_count`（长度均为 C）
- 帧分数：`seg_values`——各 chunk 的 seg_scores 按行拼接的一维 float32，
  加载时按 (num_frames, num_locals) 切片还原
- 逐 observation（N 条）：`local_idx` / `start` / `end` /
  `duration` / `mean_activity` / `allow_centroid_update` / `selection_mode` /
  `has_embedding` / `embeddings`（N×dim；无 embedding 时全零且 flag=False）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import ChunkArtifacts, ChunkObservation
from .paths import ensure_parent_dir


def save_chunks(path: str, uri: str, artifacts: list[ChunkArtifacts]) -> None:
    """把整段音频的 chunk artifacts 写入 npz。"""

    ensure_parent_dir(path)

    # 各 chunk 的 seg_scores 形状为 (num_frames, num_locals)，其中 num_locals
    # 随 chunk 变化（ragged），无法直接堆成规则数组；
    # 统一拉平后拼接，加载时按 seg_num_frames × seg_num_locals 切片还原。
    seg_values = np.concatenate(
        [chunk.seg_scores.reshape(-1).astype(np.float32) for chunk in artifacts]
    ) if artifacts else np.zeros(0, dtype=np.float32)

    observations = [obs for chunk in artifacts for obs in chunk.observations]
    # embedding 维度从第一条有效 observation 推断；全部缺失时退化为 (N, 0)。
    embedding_dim = 0
    for obs in observations:
        if obs.embedding is not None:
            embedding_dim = int(obs.embedding.shape[0])
            break
    # 无 embedding 的 observation 以全零行占位，由 has_embedding 标记区分。
    embeddings = np.zeros((len(observations), embedding_dim), dtype=np.float32)
    for row, obs in enumerate(observations):
        if obs.embedding is not None:
            embeddings[row] = obs.embedding

    np.savez(
        path,
        uri=np.array(uri),
        # 逐 chunk。
        chunk_index=np.array([c.chunk_index for c in artifacts], dtype=np.int64),
        chunk_start=np.array([c.chunk_start for c in artifacts], dtype=np.float64),
        commit_start=np.array([c.commit_start for c in artifacts], dtype=np.float64),
        commit_end=np.array([c.commit_end for c in artifacts], dtype=np.float64),
        frame_step=np.array([c.frame_step for c in artifacts], dtype=np.float64),
        seg_num_frames=np.array(
            [c.seg_scores.shape[0] for c in artifacts], dtype=np.int64
        ),
        seg_num_locals=np.array(
            [c.seg_scores.shape[1] for c in artifacts], dtype=np.int64
        ),
        obs_count=np.array(
            [len(c.observations) for c in artifacts], dtype=np.int64
        ),
        # 帧分数（拼接）。
        seg_values=seg_values,
        # 逐 observation。
        local_idx=np.array([o.local_idx for o in observations], dtype=np.int64),
        start=np.array([o.start for o in observations], dtype=np.float64),
        end=np.array([o.end for o in observations], dtype=np.float64),
        duration=np.array([o.duration for o in observations], dtype=np.float64),
        mean_activity=np.array(
            [o.mean_activity for o in observations], dtype=np.float64
        ),
        allow_centroid_update=np.array(
            [o.allow_centroid_update for o in observations], dtype=bool
        ),
        selection_mode=np.array([o.selection_mode for o in observations]),
        has_embedding=np.array(
            [o.embedding is not None for o in observations], dtype=bool
        ),
        embeddings=embeddings,
    )


def load_chunks(path: str) -> tuple[str, list[ChunkArtifacts]]:
    """读取 npz，还原 uri 与逐 chunk 的 ChunkArtifacts。"""

    data = np.load(Path(path), allow_pickle=False)
    uri = str(data["uri"])

    num_chunks = int(data["chunk_start"].shape[0])
    seg_num_frames = data["seg_num_frames"]
    seg_num_locals = data["seg_num_locals"]
    obs_count = data["obs_count"]
    seg_values = data["seg_values"]

    # seg_offset / obs_offset 分别是拼接数组中的游标：逐 chunk 按
    # (num_frames × num_locals) 与 obs_count 切片，还原 ragged 结构。
    seg_offset = 0
    obs_offset = 0
    artifacts: list[ChunkArtifacts] = []
    for chunk_pos in range(num_chunks):
        num_frames = int(seg_num_frames[chunk_pos])
        num_locals = int(seg_num_locals[chunk_pos])
        seg_size = num_frames * num_locals
        seg_scores = seg_values[seg_offset : seg_offset + seg_size].reshape(
            num_frames, num_locals
        )
        seg_offset += seg_size

        observations: list[ChunkObservation] = []
        for obs_pos in range(obs_offset, obs_offset + int(obs_count[chunk_pos])):
            # has_embedding=False 的行为占位零，还原为 None 而非零向量，
            # 避免下游把零向量误当作有效 embedding 参与相似度计算。
            embedding = (
                data["embeddings"][obs_pos].copy()
                if bool(data["has_embedding"][obs_pos])
                else None
            )
            observations.append(
                ChunkObservation(
                    local_idx=int(data["local_idx"][obs_pos]),
                    start=float(data["start"][obs_pos]),
                    end=float(data["end"][obs_pos]),
                    duration=float(data["duration"][obs_pos]),
                    embedding=embedding,
                    mean_activity=float(data["mean_activity"][obs_pos]),
                    allow_centroid_update=bool(data["allow_centroid_update"][obs_pos]),
                    selection_mode=str(data["selection_mode"][obs_pos]),
                )
            )
        obs_offset += int(obs_count[chunk_pos])

        artifacts.append(
            ChunkArtifacts(
                chunk_index=int(data["chunk_index"][chunk_pos]),
                seg_scores=seg_scores,
                frame_step=float(data["frame_step"][chunk_pos]),
                chunk_start=float(data["chunk_start"][chunk_pos]),
                commit_start=float(data["commit_start"][chunk_pos]),
                commit_end=float(data["commit_end"][chunk_pos]),
                observations=observations,
            )
        )

    return uri, artifacts


__all__ = ["save_chunks", "load_chunks"]
