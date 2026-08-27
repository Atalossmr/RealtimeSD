#!/usr/bin/env python3
"""E4c AHC + oracle 说话人数：固定 n_clusters=真值说话人数，量化簇数选不准的损失。

现有 ahc 后端用 distance_threshold 自动定簇数；本脚本逐文件 load_chunks，
按 chunk 顺序收集全部带 embedding 的 observation，用 sklearn
AgglomerativeClustering(n_clusters=ref 说话人数, metric='cosine',
linkage=config.ahc_linkage) 聚类，再按原 chunk 顺序重建逐 chunk 的
local_to_global（同 chunk 同 local 的多条 observation 取第一条，与 ahc 后端
finalize 的重建口径一致；无 embedding 的 local 不进 mapping，writer 对其帧
直接丢弃），重放 AppendOnlyRTTMWriter 写 {uri}.raw.rttm（落 raw/ 子目录，
避免被 sys_suffix='.rttm' 的批量评估误收），再叠加 refined 级逻辑
（merged_into 传空 dict——离线 AHC 无流式 merge；centroids/counts 用
post_process.cluster_stats 从聚类结果算；min_count 用配置的
post_merge_min_count）写 {uri}.rttm，最后 compute_der_batch 评 DER。

用法：
  .venv/bin/python tools/experiments/ahc_oracle_k.py \
      --input exp/experiments/task1-6_extract/embeddings \
      --ref datasets/task1-6/rttm \
      --config config/config.yaml \
      --output-dir exp/experiments/e4c_ahc_oracle_k/rttm \
      --report exp/experiments/e4c_ahc_oracle_k/report.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from compute_der import compute_der_batch  # noqa: E402
from rttm_utils import load_rttm  # noqa: E402

from common.config import ChunkPipelineConfig  # noqa: E402
from diarization.cluster.post_process import (  # noqa: E402
    cluster_stats,
    write_refined_rttm,
)
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("ahc_oracle_k")


def load_base_config(config_path: str) -> ChunkPipelineConfig:
    """直接从 YAML 构造配置（与 tools/sweep_params.py 同口径）。"""

    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg_dict = yaml.safe_load(file_obj) or {}
    field_names = {f.name for f in dataclasses.fields(ChunkPipelineConfig)}
    return ChunkPipelineConfig(
        **{k: v for k, v in cfg_dict.items() if k in field_names}
    )


def cluster_oracle_k(
    chunks_path: Path, num_speakers: int, config: ChunkPipelineConfig
) -> tuple[str, list, list[dict[int, int]], dict, dict]:
    """单文件聚类：返回 (uri, artifacts, 逐 chunk local_to_global, centroids, counts)。"""

    from sklearn.cluster import AgglomerativeClustering

    uri, artifacts = load_chunks(str(chunks_path))

    # 按 chunk 顺序收集带 embedding 的 observation（cursor 顺序与下方重建
    # 映射时严格一致，同 ahc 后端 finalize 的口径）。
    embedded_per_chunk = [
        [obs for obs in chunk.observations if obs.embedding is not None]
        for chunk in artifacts
    ]
    observations = [obs for chunk in embedded_per_chunk for obs in chunk]
    if not observations:
        return uri, artifacts, [{} for _ in artifacts], {}, {}

    embeddings = np.stack([obs.embedding for obs in observations])
    n_clusters = min(num_speakers, len(observations))
    model = AgglomerativeClustering(
        metric="cosine",
        linkage=config.ahc_linkage,
        n_clusters=n_clusters,
    )
    labels = model.fit_predict(embeddings)
    logger.info(
        "[ahc-oracle] %s: observations=%d n_clusters=%d (ref speakers=%d)",
        uri,
        len(observations),
        n_clusters,
        num_speakers,
    )

    centroids, _ = cluster_stats(labels, observations, embeddings)
    counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
    }

    # 重建逐 chunk local_to_global：同 chunk 同 local 的多条 observation
    # 取第一条（cursor 顺序保证）；无 embedding 的 local 不进 mapping。
    assignments: list[dict[int, int]] = []
    cursor = 0
    for chunk_obs in embedded_per_chunk:
        local_to_global: dict[int, int] = {}
        for obs in chunk_obs:
            local_to_global.setdefault(int(obs.local_idx), int(labels[cursor]))
            cursor += 1
        assignments.append(local_to_global)
    return uri, artifacts, assignments, centroids, counts


def replay_file(
    chunks_path: Path,
    ref_dir: Path,
    output_dir: Path,
    config: ChunkPipelineConfig,
) -> None:
    """单文件：oracle-k 聚类 + 重放 writer + refined 级输出。"""

    uri_tmp = chunks_path.name[: -len(".chunks.npz")]
    ref_path = ref_dir / f"{uri_tmp}.rttm"
    if not ref_path.is_file():
        raise SystemExit(f"ref RTTM not found for {uri_tmp}: {ref_path}")
    num_speakers = len(load_rttm(ref_path))

    uri, artifacts, assignments, centroids, counts = cluster_oracle_k(
        chunks_path, num_speakers, config
    )

    raw_dir = output_dir / "raw"
    raw_path = raw_dir / f"{uri}.raw.rttm"
    writer = AppendOnlyRTTMWriter(
        str(raw_path),
        uri,
        config.min_segment_duration,
        config.segment_merge_gap,
        False,
    )
    for chunk, local_to_global in zip(artifacts, assignments):
        writer.consume_chunk(
            chunk.seg_scores,
            chunk.frame_step,
            chunk.chunk_start,
            chunk.commit_start,
            chunk.commit_end,
            local_to_global,
        )
        writer.close_inactive(chunk.commit_end)
    writer.finalize()

    # refined 级：离线 AHC 无流式 merge（merged_into 空），仅叠小样本 post-merge。
    write_refined_rttm(
        str(raw_path),
        str(output_dir / f"{uri}.rttm"),
        centroids=centroids,
        merged_into={},
        counts=counts,
        min_count=int(config.post_merge_min_count),
        min_similarity=float(config.post_merge_min_similarity),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--output-dir", required=True, help="系统 RTTM 输出目录")
    parser.add_argument("--report", required=True, help="结果报告 txt 路径")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    config = load_base_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    for chunks_path in chunks_files:
        replay_file(chunks_path, Path(args.ref), output_dir, config)
        log(f"[replay] {chunks_path.name[: -len('.chunks.npz')]} done")

    results, skipped, global_result = compute_der_batch(
        ref_path=args.ref,
        sys_path=str(output_dir),
        collar=0.0,
        ignore_overlap=False,
        sys_suffix=".rttm",
        ref_suffix=".rttm",
    )
    log()
    log(f"{'file':<6} {'MS%':>7} {'FA%':>7} {'SER%':>7} {'DER%':>7}")
    for row in results:
        log(
            f"{Path(row['sys_rttm']).stem:<6} {row['ms']:>7.2f} {row['fa']:>7.2f} "
            f"{row['ser']:>7.2f} {row['der']:>7.2f}"
        )
    for filename, reason in skipped:
        log(f"skip {filename}: {reason}")
    if global_result is not None:
        log("-" * 40)
        log(
            f"GLOBAL ms={global_result['ms']:.2f} fa={global_result['fa']:.2f} "
            f"ser={global_result['ser']:.2f} der={global_result['der']:.2f} "
            f"({global_result['files']} files)"
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
