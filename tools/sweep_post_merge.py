#!/usr/bin/env python3
"""post-merge 参数扫描：复用 chunks.npz，对小样本簇合并参数做网格实验并算 DER。

只跑聚类阶段（不加载任何模型）。ahc 后端在 finalize 内重映射标签，输出
仍为 .ahc.rttm；streaming 后端输出 .raw.rttm（append-only）并动态重生成
.refined.rttm（merge 修正 + final 时叠加小样本合并），评估 refined。
min_duration = 0 的组合即基线。

用法：
  .venv/bin/python tools/sweep_post_merge.py \
      --input exp/der_full/extract \
      --ref datasets/aishell4-test/rttm \
      --config tmp/config_der_ahc.yaml --backend ahc \
      --durations 0,5,10,15,30 --similarities 0.0,0.3,0.5 \
      --output exp/der_full/post_merge_ahc.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_der import compute_der_batch  # noqa: E402

from diarization.cluster.backends import build_assigner  # noqa: E402
from diarization.cluster.post_merge import write_refined_rttm  # noqa: E402
from diarization.cluster.runner import run_clustering  # noqa: E402
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.config import ChunkPipelineConfig  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("sweep_post_merge")


def load_base_config(config_path: str) -> ChunkPipelineConfig:
    """直接从 YAML 构造配置（跳过 CLI 合并，只取 dataclass 字段）。"""

    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg_dict = yaml.safe_load(file_obj) or {}
    field_names = {f.name for f in dataclasses.fields(ChunkPipelineConfig)}
    return ChunkPipelineConfig(
        **{k: v for k, v in cfg_dict.items() if k in field_names}
    )


def run_one_combo(
    base_config: ChunkPipelineConfig,
    chunks_files: list[Path],
    backend: str,
    min_duration: float,
    min_similarity: float,
    work_dir: Path,
    ref_dir: Path,
) -> dict[str, float]:
    """用一组 post-merge 参数重放全部 chunk artifacts，返回 DER 摘要。"""

    config = dataclasses.replace(
        base_config,
        clustering_backend=backend,
        post_merge_min_speech_duration=min_duration,
        post_merge_min_similarity=min_similarity,
    )
    output_tag = ""
    for chunks_path in chunks_files:
        uri, artifacts = load_chunks(str(chunks_path))
        assigner = build_assigner(config)
        output_tag = assigner.output_tag
        rttm_path = work_dir / f"{uri}.{output_tag}.rttm"
        writer = AppendOnlyRTTMWriter(
            str(rttm_path),
            uri,
            config.min_segment_duration,
            config.streaming_merge_gap,
            False,
        )
        run_clustering(artifacts, assigner, writer)
        # streaming 后端：refined 为最终输出（merge 修正 + final 小样本合并）。
        if not assigner.deferred:
            write_refined_rttm(
                str(rttm_path),
                str(work_dir / f"{uri}.refined.rttm"),
                centroids=getattr(assigner, "centroids", {}),
                merged_into=getattr(assigner, "merged_into", {}),
                min_duration=min_duration,
                min_similarity=min_similarity,
            )
    # streaming 一律评估 refined（最终输出）；ahc 评估后端原生输出。
    sys_suffix = (
        ".refined.rttm" if backend == "streaming" else f".{output_tag}.rttm"
    )
    _, _, global_result = compute_der_batch(
        ref_path=str(ref_dir),
        sys_path=str(work_dir),
        collar=0.0,
        ignore_overlap=False,
        sys_suffix=sys_suffix,
        ref_suffix=".rttm",
    )
    assert global_result is not None
    return {
        "ms": float(global_result["ms"]),
        "fa": float(global_result["fa"]),
        "ser": float(global_result["ser"]),
        "der": float(global_result["der"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--backend", default="ahc", choices=["ahc", "streaming"],
    )
    parser.add_argument(
        "--durations",
        default="0,5,10,15,30",
        help="post_merge_min_speech_duration 网格（逗号分隔，含 0=基线）",
    )
    parser.add_argument(
        "--similarities",
        default="0.0,0.3,0.5",
        help="post_merge_min_similarity 网格（逗号分隔）",
    )
    parser.add_argument("--output", required=True, help="结果 CSV 路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ref_dir = Path(args.ref)
    durations = [float(x) for x in args.durations.split(",")]
    similarities = [float(x) for x in args.similarities.split(",")]

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")
    base_config = load_base_config(args.config)

    rows: list[dict[str, float]] = []
    for min_duration in durations:
        for min_similarity in similarities:
            if min_duration == 0.0 and min_similarity != similarities[0]:
                continue  # 基线与 similarity 无关，只跑一组
            with tempfile.TemporaryDirectory(prefix="post_merge_rttm_") as tmp_dir:
                metrics = run_one_combo(
                    base_config,
                    chunks_files,
                    args.backend,
                    min_duration,
                    min_similarity,
                    Path(tmp_dir),
                    ref_dir,
                )
            row = {
                "backend": args.backend,
                "min_speech_duration": min_duration,
                "min_similarity": min_similarity,
                **metrics,
            }
            rows.append(row)
            logger.info(
                "backend=%s min_dur=%.1f min_sim=%.2f -> MS=%.2f FA=%.2f "
                "SER=%.2f DER=%.2f",
                args.backend,
                min_duration,
                min_similarity,
                metrics["ms"],
                metrics["fa"],
                metrics["ser"],
                metrics["der"],
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: r["der"])
    baseline = next(r for r in rows if r["min_speech_duration"] == 0.0)
    logger.info("baseline: %s", baseline)
    logger.info("best: %s", best)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
