#!/usr/bin/env python3
"""阈值扫描：复用 chunks.npz，对 streaming 后端的两阈值做网格实验并算 DER。

只跑聚类阶段（不加载任何模型），每个 (global_match_threshold, new_speaker_threshold)
组合重放全部 npz 生成 RTTM，再用本目录 compute_der.py 的批量评测算 global DER。

用法：
  .venv/bin/python tools/sweep_thresholds.py \
      --input exp/seg_duration_stat \
      --ref datasets/aishell4-test/rttm \
      --config config/config.yaml \
      --output exp/threshold_sweep/baseline.csv
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
from diarization.cluster.runner import run_clustering  # noqa: E402
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.config import ChunkPipelineConfig  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("sweep_thresholds")


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
    match_thr: float,
    new_thr: float,
    work_dir: Path,
) -> dict[str, float]:
    """用一组阈值重放全部 chunk artifacts，返回 DER 摘要。"""

    config = dataclasses.replace(
        base_config,
        global_match_threshold=match_thr,
        new_speaker_threshold=new_thr,
    )
    for chunks_path in chunks_files:
        uri, artifacts = load_chunks(str(chunks_path))
        writer = AppendOnlyRTTMWriter(
            str(work_dir / f"{uri}.streaming.rttm"),
            uri,
            config.min_segment_duration,
            config.streaming_merge_gap,
            False,
        )
        run_clustering(artifacts, build_assigner(config), writer)
    _, _, global_result = compute_der_batch(
        ref_path=str(REF_DIR),
        sys_path=str(work_dir),
        collar=0.0,
        ignore_overlap=False,
        sys_suffix=".streaming.rttm",
        ref_suffix=".rttm",
    )
    assert global_result is not None
    return {
        "ms": float(global_result["ms"]),
        "fa": float(global_result["fa"]),
        "ser": float(global_result["ser"]),
        "der": float(global_result["der"]),
    }


REF_DIR: Path


def main() -> None:
    global REF_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", required=True, help="结果 CSV 路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    REF_DIR = Path(args.ref)

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")
    base_config = load_base_config(args.config)

    grid = [0.40, 0.45, 0.50, 0.55, 0.60]
    rows: list[dict[str, float]] = []
    for match_thr in grid:
        for new_thr in grid:
            with tempfile.TemporaryDirectory(prefix="sweep_rttm_") as tmp_dir:
                metrics = run_one_combo(
                    base_config,
                    chunks_files,
                    match_thr,
                    new_thr,
                    Path(tmp_dir),
                )
            row = {
                "global_match_threshold": match_thr,
                "new_speaker_threshold": new_thr,
                **metrics,
            }
            rows.append(row)
            logger.info(
                "match=%.2f new=%.2f -> MS=%.2f FA=%.2f SER=%.2f DER=%.2f",
                match_thr,
                new_thr,
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
    logger.info("best: %s", best)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
