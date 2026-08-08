#!/usr/bin/env python3
"""min_segment_duration_for_new_speaker 扫描：复用 chunks.npz 评估对聚类的影响。

只跑聚类阶段（不加载模型）：固定 global_match_threshold / new_speaker_threshold
为配置值，扫 min_segment_duration_for_new_speaker，每个取值重放全部 npz 生成
RTTM 并算 global DER（ms/fa/ser/der）。

用法：
  .venv/bin/python tools/sweep_new_speaker_duration.py \
      --input exp/seg_duration_max3 \
      --ref datasets/aishell4-test/rttm \
      --config config/config.yaml \
      --output exp/threshold_sweep/new_speaker_min_dur.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_der import compute_der_batch  # noqa: E402
from sweep_thresholds import load_base_config  # noqa: E402

from diarization.cluster.backends import build_assigner  # noqa: E402
from diarization.cluster.runner import run_clustering  # noqa: E402
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("sweep_new_speaker_duration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", required=True, help="结果 CSV 路径")
    parser.add_argument(
        "--grid",
        default="0.30,0.50,0.80,1.00,1.50,2.00",
        help="逗号分隔的扫描取值（秒）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")
    base_config = load_base_config(args.config)
    grid = [float(v) for v in args.grid.split(",")]

    import csv

    rows: list[dict[str, float]] = []
    for min_dur in grid:
        config = dataclasses.replace(
            base_config, min_segment_duration_for_new_speaker=min_dur
        )
        with tempfile.TemporaryDirectory(prefix="sweep_nsd_") as tmp_dir:
            for chunks_path in chunks_files:
                uri, artifacts = load_chunks(str(chunks_path))
                writer = AppendOnlyRTTMWriter(
                    str(Path(tmp_dir) / f"{uri}.streaming.rttm"),
                    uri,
                    config.min_segment_duration,
                    config.streaming_merge_gap,
                    False,
                )
                run_clustering(artifacts, build_assigner(config), writer)
            _, _, global_result = compute_der_batch(
                ref_path=args.ref,
                sys_path=tmp_dir,
                collar=0.0,
                ignore_overlap=False,
                sys_suffix=".streaming.rttm",
                ref_suffix=".rttm",
            )
        assert global_result is not None
        row = {
            "min_segment_duration_for_new_speaker": min_dur,
            "ms": float(global_result["ms"]),
            "fa": float(global_result["fa"]),
            "ser": float(global_result["ser"]),
            "der": float(global_result["der"]),
        }
        rows.append(row)
        logger.info(
            "min_dur=%.2f -> MS=%.2f FA=%.2f SER=%.2f DER=%.2f",
            min_dur,
            row["ms"],
            row["fa"],
            row["ser"],
            row["der"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
