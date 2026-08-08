#!/usr/bin/env python3
"""merge_threshold 扫描：复用 chunks.npz，对 streaming 后端的 merge 阈值做实验并算 DER。

只跑聚类阶段（不加载任何模型），每个 merge_threshold 重放全部 npz 生成 RTTM，
再用本目录 compute_der.py 的批量评测算 global DER。match/new 阈值等其余参数
保持 YAML 原值不动。merge_threshold > 1.0 等价于关闭 merge（余弦相似度 ≤ 1）。

用法：
  .venv/bin/python tools/sweep_merge_threshold.py \
      --input exp/merge_sweep/chunks \
      --ref datasets/aishell4-test/rttm \
      --config config/config.yaml \
      --output exp/merge_sweep/merge_threshold.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_der import compute_der_batch  # noqa: E402
from sweep_thresholds import load_base_config  # noqa: E402

from diarization.cluster.backends import build_assigner  # noqa: E402
from diarization.cluster.runner import run_clustering  # noqa: E402
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.config import ChunkPipelineConfig  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("sweep_merge_threshold")

REF_DIR: Path


def count_rttm_speakers(rttm_dir: str, suffix: str) -> dict[str, int]:
    """统计目录下每个 RTTM 文件的 unique speaker 数（uri -> 数量）。"""

    counts: dict[str, int] = {}
    for path in sorted(Path(rttm_dir).glob(f"*{suffix}")):
        speakers = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("SPEAKER"):
                speakers.add(line.split()[7])
        counts[path.name[: -len(suffix)]] = len(speakers)
    return counts


def run_one_threshold(
    base_config: ChunkPipelineConfig,
    chunks_files: list[Path],
    merge_threshold: float,
    hold_chunks: int,
    protect_established: bool,
    work_dir: Path,
    match_threshold: Optional[float] = None,
    new_threshold: Optional[float] = None,
) -> dict[str, float]:
    """用一组阈值组合重放全部 chunk artifacts，返回 DER 摘要。

    match_threshold / new_threshold 为 None 时保持 YAML 原值。
    """

    overrides = {
        "merge_threshold": merge_threshold,
        "new_speaker_hold_chunks": hold_chunks,
        "merge_protect_established": protect_established,
    }
    if match_threshold is not None:
        overrides["global_match_threshold"] = match_threshold
    if new_threshold is not None:
        overrides["new_speaker_threshold"] = new_threshold
    config = dataclasses.replace(base_config, **overrides)
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
    sys_speakers = sum(count_rttm_speakers(str(work_dir), ".streaming.rttm").values())
    return {
        "ms": float(global_result["ms"]),
        "fa": float(global_result["fa"]),
        "ser": float(global_result["ser"]),
        "der": float(global_result["der"]),
        "sys_speakers": float(sys_speakers),
    }


def main() -> None:
    global REF_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", required=True, help="结果 CSV 路径")
    parser.add_argument(
        "--thresholds",
        default="1.01,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="逗号分隔的 merge_threshold 列表；>1.0 等价关闭 merge",
    )
    parser.add_argument(
        "--hold-chunks",
        default="0",
        help="逗号分隔的 new_speaker_hold_chunks 列表，与 thresholds 做笛卡尔积",
    )
    parser.add_argument(
        "--protect-established",
        action="store_true",
        help="开启 merge_protect_established（已存活过缓冲期的 speaker 禁止被并）",
    )
    parser.add_argument(
        "--match-thresholds",
        default=None,
        help="逗号分隔的 global_match_threshold 列表；缺省保持 YAML 值",
    )
    parser.add_argument(
        "--new-thresholds",
        default=None,
        help="逗号分隔的 new_speaker_threshold 列表；缺省保持 YAML 值",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    REF_DIR = Path(args.ref)

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")
    base_config = load_base_config(args.config)
    ref_speakers = sum(count_rttm_speakers(str(REF_DIR), ".rttm").values())
    logger.info("ref_speakers(total)=%d", ref_speakers)

    thresholds = [float(token) for token in args.thresholds.split(",")]
    hold_list = [int(token) for token in args.hold_chunks.split(",")]
    match_list: list[Optional[float]] = (
        [float(token) for token in args.match_thresholds.split(",")]
        if args.match_thresholds
        else [None]
    )
    new_list: list[Optional[float]] = (
        [float(token) for token in args.new_thresholds.split(",")]
        if args.new_thresholds
        else [None]
    )
    rows: list[dict[str, float]] = []
    for hold_chunks in hold_list:
        for match_thr in match_list:
            for new_thr in new_list:
                for merge_threshold in thresholds:
                    with tempfile.TemporaryDirectory(
                        prefix="sweep_merge_rttm_"
                    ) as tmp_dir:
                        metrics = run_one_threshold(
                            base_config,
                            chunks_files,
                            merge_threshold,
                            hold_chunks,
                            args.protect_established,
                            Path(tmp_dir),
                            match_threshold=match_thr,
                            new_threshold=new_thr,
                        )
                    row = {
                        "global_match_threshold": match_thr
                        if match_thr is not None
                        else base_config.global_match_threshold,
                        "new_speaker_threshold": new_thr
                        if new_thr is not None
                        else base_config.new_speaker_threshold,
                        "merge_threshold": merge_threshold,
                        "hold_chunks": hold_chunks,
                        "protect_established": int(args.protect_established),
                        **metrics,
                    }
                    rows.append(row)
                    logger.info(
                        "match=%.2f new=%.2f merge=%.2f hold=%d protect=%d -> "
                        "MS=%.2f FA=%.2f SER=%.2f DER=%.2f sys_spk=%d",
                        row["global_match_threshold"],
                        row["new_speaker_threshold"],
                        merge_threshold,
                        hold_chunks,
                        int(args.protect_established),
                        metrics["ms"],
                        metrics["fa"],
                        metrics["ser"],
                        metrics["der"],
                        int(metrics["sys_speakers"]),
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
