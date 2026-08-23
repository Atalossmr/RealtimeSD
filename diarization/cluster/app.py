"""聚类阶段 CLI：<stem>.chunks.npz -> 聚类 + RTTM 输出。

读取嵌入提取阶段（python -m diarization.extract.app）产出的中间文件，用 YAML 配置的聚类后端
（clustering_backend: streaming / ahc）做 local->global 分配并输出 RTTM
（两后端均为 raw + refined 两级及 speakers.json sidecar）。
不依赖音频与模型，可独立运行。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..utils import load_chunks
from ..config import build_arg_parser, config_from_args, merge_args_with_config
from ..utils import setup_logger
from .backends import build_clusterer
from .rttm_writer import AppendOnlyRTTMWriter
from .runner import run_clustering


logger = logging.getLogger(__name__)


def build_cluster_arg_parser() -> argparse.ArgumentParser:
    """聚类阶段参数解析器（调参项仍以 YAML 为唯一来源）。

    基于完整 pipeline parser 构建，保证 YAML 键名校验接受全部既有配置项
    （模型类参数在聚类阶段不生效，仅为兼容同一份 config.yaml）。
    """

    parser = build_arg_parser()
    parser.description = "读取 .chunks.npz，按 YAML 配置的聚类后端输出 RTTM"
    parser.add_argument(
        "--input",
        default=None,
        help="单个 .chunks.npz、包含 npz 的目录，或每行一个路径的文本文件",
    )
    return parser


def collect_chunks_paths(input_path: str) -> list[str]:
    """收集 .chunks.npz 路径（单文件 / 目录 / 文本清单）。"""

    path = Path(input_path)
    if path.is_dir():
        return [str(p) for p in sorted(path.rglob("*.chunks.npz"))]
    if path.is_file() and path.name.endswith(".chunks.npz"):
        return [str(path)]
    if path.is_file():
        with open(path, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def cluster_file(chunks_path: str, config, output_dir: str) -> str:
    """对单个 chunks.npz 执行聚类并输出 RTTM，返回输出路径。"""

    uri, artifacts = load_chunks(chunks_path)
    clusterer = build_clusterer(config)
    rttm_path = str(Path(output_dir) / "rttm" / f"{uri}.{clusterer.output_tag}.rttm")
    writer = AppendOnlyRTTMWriter(
        rttm_path,
        uri,
        config.min_segment_duration,
        config.segment_merge_gap,
        config.show_rttm,
    )

    # refined 级（两后端通用）：streaming 逐 chunk + merge 事件动态重生成，
    # ahc（deferred）在 EOF 重放完成后最终刷新；均在 EOF 叠加小样本合并。
    from .post_process import RefinedRTTMWriter

    base = rttm_path[: -len(f".{clusterer.output_tag}.rttm")]
    refiner = RefinedRTTMWriter(
        rttm_path,
        f"{base}.refined.rttm",
        clusterer,
        min_count=config.post_merge_min_count,
        min_similarity=config.post_merge_min_similarity,
    )

    run_clustering(artifacts, clusterer, writer, refiner=refiner)
    return rttm_path


def main() -> None:
    """CLI 入口。"""

    parser = build_cluster_arg_parser()
    raw_args = parser.parse_args()
    args = merge_args_with_config(parser, raw_args, sys.argv[1:])
    for required in ("input", "output_dir"):
        if getattr(args, required, None) in {None, ""}:
            parser.error(f"--{required} is required")

    config = config_from_args(args)
    chunks_paths = collect_chunks_paths(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    run_log_path = os.path.join(args.output_dir, "logs", "cluster.log")
    setup_logger(bool(getattr(args, "verbose", False)), run_log_path)

    logger.info("Run log is written to %s", run_log_path)
    logger.info(
        "Clustering %d chunks file(s) with backend=%s",
        len(chunks_paths),
        config.clustering_backend,
    )

    failed: list[str] = []
    for chunks_path in chunks_paths:
        logger.info("Clustering %s", chunks_path)
        try:
            rttm_path = cluster_file(chunks_path, config, args.output_dir)
        except Exception:
            # 单文件失败不中断整批：记录后继续，最后统一非零退出。
            failed.append(chunks_path)
            logger.exception("Failed to cluster %s", chunks_path)
            continue
        logger.info("Wrote RTTM to %s", rttm_path)

    if failed:
        logger.error(
            "%d/%d file(s) failed: %s", len(failed), len(chunks_paths), failed
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
