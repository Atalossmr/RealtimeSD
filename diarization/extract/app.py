"""嵌入提取阶段 CLI：音频 -> <stem>.chunks.npz。

只运行 segmentation + embedding 并把 chunk artifacts 落盘，不做聚类与 RTTM 输出；
产物由 cluster_chunks.py（聚类阶段）消费。
"""

from __future__ import annotations

import logging
import os
import sys

from ..config import (
    build_arg_parser,
    config_from_args,
    merge_args_with_config,
    validate_runtime_args,
)
from ..utils import collect_audio_paths, setup_logger
from .extractor import ChunkExtractor


logger = logging.getLogger(__name__)


def main() -> None:
    """CLI 入口。"""

    parser = build_arg_parser()
    raw_args = parser.parse_args()
    args = merge_args_with_config(parser, raw_args, sys.argv[1:])
    validate_runtime_args(args)

    config = config_from_args(args)
    audio_paths = collect_audio_paths(args.wav)
    os.makedirs(args.output_dir, exist_ok=True)
    run_log_path = os.path.join(args.output_dir, "extract.log")
    setup_logger(bool(getattr(args, "verbose", False)), run_log_path)

    logger.info("Run log is written to %s", run_log_path)
    logger.info("Collected %d audio file(s) for extraction", len(audio_paths))

    extractor = ChunkExtractor(config, args.model_path)
    for audio_path in audio_paths:
        logger.info("Extracting %s", audio_path)
        chunks_path = extractor.extract_file(audio_path)
        logger.info("Wrote chunks to %s", chunks_path)


if __name__ == "__main__":
    main()
