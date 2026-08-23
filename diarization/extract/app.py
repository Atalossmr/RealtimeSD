"""嵌入提取阶段 CLI：音频 -> <stem>.chunks.npz。

只运行 segmentation + embedding 并把 chunk artifacts 落盘，不做聚类与 RTTM 输出；
产物由聚类阶段（python -m diarization.cluster.app）消费。
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
    run_log_path = os.path.join(args.output_dir, "logs", "extract.log")
    setup_logger(bool(getattr(args, "verbose", False)), run_log_path)

    logger.info("Run log is written to %s", run_log_path)
    logger.info("Collected %d audio file(s) for extraction", len(audio_paths))

    extractor = ChunkExtractor(config, args.model_path)
    failed: list[str] = []
    for audio_path in audio_paths:
        logger.info("Extracting %s", audio_path)
        try:
            chunks_path = extractor.extract_file(audio_path)
        except Exception:
            # 单文件失败不中断整批：记录后继续，最后统一非零退出。
            failed.append(audio_path)
            logger.exception("Failed to extract %s", audio_path)
            continue
        logger.info("Wrote chunks to %s", chunks_path)

    if failed:
        logger.error("%d/%d file(s) failed: %s", len(failed), len(audio_paths), failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
