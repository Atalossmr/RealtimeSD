"""chunk 管线 CLI 入口编排模块。"""

from __future__ import annotations

import logging
import os
import sys

from .config import (
    build_arg_parser,
    config_from_args,
    merge_args_with_config,
    validate_runtime_args,
)
from .pipeline import ChunkDiarizationPipeline
from .utils import collect_audio_paths, setup_logger


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
    run_log_path = os.path.join(args.output_dir, "run.log")
    setup_logger(bool(getattr(args, "verbose", False)), run_log_path)

    logger.info("Run log is written to %s", run_log_path)
    logger.info("Collected %d audio file(s) for processing", len(audio_paths))

    pipeline = ChunkDiarizationPipeline(config, args.model_path)
    failed: list[str] = []
    for audio_path in audio_paths:
        logger.info("Processing %s", audio_path)
        try:
            output_path = pipeline.process_file(audio_path)
        except Exception:
            # 单文件失败不中断整批：记录后继续，最后统一非零退出。
            failed.append(audio_path)
            logger.exception("Failed to process %s", audio_path)
            continue
        logger.info("Wrote raw RTTM to %s", output_path)

    if failed:
        logger.error(
            "%d/%d file(s) failed: %s", len(failed), len(audio_paths), failed
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
