"""应用入口编排模块。"""

from __future__ import annotations

import logging
import os

import sys

from .cli import (
    build_arg_parser,
    config_from_args,
    merge_args_with_config,
    validate_runtime_args,
)
from .pipeline import NativeOnlineSpeakerDiarization
from .utils import collect_audio_paths, setup_logger


logger = logging.getLogger(__name__)


def main() -> None:
    """CLI 入口。

    这里故意只保留“参数解析 + 初始化 + 批量跑文件”的最薄一层，
    让外部维护者能一眼看到脚本真正的运行入口，而不是再去翻大量算法细节。
    """

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

    pipeline = NativeOnlineSpeakerDiarization(config, args.model_path)
    for audio_path in audio_paths:
        logger.info("Processing %s", audio_path)
        output_path = pipeline.process_file(audio_path)
        logger.info("Wrote streaming RTTM to %s", output_path)
