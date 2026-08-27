"""stage1 端到端 CLI：音频 -> RTTM（不启用 separation 分段导出）。

用法：

    python -m diarization.app --wav <音频> --output_dir <目录> [--config ...]

主体复用仓库根目录 run.py 的 `run_diarization`（`python -m diarization.app`
从仓库根目录运行时 run.py 可导入），避免复制组合逻辑。仅接受单个音频文件
（与 run.py 一致，不做目录批量）；separation 不启用 ⇒ 无 segments 输出，
ahc 后端也可用（RTTM-only 本来就支持）。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from common.config import (
    build_arg_parser,
    config_from_args,
    merge_args_with_config,
    validate_runtime_args,
)
from common.utils import setup_logger
from run import run_diarization


logger = logging.getLogger(__name__)


def main() -> None:
    """CLI 入口。"""

    parser = build_arg_parser()
    raw_args = parser.parse_args()
    args = merge_args_with_config(parser, raw_args, sys.argv[1:])
    validate_runtime_args(args)

    audio = Path(args.wav)
    if not audio.exists():
        parser.error(f"音频输入不存在: {audio}")
    if audio.is_dir():
        parser.error(f"音频输入只接受单个文件，不支持目录: {audio}")

    config = config_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    run_log_path = os.path.join(args.output_dir, "logs", "run.log")
    setup_logger(bool(getattr(args, "verbose", False)), run_log_path)

    logger.info("Run log is written to %s", run_log_path)
    logger.info("Processing %s", audio)
    rttm_path = run_diarization(
        config,
        str(audio),
        Path(args.output_dir),
        model_path=args.model_path,
        with_separation=False,
    )
    logger.info("Wrote raw RTTM to %s", rttm_path)


if __name__ == "__main__":
    main()
