#!/usr/bin/env python3
"""实时 diarization 一键运行 CLI（薄封装：编排本体在 `pipeline.process`）。

本脚本只负责：argparse 与环境变量缺省、CLI/YAML 配置合并。管线本体
（输出目录、结果汇总）见 pipeline.api。管线就是「音频 → RTTM」。

环境变量作为对应命令行参数的缺省值：

    MODEL_PATH      → --model_path    HF_TOKEN     → --hf_token
    HF_CACHE_DIR    → --hf_cache_dir  DEBUG=1      → --debug
    SHOW_RTTM=1     → --show_rttm     OUTPUT_ROOT  → --output_root
    RUN_NAME        → --run_name      CONFIG_PATH  → --config

用法：

    python3 run.py <音频文件|目录|清单文本> [--config config/config.yaml] [--run_name default]
    # 目录/清单为批量模式：模型只加载一次，逐文件处理
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pipeline
from common.config import config_from_args, merge_args_with_config
from common.utils import collect_audio_paths
from pipeline.cli import add_common_args, check_input_paths, env_flag

_TEST_NAME = "common"  # 输出固定在 {output_root}/common/{run_name}


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="实时 diarization 一键运行（音频 → RTTM）",
        allow_abbrev=False,  # 见 common/config/pipeline.py 的同名单元注释
    )
    parser.add_argument(
        "audio",
        help="输入音频：单个文件（wav/flac/mp3）、目录，或每行一个路径的清单文本",
    )
    add_common_args(parser)
    parser.add_argument("--device", default="auto", help="运行设备，如 auto/cpu/cuda:0")
    parser.add_argument(
        "--model_type",
        default="eres2netv2",
        choices=["eres2netv2"],
        help="speaker encoder 类型，当前仅支持 eres2netv2",
    )
    parser.add_argument(
        "--segmentation_model",
        default="pyannote/segmentation-3.0",
        help="pyannote segmentation 模型名",
    )
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    parser.add_argument(
        "--debug", action="store_true", default=env_flag("DEBUG", False)
    )
    parser.add_argument(
        "--show_rttm", action="store_true", default=env_flag("SHOW_RTTM", False)
    )
    return parser


def main() -> int:
    """CLI 入口：解析参数并跑管线。"""
    parser = build_parser()
    args = parser.parse_args()

    audio = Path(args.audio)
    config_path = Path(args.config)
    check_input_paths(audio, config_path)
    audio_paths = collect_audio_paths(str(audio))
    if not audio_paths:
        raise SystemExit(f"ERROR: 未收集到音频文件: {audio}")
    dirs = pipeline.resolve_run_dirs(args.output_root, _TEST_NAME, args.run_name)

    # 合并 CLI 与 YAML（argparse 默认值 < YAML < 显式 CLI），构建管线配置。
    merged_args = merge_args_with_config(parser, args, sys.argv[1:])
    # 沿用原 build_pipeline_cmd 的传参语义：--debug/--show_rttm 为 true
    # （含环境变量缺省）时作为显式 CLI 覆盖 YAML；为 false 时落回 YAML。
    if args.debug:
        merged_args.debug = True
    if args.show_rttm:
        merged_args.show_rttm = True
    config = config_from_args(merged_args)

    print("==========================================")
    print("Running with YAML config")
    print(f"audio_input={audio} ({len(audio_paths)} file(s))")
    print(f"config_path={config_path}")
    print(f"output_dir={dirs.exp_dir}")
    for name in ("model_path", "hf_cache_dir"):
        if getattr(args, name):
            print(f"{name} override enabled")
    if args.hf_token:
        print("hf_token override enabled")
    print("==========================================")

    result = pipeline.process(
        audio=audio_paths,
        config=config,
        config_path=config_path,
        output_root=args.output_root,
        run_name=args.run_name,
        test_name=_TEST_NAME,
        model_path=merged_args.model_path,
        verbose=bool(merged_args.verbose or args.debug),
        header_fields={
            "audio_input": audio,
            "num_audio_files": len(audio_paths),
            "config_path": config_path,
            "model_path_override": args.model_path,
            "hf_cache_dir_override": args.hf_cache_dir,
        },
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
