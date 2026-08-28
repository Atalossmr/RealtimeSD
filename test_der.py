#!/usr/bin/env python3
"""DER 评估脚本（替代原 test_der.sh）：跑实时管线 + 对 refined RTTM（流式后端的最终输出）计算 DER。

运行参数以 `config.yaml` 为准，脚本只补充运行时必须信息和少量覆盖项。
管线批量执行由 `pipeline.process_batch` 完成（模型只加载一次，单文件失败
不中断整批），本脚本只负责 CLI、DER 计算与退出码。
环境变量仍作为缺省值生效：

    CONFIG_PATH → --config        MODEL_PATH   → --model_path
    HF_TOKEN    → --hf_token      HF_CACHE_DIR → --hf_cache_dir
    DEVICE      → --device
    REF_RTTM / REF_RTTM_DIR → --ref_rttm
    DEBUG=0     → --no-debug      SHOW_RTTM=1  → --show_rttm
    DER_VERBOSE=0 → --no-der_verbose
    OUTPUT_ROOT → --output_root   RUN_NAME     → --run_name

用法：

    python3 test_der.py <音频文件|音频目录|清单文件> [--ref_rttm <参考 rttm 目录>]

批量（目录/清单）模式下，全部音频的 RTTM 写入同一输出目录，最后对整目录
计算逐文件 DER 与拼接全局 DER（tools/compute_der.py 目录模式）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pipeline
from common.config import config_from_args, merge_args_with_config
from diarization.utils import collect_audio_paths
from pipeline import run_der_evaluation
from pipeline.run_helpers import add_common_args, env_flag

_TEST_NAME = "der_test"  # 输出固定在 {output_root}/der_test/{run_name}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时管线 DER 评估（支持批量）")
    parser.add_argument(
        "audio",
        help="输入音频：单个 wav/flac/mp3、音频目录，或每行一个路径的文本清单",
    )
    add_common_args(parser)
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE") or None,
        help="运行设备覆盖（如 cpu/cuda/cuda:0）；缺省用 YAML 配置的 device",
    )
    parser.add_argument(
        "--ref_rttm",
        default=os.environ.get("REF_RTTM")
        or os.environ.get("REF_RTTM_DIR")
        or "./datasets/rttm/",
        help="参考 RTTM 目录；置空则跳过 DER 计算",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=env_flag("DEBUG", True),
    )
    parser.add_argument(
        "--show_rttm",
        action=argparse.BooleanOptionalAction,
        default=env_flag("SHOW_RTTM", False),
    )
    parser.add_argument(
        "--der_verbose",
        action=argparse.BooleanOptionalAction,
        default=env_flag("DER_VERBOSE", True),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"ERROR: 音频输入不存在: {audio}")
    audio_paths = collect_audio_paths(str(audio))
    if not audio_paths:
        raise SystemExit(f"ERROR: 未收集到音频文件: {audio}")

    dirs = pipeline.resolve_run_dirs(args.output_root, _TEST_NAME, args.run_name)

    print("==========================================")
    print("Testing with YAML config")
    print(f"audio_input={audio} ({len(audio_paths)} file(s))")
    print(f"config_path={args.config}")
    print(f"output_dir={dirs.exp_dir}")
    if args.model_path:
        print("model_path override enabled")
    if args.hf_cache_dir:
        print("hf_cache_dir override enabled")
    if args.device:
        print(f"device override: {args.device}")
    print("==========================================")

    # 合并 CLI 与 YAML（argparse 默认值 < YAML < 显式 CLI），构建管线配置。
    merged_args = merge_args_with_config(parser, args, sys.argv[1:])
    # 沿用原 build_pipeline_cmd 的传参语义：--debug/--show_rttm 为 true
    # （含环境变量缺省）时作为显式 CLI 覆盖 YAML；为 false 时落回 YAML。
    if args.debug:
        merged_args.debug = True
    if args.show_rttm:
        merged_args.show_rttm = True
    config = config_from_args(merged_args)

    # 批量执行：模型只加载一次，逐文件处理；单文件失败不中断整批。
    batch = pipeline.process_batch(
        audio_paths,
        config=config,
        config_path=args.config,
        output_root=args.output_root,
        run_name=args.run_name,
        test_name=_TEST_NAME,
        model_path=merged_args.model_path,
        verbose=args.debug,
        header_fields={
            "audio_input": audio,
            "num_audio_files": len(audio_paths),
            "config_path": args.config,
            "with_separation": False,
            "model_path_override": args.model_path,
            "hf_cache_dir_override": args.hf_cache_dir,
            "device_override": args.device,
            "der_verbose": int(args.der_verbose),
            "ref_path": args.ref_rttm or None,
        },
    )

    print("\n========== DER Results ==========")
    if not args.ref_rttm:
        print("DER skipped: --ref_rttm not set")
    else:
        der_rc = run_der_evaluation(
            batch.exp_dir,
            args.ref_rttm,
            args.run_name,
            args.der_verbose,
            batch.results_file,
        )
        if der_rc != 0:
            return der_rc

    print(f"\nResults saved to {batch.results_file}")
    print(f"Outputs written to: {dirs.basic_dir}")
    if batch.failed:
        print(
            f"WARNING: {len(batch.failed)} file(s) failed: {batch.failed}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
