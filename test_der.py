#!/usr/bin/env python3
"""DER 评估脚本（替代原 test_der.sh）：跑实时管线 + 对 refined RTTM（流式后端的最终输出）计算 DER。

运行参数以 `config.yaml` 为准，脚本只补充运行时必须信息和少量常用覆盖项。
环境变量仍作为缺省值生效：

    CONFIG_PATH → --config        MODEL_PATH   → --model_path
    HF_TOKEN    → --hf_token      HF_CACHE_DIR → --hf_cache_dir
    REF_RTTM / REF_RTTM_DIR → --ref_rttm
    DEBUG=0     → --no-debug      SHOW_RTTM=1  → --show_rttm
    DER_VERBOSE=0 → --no-der_verbose
    OUTPUT_ROOT → --output_root   RUN_NAME     → --run_name

用法：

    python3 test_der.py [audio_input] [--ref_rttm <参考 rttm 目录>]
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from common.run_helpers import (
    add_common_args,
    append_run_header,
    build_pipeline_cmd,
    env_flag,
)

_TEST_NAME = "der_test"  # 输出固定在 {output_root}/der_test/{run_name}


def _make_effective_config(config_path: str, exp_dir: Path) -> str:
    """生成 DER 评估用的生效配置：强制 separation_enabled=false。

    DER 只评估 diarization 的 RTTM 产物，不需要 TIGER 分段音频导出
    （节省导出耗时、避免分离模型的干扰）；其余调参项原样保留，
    生效配置落盘到 exp_dir 便于追溯。
    """

    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg = yaml.safe_load(file_obj) or {}
    cfg["separation_enabled"] = False
    effective_path = exp_dir / "config.effective.yaml"
    with open(effective_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(cfg, file_obj, allow_unicode=True, sort_keys=False)
    return str(effective_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时管线 DER 评估")
    parser.add_argument(
        "audio", nargs="?", default="./datasets/", help="输入音频（wav 文件或目录）"
    )
    add_common_args(parser)
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


def _read_summary_values(path: Path) -> dict[str, str]:
    values = {}
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                values[key] = value
    return values


def main() -> int:
    args = build_parser().parse_args()

    basic_dir = Path(args.output_root) / _TEST_NAME
    exp_dir = basic_dir / args.run_name
    results_file = basic_dir / "results.txt"

    # 与 run.py 一致：只清理本次 run 自己的输出目录，同 output_root 下的
    # 其他历史 run 目录与 results.txt 汇总记录保留。
    shutil.rmtree(exp_dir, ignore_errors=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Testing with YAML config")
    print(f"audio_input={args.audio}")
    print(f"config_path={args.config}")
    print(f"output_dir={exp_dir}")
    if args.model_path:
        print("model_path override enabled")
    if args.hf_cache_dir:
        print("hf_cache_dir override enabled")
    print("==========================================")

    # DER 只评估 RTTM：生效配置强制 separation_enabled=false（其余原样）。
    effective_config = _make_effective_config(args.config, exp_dir)

    # results.txt 按 run 追加：历史 run 的 DER 汇总记录保留。
    append_run_header(
        results_file,
        args.run_name,
        "DER test",
        {
            "audio_input": args.audio,
            "config_path": args.config,
            "effective_config": f"{effective_config} (separation_enabled=false)",
            "model_path_override": args.model_path,
            "hf_cache_dir_override": args.hf_cache_dir,
            "der_verbose": int(args.der_verbose),
            "ref_path": args.ref_rttm or None,
        },
    )

    cmd = build_pipeline_cmd(args, args.audio, exp_dir, effective_config)

    with open(exp_dir / "command.log", "w", encoding="utf-8") as log:
        log.write(f"Command: {shlex.join(cmd)}\n")
    print(f"Command: {shlex.join(cmd)}")

    pipeline_rc = subprocess.run(cmd).returncode
    if pipeline_rc != 0:
        return pipeline_rc

    rttm_count = len(list(exp_dir.glob("*.refined.rttm")))
    with open(results_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(
            f"{args.run_name} -> refined_rttm_files={rttm_count} | config={args.config}\n"
        )
    print(f"Result: {args.run_name} -> refined_rttm_files={rttm_count}")
    print(f"Pipeline log: {exp_dir}/run.log")

    print("\n========== DER Results ==========")
    if not args.ref_rttm:
        print("DER skipped: --ref_rttm not set")
    else:
        der_log = exp_dir / "der.log"
        der_summary = exp_dir / "der_summary.txt"
        with open(der_log, "w", encoding="utf-8") as log:
            if rttm_count == 0:
                message = (
                    f"{args.run_name} -> DER: SKIPPED | reason: no refined RTTM found"
                )
                print(message)
                log.write(message + "\n")
                with open(results_file, "a", encoding="utf-8") as file_obj:
                    file_obj.write(message + "\n")
            else:
                header = f"========== Computing DER for {args.run_name} =========="
                print(header)
                log.write(header + "\n")
                der_cmd = [
                    sys.executable,
                    "tools/compute_der.py",
                    "--sys",
                    str(exp_dir),
                    "--ref",
                    args.ref_rttm,
                    "--summary-file",
                    str(der_summary),
                    "--collar",
                    "0.0",
                    "--sys-suffix",
                    ".refined.rttm",
                    "--ref-suffix",
                    ".rttm",
                ]
                if args.der_verbose:
                    der_cmd.append("--verbose")
                proc = subprocess.run(der_cmd, capture_output=True, text=True)
                sys.stdout.write(proc.stdout)
                log.write(proc.stdout)
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    return proc.returncode
                if der_summary.is_file():
                    values = _read_summary_values(der_summary)
                    der = values.get("global_der") or values.get("avg_der", "NA")
                    files = values.get("global_files") or values.get("files", "0")
                    print(
                        f"{args.run_name} -> DER(global): {der}% over {files} file(s)"
                    )
                    with open(results_file, "a", encoding="utf-8") as file_obj:
                        file_obj.write(
                            f"{args.run_name} -> DER(global): {der}% | files={files}\n"
                        )

    print(f"\nResults saved to {results_file}")
    print(f"Outputs written to: {basic_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
