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

    python3 test_der.py [audio_input] [--ref_rttm datasets/任务1-6/rttm/]
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_TEST_NAME = "der_test"  # 输出固定在 {output_root}/der_test/{run_name}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw == "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时管线 DER 评估")
    parser.add_argument("audio", nargs="?", default="./datasets/任务1-6/",
                        help="输入音频（wav 文件或目录）")
    parser.add_argument("--config", default=os.environ.get("CONFIG_PATH", "./config/config.yaml"))
    parser.add_argument("--output_root", default=os.environ.get("OUTPUT_ROOT", "./exp"))
    parser.add_argument("--run_name", default=os.environ.get("RUN_NAME", "default"))
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH") or None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN") or None)
    parser.add_argument("--hf_cache_dir", default=os.environ.get("HF_CACHE_DIR") or None)
    parser.add_argument("--ref_rttm",
                        default=os.environ.get("REF_RTTM")
                        or os.environ.get("REF_RTTM_DIR")
                        or "./datasets/任务1-6/rttm/",
                        help="参考 RTTM 目录；置空则跳过 DER 计算")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction,
                        default=_env_flag("DEBUG", True))
    parser.add_argument("--show_rttm", action=argparse.BooleanOptionalAction,
                        default=_env_flag("SHOW_RTTM", False))
    parser.add_argument("--der_verbose", action=argparse.BooleanOptionalAction,
                        default=_env_flag("DER_VERBOSE", True))
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

    # 与原 test_der.sh 一致：每次运行清空 der_test 目录重建。
    shutil.rmtree(basic_dir, ignore_errors=True)
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

    with open(results_file, "w", encoding="utf-8") as file_obj:
        file_obj.write("DER test\n" + "=" * 40 + "\n")
        file_obj.write(f"audio_input: {args.audio}\nconfig_path: {args.config}\n")
        if args.model_path:
            file_obj.write(f"model_path_override: {args.model_path}\n")
        if args.hf_cache_dir:
            file_obj.write(f"hf_cache_dir_override: {args.hf_cache_dir}\n")
        file_obj.write(f"run_name: {args.run_name}\nder_verbose: {int(args.der_verbose)}\n")
        if args.ref_rttm:
            file_obj.write(f"ref_path: {args.ref_rttm}\n")
        file_obj.write("\n")

    cmd = [sys.executable, "-m", "diarization.app",
           "--wav", args.audio, "--output_dir", str(exp_dir), "--config", args.config]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    if args.hf_cache_dir:
        cmd += ["--hf_cache_dir", args.hf_cache_dir]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    if args.debug:
        cmd += ["--debug", "--verbose"]
    if args.show_rttm:
        cmd += ["--show_rttm"]

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
                message = f"{args.run_name} -> DER: SKIPPED | reason: no refined RTTM found"
                print(message)
                log.write(message + "\n")
                with open(results_file, "a", encoding="utf-8") as file_obj:
                    file_obj.write(message + "\n")
            else:
                header = f"========== Computing DER for {args.run_name} =========="
                print(header)
                log.write(header + "\n")
                der_cmd = [sys.executable, "tools/compute_der.py",
                           "--sys", str(exp_dir), "--ref", args.ref_rttm,
                           "--summary-file", str(der_summary),
                           "--collar", "0.0",
                           "--sys-suffix", ".refined.rttm",
                           "--ref-suffix", ".rttm"]
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
                    print(f"{args.run_name} -> DER(global): {der}% over {files} file(s)")
                    with open(results_file, "a", encoding="utf-8") as file_obj:
                        file_obj.write(f"{args.run_name} -> DER(global): {der}% | files={files}\n")

    print(f"\nResults saved to {results_file}")
    print(f"Outputs written to: {basic_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
