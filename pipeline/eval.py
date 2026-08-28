"""管线产物的 DER 评估：refined RTTM 目录 + 参考 RTTM → 逐文件 + 全局 DER。

封装 tools/compute_der.py 的子进程调用与结果回写（der.log / der_summary.txt /
results.txt），供 test_der.py 等评估入口复用。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _read_summary_values(path: Path) -> dict[str, str]:
    values = {}
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                values[key] = value
    return values


def run_der_evaluation(
    exp_dir: Path,
    ref_rttm: str,
    run_name: str,
    der_verbose: bool,
    results_file: Path,
) -> int:
    """对 exp_dir/rttm 下的 refined RTTM 计算 DER（逐文件 + 拼接全局）。

    返回退出码：0 成功或无可评文件；非 0 为 compute_der 子进程失败。
    """

    rttm_count = len(list((exp_dir / "rttm").glob("*.refined.rttm")))
    der_log = exp_dir / "logs" / "der.log"
    der_summary = exp_dir / "der_summary.txt"
    with open(der_log, "w", encoding="utf-8") as log:
        if rttm_count == 0:
            message = f"{run_name} -> DER: SKIPPED | reason: no refined RTTM found"
            print(message)
            log.write(message + "\n")
            with open(results_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(message + "\n")
            return 0
        header = f"========== Computing DER for {run_name} =========="
        print(header)
        log.write(header + "\n")
        der_cmd = [
            sys.executable,
            "tools/compute_der.py",
            "--sys",
            str(exp_dir / "rttm"),
            "--ref",
            ref_rttm,
            "--summary-file",
            str(der_summary),
            "--collar",
            "0.0",
            "--sys-suffix",
            ".refined.rttm",
            "--ref-suffix",
            ".rttm",
        ]
        if der_verbose:
            der_cmd.append("--verbose")
        proc = subprocess.run(der_cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        log.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            return proc.returncode
        if der_summary.is_file():
            values = _read_summary_values(der_summary)
            der = values.get("global_der", "NA")
            files = values.get("global_files", "0")
            print(f"{run_name} -> DER(global): {der}% over {files} file(s)")
            with open(results_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"{run_name} -> DER(global): {der}% | files={files}\n")
    return 0


__all__ = ["run_der_evaluation"]
