#!/usr/bin/env python3
"""DER 计算工具，依赖 speakerlab/md-eval.pl。

使用方法：
1) 单文件评估：
   python tools/compute_der.py --ref /path/to/ref.rttm --sys /path/to/sys.rttm

2) 目录批量评估（先逐文件 DER，再计算一次全局 DER）：
   python tools/compute_der.py --ref /path/to/ref_dir --sys /path/to/sys_dir \
     --sys-suffix .refined.rttm --ref-suffix .rttm

3) 导出摘要文件：
   python tools/compute_der.py --ref /path/to/ref_dir --sys /path/to/sys_dir \
     --summary-file /tmp/der_summary.txt
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np

# RTTM 解析正则
SCORED_SPEAKER_TIME = re.compile(r"(?<=SCORED SPEAKER TIME =)[\d.]+")
MISS_SPEAKER_TIME = re.compile(r"(?<=MISSED SPEAKER TIME =)[\d.]+")
FA_SPEAKER_TIME = re.compile(r"(?<=FALARM SPEAKER TIME =)[\d.]+")
ERROR_SPEAKER_TIME = re.compile(r"(?<=SPEAKER ERROR TIME =)[\d.]+")


class RTTMStats(TypedDict):
    """单个 RTTM 文件的基础统计信息。"""

    num_speakers: int
    num_segments: int
    total_duration: float
    speakers: list[str]
    speaker_durations: dict[str, float]


class DerResult(TypedDict):
    """单个系统 RTTM 的 DER 评估结果。"""

    filename: str
    ref_rttm: str
    sys_rttm: str
    ms: float
    fa: float
    ser: float
    der: float


class GlobalDerResult(TypedDict):
    ms: float
    fa: float
    ser: float
    der: float
    files: int


def rectify(arr: np.ndarray) -> np.ndarray:
    """修正 corner case 并转换为百分比。"""

    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 1
    arr *= 100.0
    return arr


def _resolve_md_eval_path() -> str:
    """查找项目内的 md-eval.pl。"""

    possible_paths = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "speakerlab",
            "md-eval.pl",
        ),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("找不到 md-eval.pl 脚本，请确保项目完整")


def compute_der(
    ref_rttm: str,
    sys_rttm: str,
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> tuple[float, float, float, float]:
    """计算单对 RTTM 的 DER，返回 (MS, FA, SER, DER)。"""

    md_eval_pl = _resolve_md_eval_path()
    cmd = [
        "perl",
        md_eval_pl,
        "-af",
        "-r",
        ref_rttm,
        "-s",
        sys_rttm,
        "-c",
        str(collar),
    ]
    if ignore_overlap:
        cmd.append("-1")

    try:
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as ex:
        stdout = ex.output

    text = stdout.decode("utf-8")
    scored_speaker_times = np.array(
        [float(m) for m in SCORED_SPEAKER_TIME.findall(text)]
    )
    miss_speaker_times = np.array([float(m) for m in MISS_SPEAKER_TIME.findall(text)])
    fa_speaker_times = np.array([float(m) for m in FA_SPEAKER_TIME.findall(text)])
    error_speaker_times = np.array([float(m) for m in ERROR_SPEAKER_TIME.findall(text)])

    with np.errstate(invalid="ignore", divide="ignore"):
        tot_error_times = miss_speaker_times + fa_speaker_times + error_speaker_times
        miss_speaker_frac = miss_speaker_times / scored_speaker_times
        fa_speaker_frac = fa_speaker_times / scored_speaker_times
        sers_frac = error_speaker_times / scored_speaker_times
        ders_frac = tot_error_times / scored_speaker_times

    miss_speaker = rectify(miss_speaker_frac)
    fa_speaker = rectify(fa_speaker_frac)
    sers = rectify(sers_frac)
    ders = rectify(ders_frac)
    return miss_speaker[-1], fa_speaker[-1], sers[-1], ders[-1]


def analyze_rttm(rttm_file: str) -> Optional[RTTMStats]:
    """分析 RTTM 文件并返回基础统计信息。"""

    if not os.path.exists(rttm_file):
        return None

    speakers: set[str] = set()
    total_duration = 0.0
    segments: list[tuple[float, str]] = []

    with open(rttm_file, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                duration = float(parts[4])
                speaker = parts[7]
            except (ValueError, IndexError):
                continue

            speakers.add(speaker)
            total_duration += duration
            segments.append((duration, speaker))

    speaker_durations: dict[str, float] = {}
    for duration, speaker in segments:
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration

    return {
        "num_speakers": len(speakers),
        "num_segments": len(segments),
        "total_duration": total_duration,
        "speakers": sorted(speakers),
        "speaker_durations": speaker_durations,
    }


def _print_rttm_stats(title: str, rttm_path: str) -> None:
    """打印单个 RTTM 的统计信息。"""

    stats = analyze_rttm(rttm_path)
    print("=" * 60)
    print(title)
    print("=" * 60)
    if not stats:
        print(f"  无法读取 RTTM: {rttm_path}")
        print("")
        return

    print(f"  说话人数量: {stats['num_speakers']}")
    print(f"  语音段数量: {stats['num_segments']}")
    print(f"  总语音时长: {stats['total_duration']:.2f}s")
    print("  说话人时长分布:")
    total_duration = float(stats["total_duration"])
    for speaker in stats["speakers"]:
        duration = float(stats["speaker_durations"][speaker])
        ratio = 0.0 if total_duration <= 0 else duration / total_duration * 100.0
        print(f"    {speaker}: {duration:.2f}s ({ratio:.1f}%)")
    print("")


def _print_verbose_result(item: DerResult, index: int, total: int) -> None:
    """打印单个样本的详细信息，兼容单文件和批量模式。"""

    print("=" * 60)
    print(f"详细结果 [{index}/{total}]: {item['filename']}")
    print("=" * 60)
    print(f"参考 RTTM: {item['ref_rttm']}")
    print(f"系统 RTTM: {item['sys_rttm']}")
    print("")
    _print_rttm_stats("参考 RTTM 统计:", str(item["ref_rttm"]))
    _print_rttm_stats("系统输出 RTTM 统计:", str(item["sys_rttm"]))
    print("  DER 明细:")
    print(f"    Missed Speech (MS):  {float(item['ms']):>6.2f}%")
    print(f"    False Alarm (FA):    {float(item['fa']):>6.2f}%")
    print(f"    Speaker Error (SER): {float(item['ser']):>6.2f}%")
    print(f"    Total DER:           {float(item['der']):>6.2f}%")
    print("")


def _collect_rttm_files(path_value: str, suffix: str, *, label: str) -> list[str]:
    """收集单文件或目录下 RTTM 文件。"""

    if not os.path.exists(path_value):
        raise FileNotFoundError(f"找不到 {label.upper()} 路径: {path_value}")

    if os.path.isfile(path_value):
        return [path_value]

    root = Path(path_value)
    if not root.is_dir():
        raise ValueError(f"无法识别 {label.upper()} 路径类型: {path_value}")

    files = sorted(str(path) for path in root.iterdir() if path.name.endswith(suffix))
    if not files:
        raise FileNotFoundError(
            f"目录中没有匹配 {suffix} 的 {label.upper()} RTTM: {path_value}"
        )
    return files


def _match_reference(
    sys_rttm: str,
    *,
    ref_path: str,
    sys_count: int,
    sys_suffix: str,
    ref_suffix: str,
) -> tuple[Optional[str], Optional[str]]:
    """为单个系统 RTTM 匹配参考 RTTM。"""

    filename = os.path.basename(sys_rttm)
    stem = (
        filename[: -len(sys_suffix)]
        if filename.endswith(sys_suffix)
        else Path(filename).stem
    )

    if os.path.isfile(ref_path):
        if sys_count != 1:
            return None, "single REF RTTM provided but multiple system RTTMs found"
        return ref_path, None

    if not os.path.isdir(ref_path):
        return None, f"invalid reference path: {ref_path}"

    candidate_ref = os.path.join(ref_path, f"{stem}{ref_suffix}")
    if not os.path.exists(candidate_ref):
        return None, f"missing reference: {candidate_ref}"
    return candidate_ref, None


def _concat_rttm_files(paths: list[str], concat_path: str) -> None:
    with open(concat_path, "w", encoding="utf-8") as out_f:
        for p in paths:
            with open(p, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())


def compute_der_batch(
    *,
    ref_path: str,
    sys_path: str,
    collar: float = 0.25,
    ignore_overlap: bool = False,
    sys_suffix: str = ".refined.rttm",
    ref_suffix: str = ".rttm",
) -> tuple[list[DerResult], list[tuple[str, str]], Optional[GlobalDerResult]]:
    """批量计算 RTTM DER。

    返回：
    - results: 每个成功样本的详细结果
    - skipped: `(filename, reason)` 列表
    """

    sys_rttms = _collect_rttm_files(sys_path, sys_suffix, label="sys")
    results: list[DerResult] = []
    skipped: list[tuple[str, str]] = []

    matched_ref_paths: list[str] = []
    matched_sys_paths: list[str] = []

    for current_sys in sys_rttms:
        filename = os.path.basename(current_sys)
        current_ref, reason = _match_reference(
            current_sys,
            ref_path=ref_path,
            sys_count=len(sys_rttms),
            sys_suffix=sys_suffix,
            ref_suffix=ref_suffix,
        )
        if current_ref is None:
            skipped.append((filename, str(reason)))
            continue

        ms, fa, ser, der = compute_der(
            current_ref,
            current_sys,
            collar=collar,
            ignore_overlap=ignore_overlap,
        )
        results.append(
            {
                "filename": filename,
                "ref_rttm": current_ref,
                "sys_rttm": current_sys,
                "ms": ms,
                "fa": fa,
                "ser": ser,
                "der": der,
            }
        )
        matched_ref_paths.append(current_ref)
        matched_sys_paths.append(current_sys)

    global_result: Optional[GlobalDerResult] = None
    if matched_ref_paths and matched_sys_paths:
        with tempfile.TemporaryDirectory(prefix="der_eval_") as tmp_dir:
            concat_ref = os.path.join(tmp_dir, "concat_ref_rttm")
            concat_sys = os.path.join(tmp_dir, "concat_sys_rttm")
            _concat_rttm_files(matched_ref_paths, concat_ref)
            _concat_rttm_files(matched_sys_paths, concat_sys)
            g_ms, g_fa, g_ser, g_der = compute_der(
                concat_ref,
                concat_sys,
                collar=collar,
                ignore_overlap=ignore_overlap,
            )
            global_result = {
                "ms": g_ms,
                "fa": g_fa,
                "ser": g_ser,
                "der": g_der,
                "files": len(matched_ref_paths),
            }

    return results, skipped, global_result


def _write_summary(
    summary_file: str,
    results: list[DerResult],
    skipped: list[tuple[str, str]],
    global_result: Optional[GlobalDerResult],
) -> None:
    """把批量 DER 摘要写成简单 key=value 文件，便于 shell 直接读取。"""

    avg_der = "NA"
    avg_ms = "NA"
    avg_fa = "NA"
    avg_ser = "NA"
    if results:
        avg_der = f"{sum(float(item['der']) for item in results) / len(results):.4f}"
        avg_ms = f"{sum(float(item['ms']) for item in results) / len(results):.4f}"
        avg_fa = f"{sum(float(item['fa']) for item in results) / len(results):.4f}"
        avg_ser = f"{sum(float(item['ser']) for item in results) / len(results):.4f}"

    global_ms = "NA"
    global_fa = "NA"
    global_ser = "NA"
    global_der = "NA"
    global_files = "0"
    if global_result:
        global_ms = f"{float(global_result['ms']):.4f}"
        global_fa = f"{float(global_result['fa']):.4f}"
        global_ser = f"{float(global_result['ser']):.4f}"
        global_der = f"{float(global_result['der']):.4f}"
        global_files = str(int(global_result["files"]))

    with open(summary_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"files={len(results)}\n")
        file_obj.write(f"skipped={len(skipped)}\n")
        file_obj.write(f"avg_ms={avg_ms}\n")
        file_obj.write(f"avg_fa={avg_fa}\n")
        file_obj.write(f"avg_ser={avg_ser}\n")
        file_obj.write(f"avg_der={avg_der}\n")
        file_obj.write(f"global_files={global_files}\n")
        file_obj.write(f"global_ms={global_ms}\n")
        file_obj.write(f"global_fa={global_fa}\n")
        file_obj.write(f"global_ser={global_ser}\n")
        file_obj.write(f"global_der={global_der}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="计算说话人分割 DER")
    parser.add_argument(
        "--ref", required=True, help="参考 RTTM 路径，可传单个文件或目录"
    )
    parser.add_argument(
        "--sys", required=True, help="系统 RTTM 路径，可传单个文件或目录"
    )
    parser.add_argument(
        "--sys-suffix",
        default=".rttm",
        help="批量模式下系统 RTTM 文件后缀，默认 .refined.rttm",
    )
    parser.add_argument(
        "--ref-suffix",
        default=".rttm",
        help="批量模式下参考 RTTM 文件后缀，默认 .rttm",
    )
    parser.add_argument("--summary-file", help="把批量评估摘要写到该文件")
    parser.add_argument("--collar", type=float, default=0.0, help="宽容边界 (秒)")
    parser.add_argument("--ignore-overlap", action="store_true", help="忽略重叠语音")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    try:
        results, skipped, global_result = compute_der_batch(
            ref_path=args.ref,
            sys_path=args.sys,
            collar=args.collar,
            ignore_overlap=args.ignore_overlap,
            sys_suffix=args.sys_suffix,
            ref_suffix=args.ref_suffix,
        )
    except Exception as exc:
        print(f"计算 DER 时出错: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    if args.verbose and results:
        for index, item in enumerate(results, start=1):
            _print_verbose_result(item, index, len(results))

    print("=" * 60)
    print(f"DER 评估结果 (collar = {args.collar}s)")
    print("=" * 60)

    if len(results) == 1 and not skipped:
        item = results[0]
        print(f"  Missed Speech (MS):  {float(item['ms']):>6.2f}%")
        print(f"  False Alarm (FA):    {float(item['fa']):>6.2f}%")
        print(f"  Speaker Error (SER): {float(item['ser']):>6.2f}%")
        print("-" * 60)
        print(f"  Total DER:           {float(item['der']):>6.2f}%")
    else:
        for item in results:
            print(
                f"{item['filename']}\tMS={float(item['ms']):.2f}\tFA={float(item['fa']):.2f}"
                f"\tSER={float(item['ser']):.2f}\tDER={float(item['der']):.2f}"
            )
        print("-" * 60)
        if results:
            avg_der = sum(float(item["der"]) for item in results) / len(results)
            print(f"AVERAGE\tfiles={len(results)}\tDER={avg_der:.2f}")
            if global_result:
                print(
                    f"GLOBAL\tfiles={int(global_result['files'])}\t"
                    f"MS={float(global_result['ms']):.2f}\t"
                    f"FA={float(global_result['fa']):.2f}\t"
                    f"SER={float(global_result['ser']):.2f}\t"
                    f"DER={float(global_result['der']):.2f}"
                )
        else:
            print("AVERAGE\tfiles=0\tDER=NA")

    if skipped:
        print("")
        print("Skipped files:")
        for filename, reason in skipped:
            print(f"  {filename}: {reason}")

    if args.summary_file:
        _write_summary(args.summary_file, results, skipped, global_result)
        print(f"\n摘要已保存到: {args.summary_file}")
    elif os.path.isfile(args.sys):
        output_dir = os.path.dirname(args.sys)
        if output_dir:
            result_file = os.path.join(output_dir, "der_result.txt")
            _write_summary(result_file, results, skipped, global_result)
            print(f"\n结果已保存到: {result_file}")

    if global_result:
        return 0 if np.isfinite(float(global_result["der"])) else 1
    if results:
        avg_der = sum(float(item["der"]) for item in results) / len(results)
        return 0 if np.isfinite(avg_der) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
