#!/usr/bin/env python3
"""DER 计算工具，依赖 common/speakerlab/md-eval.pl。

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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np

# 作为脚本/子进程运行时 sys.path[0] 是 tools/，补仓库根以 import separation。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from separation.rttm import load_rttm  # noqa: E402

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
    """全部匹配样本拼接后的全局 DER 评估结果。"""

    ms: float
    fa: float
    ser: float
    der: float
    files: int


def rectify(arr: np.ndarray) -> np.ndarray:
    """修正 corner case 并转换为百分比（NaN 归 0、inf 归 1，再乘 100）。

    Args:
        arr: 比例数组（原地修改）。

    Returns:
        百分比数组。
    """

    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 1
    arr *= 100.0
    return arr


def _resolve_md_eval_path() -> str:
    """查找项目内的 md-eval.pl。

    Returns:
        md-eval.pl 的绝对路径。

    Raises:
        FileNotFoundError: 项目内找不到 md-eval.pl。
    """

    possible_paths = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "common",
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
    collar: float = 0.0,
    ignore_overlap: bool = False,
) -> tuple[float, float, float, float]:
    """计算单对 RTTM 的 DER，返回 (MS, FA, SER, DER)。

    Args:
        ref_rttm: 参考 RTTM 路径。
        sys_rttm: 系统 RTTM 路径。
        collar: 宽容边界（秒）。
        ignore_overlap: 是否忽略重叠语音。

    Returns:
        (MS, FA, SER, DER)，均为百分比。

    Raises:
        RuntimeError: 找不到 perl、md-eval 执行失败或输出缺少统计行。
        FileNotFoundError: 项目内找不到 md-eval.pl。
    """

    md_eval_pl = _resolve_md_eval_path()
    perl = shutil.which("perl")
    if perl is None:
        raise RuntimeError("找不到 perl 解释器（md-eval.pl 依赖），请先安装 perl")
    cmd = [
        perl,
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
        # md-eval 失败时其输出不含统计行，继续解析只会得到假 DER，直接报错。
        excerpt = ex.output.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(
            f"md-eval 执行失败（rc={ex.returncode}）: ref={ref_rttm} "
            f"sys={sys_rttm}\n{excerpt}"
        ) from ex

    text = stdout.decode("utf-8")
    scored_speaker_times = np.array(
        [float(m) for m in SCORED_SPEAKER_TIME.findall(text)]
    )
    miss_speaker_times = np.array([float(m) for m in MISS_SPEAKER_TIME.findall(text)])
    fa_speaker_times = np.array([float(m) for m in FA_SPEAKER_TIME.findall(text)])
    error_speaker_times = np.array([float(m) for m in ERROR_SPEAKER_TIME.findall(text)])

    if scored_speaker_times.size == 0:
        # md-eval 退出码为 0 但输出无统计行（如输入为空/格式错误），
        # 此时 [-1] 取值会得到裸 IndexError，报出可诊断的错误。
        raise RuntimeError(
            f"md-eval 输出缺少统计行: ref={ref_rttm} sys={sys_rttm}\n"
            f"{text[-500:]}"
        )

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
    """分析 RTTM 文件并返回基础统计信息。

    解析复用 separation.rttm.load_rttm（仅 SPEAKER 行参与统计）。

    Args:
        rttm_file: RTTM 文件路径。

    Returns:
        统计信息（说话人数/段数/总时长/各说话人时长）；文件不存在返回 None。
    """

    if not os.path.exists(rttm_file):
        return None

    turns = load_rttm(rttm_file)
    speaker_durations = {
        speaker: sum(end - start for start, end in segments)
        for speaker, segments in turns.items()
    }
    return {
        "num_speakers": len(turns),
        "num_segments": sum(len(segments) for segments in turns.values()),
        "total_duration": sum(speaker_durations.values()),
        "speakers": sorted(turns),
        "speaker_durations": speaker_durations,
    }


def _print_rttm_stats(title: str, rttm_path: str) -> None:
    """打印单个 RTTM 的统计信息。

    Args:
        title: 输出标题。
        rttm_path: RTTM 文件路径。
    """

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
    """打印单个样本的详细信息，兼容单文件和批量模式。

    Args:
        item: 单样本 DER 结果。
        index: 当前序号（从 1 起）。
        total: 样本总数。
    """

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
    """收集单文件或目录下 RTTM 文件。

    Args:
        path_value: RTTM 文件或目录路径。
        suffix: 目录模式下按文件名后缀筛选。
        label: 错误信息中的路径角色（如 sys/ref）。

    Returns:
        RTTM 文件路径列表（目录模式按文件名排序）。

    Raises:
        FileNotFoundError: 路径不存在或目录下无匹配文件。
        ValueError: 路径既非文件也非目录。
    """

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
    """为单个系统 RTTM 匹配参考 RTTM。

    Args:
        sys_rttm: 系统 RTTM 路径。
        ref_path: 参考 RTTM 文件或目录路径。
        sys_count: 系统 RTTM 总数（ref 为单文件时要求为 1）。
        sys_suffix: 系统 RTTM 后缀（剥掉后得到 stem）。
        ref_suffix: 参考 RTTM 后缀。

    Returns:
        (匹配的参考 RTTM 路径, None)；匹配失败时为 (None, 原因)。
    """

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
    """把多个 RTTM 顺序拼接为一个文件（用于全局 DER）。

    Args:
        paths: 待拼接的 RTTM 路径列表。
        concat_path: 拼接输出路径。
    """

    with open(concat_path, "w", encoding="utf-8") as out_f:
        for p in paths:
            with open(p, "r", encoding="utf-8") as in_f:
                content = in_f.read()
            out_f.write(content)
            # 文件末尾缺换行时补齐，否则下一文件的首行与之粘连，
            # 粘连行会被 md-eval 静默忽略，全局 DER 被低估。
            if content and not content.endswith("\n"):
                out_f.write("\n")


def compute_der_batch(
    *,
    ref_path: str,
    sys_path: str,
    collar: float = 0.0,
    ignore_overlap: bool = False,
    sys_suffix: str = ".refined.rttm",
    ref_suffix: str = ".rttm",
) -> tuple[list[DerResult], list[tuple[str, str]], Optional[GlobalDerResult]]:
    """批量计算 RTTM DER。

    Args:
        ref_path: 参考 RTTM 文件或目录路径。
        sys_path: 系统 RTTM 文件或目录路径。
        collar: 宽容边界（秒）。
        ignore_overlap: 是否忽略重叠语音。
        sys_suffix: 批量模式下系统 RTTM 文件后缀。
        ref_suffix: 批量模式下参考 RTTM 文件后缀。

    Returns:
        `(results, skipped, global_result)`：

        - results: 每个成功样本的详细结果
        - skipped: `(filename, reason)` 列表
        - global_result: 拼接全部匹配样本后算出的全局 DER（无匹配样本时为 None）
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

        ref_stats = analyze_rttm(current_ref)
        if ref_stats is not None and ref_stats["num_segments"] == 0:
            # 空参考标注会算出 DER=0%（rectify 把 NaN 归 0），掩盖数据问题。
            print(f"WARNING: 参考标注无有效语音段: {current_ref}")

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
    """把批量 DER 摘要写成简单 key=value 文件，便于 shell 直接读取。

    Args:
        summary_file: 摘要输出路径。
        results: 每个成功样本的详细结果。
        skipped: `(filename, reason)` 列表。
        global_result: 全局 DER；无匹配样本时各字段写 NA。
    """

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
        file_obj.write(f"global_files={global_files}\n")
        file_obj.write(f"global_ms={global_ms}\n")
        file_obj.write(f"global_fa={global_fa}\n")
        file_obj.write(f"global_ser={global_ser}\n")
        file_obj.write(f"global_der={global_der}\n")


def main() -> int:
    """CLI 入口：单文件或批量评估 DER。

    Returns:
        退出码（0 正常；1 评估出错、全部被跳过或全局 DER 非有限值）。
    """

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
        help="批量模式下系统 RTTM 文件后缀，默认 .rttm",
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
        if global_result:
            print(
                f"GLOBAL\tfiles={int(global_result['files'])}\t"
                f"MS={float(global_result['ms']):.2f}\t"
                f"FA={float(global_result['fa']):.2f}\t"
                f"SER={float(global_result['ser']):.2f}\t"
                f"DER={float(global_result['der']):.2f}"
            )
        else:
            print("GLOBAL\tfiles=0\tDER=NA")

    if skipped:
        print("")
        print("Skipped files:")
        for filename, reason in skipped:
            print(f"  {filename}: {reason}")

    if args.summary_file:
        _write_summary(args.summary_file, results, skipped, global_result)
        print(f"\n摘要已保存到: {args.summary_file}")

    if not results:
        # 一个样本都没评上（全部 skip）时返回非零，避免 sweep/CI 把
        # "没评上" 当成评估成功。
        if skipped:
            print("\n错误: 全部文件被跳过，未产生任何 DER 结果")
            return 1
        return 0
    # results 非空必有匹配样本，global_result 一定已算出。
    assert global_result is not None
    return 0 if np.isfinite(float(global_result["der"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
