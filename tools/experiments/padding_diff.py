#!/usr/bin/env python3
"""E7 padding 泄漏量化：同一批 observation 在两套提取缓存间的 embedding 差分。

背景：`NativeERes2NetV2SegmentEmbedder.embed_segments` 把批内特征零填充到
最长帧，而 ERes2NetV2 的 TSTP 池化无长度掩码——同一段音频的 embedding 随
batch 队友变化。本脚本对齐两套缓存（同一音频、同一分割/track 逻辑，仅
segment_batch_size 不同）的同名 observation，量化 padding 造成的扰动：

1. 余弦距离分布（按 observation 时长分桶）：扰动随 padding 比例增大，
   短段应最严重；
2. 最近质心翻转率：以参考 RTTM 映射说话人、用 base 缓存建说话人质心，
   统计 embedding 换到对照缓存后最近质心易主的比例（扰动是否大到足以
   改变聚类决策的直接度量）。

建议同时跑一个"同配置重复提取"的对照组（bs1 vs bs1-repeat）：GPU 非确定性
本身有噪声地板，只有显著高于地板的距离才能归因于 padding。

用法：
  .venv/bin/python tools/experiments/padding_diff.py \
      --base exp/experiments/e7_padding_leak/bs1/embeddings \
      --other bs4=exp/experiments/e7_padding_leak/bs4/embeddings \
      --other bs1b=exp/experiments/e7_padding_leak/bs1b/embeddings \
      --rttm datasets/task1-6/rttm \
      --output exp/experiments/e7_padding_leak/padding_diff.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analyze_embeddings import load_rttm, map_to_ref  # noqa: E402

# 时长分桶：与 analyze_embeddings 同口径，但把 max_segment_duration_for_embedding
# 截断点（4.0s）单独成桶——恰好 4.0s 的 observation 恒为批内最长，从不被
# padding，理论距离为 0，是天然的对照组。
DUR_BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.99), (3.99, np.inf)]
DUR_LABELS = ["<0.5s", "0.5-1s", "1-2s", "2-3.99s", "=4s截断"]


def load_obs(npz_path: Path) -> dict[tuple, tuple[np.ndarray, float, float, float]]:
    """读一套缓存：key=(chunk_index, local_idx, start, end)（毫秒精度取整）。

    只保留 has_embedding 的 observation；embedding 读入后 L2 归一化。

    Args:
        npz_path: chunks.npz 路径。

    Returns:
        {key: (embedding, duration, start, end)} 字典。
    """

    data = np.load(npz_path, allow_pickle=False)
    keys = zip(
        data["chunk_index"].tolist(),
        data["local_idx"].tolist(),
        np.round(data["start"], 3).tolist(),
        np.round(data["end"], 3).tolist(),
    )
    obs: dict[tuple, tuple[np.ndarray, float, float, float]] = {}
    for row, key in enumerate(keys):
        if not bool(data["has_embedding"][row]):
            continue
        emb = data["embeddings"][row].astype(np.float64)
        norm = max(np.linalg.norm(emb), 1e-12)
        obs[key] = (
            emb / norm,
            float(data["duration"][row]),
            float(data["start"][row]),
            float(data["end"][row]),
        )
    return obs


def bucket_of(duration: float) -> int:
    """返回 observation 时长所属分桶下标（超长归入最后一桶）。"""

    for i, (lo, hi) in enumerate(DUR_BUCKETS):
        if lo <= duration < hi:
            return i
    return len(DUR_BUCKETS) - 1


def diff_pair(
    base: dict, other: dict, segs, spk_index
) -> tuple[list[float], list[int], int, int, int]:
    """对齐两套缓存的 observation，返回 (逐对余弦距离, 时长桶, 翻转数, 可判翻转数, 对齐数)。

    Args:
        base: 基准缓存（load_obs 返回），翻转判定的质心由它建立。
        other: 对照缓存（load_obs 返回）。
        segs: 参考语音段 [(start, end, spk)]。
        spk_index: 说话人名 -> 下标。

    Returns:
        (逐对余弦距离列表, 时长桶下标列表, 翻转数, 可判翻转数, 对齐数)。
    """

    dists: list[float] = []
    buckets: list[int] = []
    flips = 0
    flippable = 0
    aligned_keys = sorted(set(base) & set(other))
    # 用 base 的全部 observation 建参考说话人质心（翻转判定的基准）。
    cents: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for key in aligned_keys:
        emb, _, start, end = base[key]
        label = map_to_ref(start, end, segs, spk_index)
        if label < 0:
            continue
        cents[label] = cents.get(label, np.zeros_like(emb)) + emb
        counts[label] = counts.get(label, 0) + 1
    for label in cents:
        cents[label] /= max(np.linalg.norm(cents[label]), 1e-12)
    cent_labels = sorted(cents)
    cent_mat = np.stack([cents[k] for k in cent_labels]) if cent_labels else None

    for key in aligned_keys:
        emb_b, dur, start, end = base[key]
        emb_o = other[key][0]
        dists.append(float(1.0 - emb_b @ emb_o))
        buckets.append(bucket_of(dur))
        # 最近质心翻转：>=2 个质心才可谈翻转。
        if cent_mat is not None and len(cent_labels) >= 2:
            flippable += 1
            nb = cent_labels[int(np.argmax(cent_mat @ emb_b))]
            no = cent_labels[int(np.argmax(cent_mat @ emb_o))]
            if nb != no:
                flips += 1
    return dists, buckets, flips, flippable, len(aligned_keys)


def main() -> None:
    """CLI 入口：逐文件对齐差分并输出汇总报告。"""

    parser = argparse.ArgumentParser(description="两套 chunks.npz 缓存的 embedding 差分")
    parser.add_argument("--base", required=True, help="基准缓存目录（如 bs=1）")
    parser.add_argument(
        "--other",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="对照缓存目录（可重复），如 bs4=exp/.../bs4/embeddings",
    )
    parser.add_argument("--rttm", required=True, help="参考 RTTM 目录（翻转分析用）")
    parser.add_argument("--output", required=True, help="报告输出路径（txt）")
    args = parser.parse_args()

    base_dir = Path(args.base)
    others = []
    for item in args.other:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--other 需要 NAME=DIR 形式，收到: {item}")
        others.append((name, Path(path)))

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    npz_files = sorted(base_dir.glob("*.chunks.npz"))
    if not npz_files:
        raise SystemExit(f"no chunks.npz under {base_dir}")

    # name -> 全局汇总（跨文件拼接）
    summary: dict[str, dict[str, list]] = {
        name: {"dists": [], "buckets": [], "flips": 0, "flippable": 0,
               "aligned": 0, "only_base": 0, "only_other": 0}
        for name, _ in others
    }

    for npz_path in npz_files:
        uri = npz_path.name[: -len(".chunks.npz")]
        rttm_path = Path(args.rttm) / f"{uri}.rttm"
        if not rttm_path.is_file():
            log(f"skip {uri}: ref RTTM not found: {rttm_path}")
            continue
        segs, spks = load_rttm(rttm_path)
        spk_index = {s: i for i, s in enumerate(spks)}
        base = load_obs(npz_path)
        log(f"=== File {uri} (base obs: {len(base)}) ===")
        for name, other_dir in others:
            other_path = other_dir / npz_path.name
            if not other_path.is_file():
                log(f"  {name}: missing {other_path}")
                continue
            other = load_obs(other_path)
            dists, buckets, flips, flippable, n_aligned = diff_pair(
                base, other, segs, spk_index
            )
            only_base = len(set(base) - set(other))
            only_other = len(set(other) - set(base))
            agg = summary[name]
            agg["dists"].extend(dists)
            agg["buckets"].extend(buckets)
            agg["flips"] += flips
            agg["flippable"] += flippable
            agg["aligned"] += n_aligned
            agg["only_base"] += only_base
            agg["only_other"] += only_other
            d = np.array(dists)
            log(
                f"  {name}: aligned={n_aligned} only_base={only_base} "
                f"only_other={only_other} | cos-dist mean={d.mean():.4f} "
                f"p50={np.median(d):.4f} p90={np.quantile(d, 0.9):.4f} "
                f"max={d.max():.4f} | centroid-flip {flips}/{flippable} "
                f"({100 * flips / max(flippable, 1):.2f}%)"
            )
        log()

    # ---------- 汇总：分桶距离分布 + 翻转率 ----------
    log("=== Summary: cosine distance by duration bucket ===")
    header = f"{'compare':>8} {'bucket':>8} {'n':>6} {'mean':>8} {'p50':>8} {'p90':>8} {'max':>8}"
    log(header)
    for name, _ in others:
        agg = summary[name]
        d = np.array(agg["dists"])
        b = np.array(agg["buckets"])
        for i, label in enumerate(DUR_LABELS):
            m = b == i
            if not m.any():
                continue
            dm = d[m]
            log(
                f"{name:>8} {label:>8} {int(m.sum()):>6} {dm.mean():>8.4f} "
                f"{np.median(dm):>8.4f} {np.quantile(dm, 0.9):>8.4f} {dm.max():>8.4f}"
            )
        log(
            f"{name:>8} {'ALL':>8} {len(d):>6} {d.mean():>8.4f} "
            f"{np.median(d):>8.4f} {np.quantile(d, 0.9):>8.4f} {d.max():>8.4f}"
        )
    log()
    for name, _ in others:
        agg = summary[name]
        log(
            f"{name}: aligned={agg['aligned']} only_base={agg['only_base']} "
            f"only_other={agg['only_other']} | centroid-flip {agg['flips']}/"
            f"{agg['flippable']} ({100 * agg['flips'] / max(agg['flippable'], 1):.2f}%)"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
