#!/usr/bin/env python3
"""嵌入质量分析：复用 chunks.npz + 参考 RTTM，评估 embedding 的说话人可区分性。

只读 npz 与 RTTM（纯 numpy，不加载任何模型、不依赖项目代码），逐文件：

1. 基础统计：observation 数量、embedding 维度、L2 范数、时长分布、
   与参考说话人映射的 ambiguous 比例
2. 可区分性：within/between-speaker 余弦相似度分布、d-prime、
   最优单阈值二分错误率（embedding 本身的理论下限，调聚类参数无法突破）
3. 混淆矩阵：对 d-prime 最差的若干文件输出参考说话人质心两两相似度矩阵，
   标出不可分对（>0.85）及其最近质心误分配率
4. 时长效应：按时长分桶看短段 embedding 是否更噪声

产物：separability_hist.png / centroid_matrix_<uri>.png / duration_effect.png /
summary.txt，全部落在 --output 目录。

用法：
  .venv/bin/python tools/analyze_embeddings.py \
      --input exp/sweep_task1-6/embeddings \
      --ref datasets/task1-6/rttm \
      --output exp/sweep_task1-6/embedding_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DUR_BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)]
DUR_LABELS = ["<0.5s", "0.5-1s", "1-2s", "2-4s", ">4s"]
# 质心余弦相似度超过该值视为不可分说话人对。
INSEPARABLE_SIM = 0.85


def load_rttm(path: Path) -> tuple[list[tuple[float, float, str]], list[str]]:
    """解析 RTTM 为语音段与说话人列表。

    Args:
        path: RTTM 文件路径。

    Returns:
        (按 start 排序的语音段 [(start, end, spk)], 排序后的说话人列表)。
    """

    segs = []
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start, dur = float(parts[3]), float(parts[4])
                segs.append((start, start + dur, parts[7]))
    segs.sort(key=lambda s: s[0])
    spks = sorted({s[2] for s in segs}, key=lambda x: (len(x), x))
    return segs, spks


def map_to_ref(
    start: float, end: float, segs: list[tuple[float, float, str]], spk_index: dict[str, int]
) -> int:
    """按最大重叠把 observation 映射到参考说话人下标。

    最大重叠不足 observation 时长 50% 时返回 -1（ambiguous，不计入统计）。

    Args:
        start: observation 起点（秒）。
        end: observation 终点（秒）。
        segs: 参考语音段 [(start, end, spk)]，按 start 排序。
        spk_index: 说话人名 -> 下标。

    Returns:
        参考说话人下标；ambiguous 或时长非正返回 -1。
    """

    dur = end - start
    if dur <= 0:
        return -1
    best_ov, best_i = 0.0, -1
    for seg_start, seg_end, spk in segs:
        if seg_start >= end:
            break
        overlap = min(end, seg_end) - max(start, seg_start)
        if overlap > best_ov:
            best_ov, best_i = overlap, spk_index[spk]
    return best_i if best_ov >= 0.5 * dur else -1


def pair_sims(
    embeddings: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """mask 选中的 observation 子集内，同类对与异类对的余弦相似度。

    embeddings 须已 L2 归一化；labels=-1 的 observation 不参与。

    Args:
        embeddings: L2 归一化的 embedding 矩阵 (N, dim)。
        labels: 每个 observation 的参考说话人下标（-1 表示 ambiguous）。
        mask: 参与统计的 observation 布尔掩码。

    Returns:
        (同类对相似度数组, 异类对相似度数组)。
    """

    idx = np.where((labels >= 0) & mask)[0]
    within, between = [], []
    for pos, i in enumerate(idx):
        rest = idx[pos + 1 :]
        if len(rest) == 0:
            continue
        sims = embeddings[rest] @ embeddings[i]
        same = labels[rest] == labels[i]
        within.append(sims[same])
        between.append(sims[~same])
    w = np.concatenate(within) if within else np.array([])
    b = np.concatenate(between) if between else np.array([])
    return w, b


def dprime(within: np.ndarray, between: np.ndarray) -> float:
    """两类相似度分布的 d-prime（可分性指标，越大越可分）。

    Args:
        within: 同类对相似度。
        between: 异类对相似度。

    Returns:
        d-prime；合并方差为 0 时返回 nan。
    """

    denom = np.sqrt((within.var() + between.var()) / 2)
    if denom <= 0:
        return float("nan")
    return float(abs(within.mean() - between.mean()) / denom)


def best_threshold_err(within: np.ndarray, between: np.ndarray) -> tuple[float, float]:
    """最优单阈值二分错误率（within 应高于阈值、between 应低于阈值）。

    Args:
        within: 同类对相似度。
        between: 异类对相似度。

    Returns:
        (最低错误率, 对应阈值)。
    """

    thr = np.unique(np.concatenate([within, between]))
    if len(thr) > 4000:
        thr = np.quantile(thr, np.linspace(0, 1, 4000))
    w_below = np.searchsorted(np.sort(within), thr)  # within 误判
    b_above = len(between) - np.searchsorted(np.sort(between), thr)  # between 误判
    err = (w_below + b_above) / (len(within) + len(between))
    best = int(np.argmin(err))
    return float(err[best]), float(thr[best])


def analyze_file(
    npz_path: Path, rttm_path: Path
) -> dict[str, object]:
    """加载单个文件的 npz 与参考 RTTM，返回统计与可分性结果。

    Args:
        npz_path: chunks.npz 路径。
        rttm_path: 参考 RTTM 路径。

    Returns:
        统计与可分性结果字典（含归一化 embeddings、labels、within/between
        相似度、d-prime、最优阈值错误率等）。
    """

    data = np.load(npz_path, allow_pickle=False)
    segs, spks = load_rttm(rttm_path)
    spk_index = {s: i for i, s in enumerate(spks)}

    has = data["has_embedding"]
    embeddings = data["embeddings"][has].astype(np.float64)
    start, end, dur = data["start"][has], data["end"][has], data["duration"][has]

    norms = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / np.maximum(norms, 1e-12)[:, None]

    labels = np.array(
        [map_to_ref(s, e, segs, spk_index) for s, e in zip(start, end)]
    )

    within, between = pair_sims(
        normalized, labels, np.ones(len(normalized), dtype=bool)
    )
    err, thr = best_threshold_err(within, between)

    return {
        "uri": str(data["uri"]),
        "spks": spks,
        "k": len(spks),
        "embeddings": normalized,
        "labels": labels,
        "dur": dur,
        "n_obs": int(len(has)),
        "n_emb": int(has.sum()),
        "n_amb": int((labels == -1).sum()),
        "dim": int(embeddings.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "dur_quantiles": np.quantile(dur, [0, 0.25, 0.5, 0.75, 1.0]),
        "within": within,
        "between": between,
        "dp": dprime(within, between),
        "err": err,
        "thr": thr,
    }


def report_confusion(result: dict[str, object], out_dir: Path, log) -> None:
    """输出单文件的参考说话人质心相似度矩阵、不可分对与热图。

    Args:
        result: analyze_file 的返回。
        out_dir: 热图输出目录（centroid_matrix_<uri>.png）。
        log: 日志函数（print + 收集到 summary）。
    """

    embeddings = result["embeddings"]  # type: ignore[assignment]
    labels = result["labels"]  # type: ignore[assignment]
    dur = result["dur"]  # type: ignore[assignment]
    spks: list[str] = result["spks"]  # type: ignore[assignment]
    k: int = result["k"]  # type: ignore[assignment]
    uri: str = result["uri"]  # type: ignore[assignment]

    counts, tot_dur, cents = [], [], []
    for i in range(k):
        m = labels == i
        counts.append(int(m.sum()))
        tot_dur.append(float(dur[m].sum()))
        centroid = embeddings[m].mean(axis=0) if m.any() else np.zeros(embeddings.shape[1])
        cents.append(centroid / max(np.linalg.norm(centroid), 1e-12))
    cents_arr = np.array(cents)
    sim_matrix = cents_arr @ cents_arr.T

    log(f"\n=== File {uri}: centroid cosine similarity matrix ===")
    log("speaker stats (id: n_obs, total_speech_s):")
    for i, spk in enumerate(spks):
        log(f"  spk {spk}: {counts[i]} obs, {tot_dur[i]:.1f}s")
    log("      " + " ".join(f"{s:>6}" for s in spks))
    for i, spk in enumerate(spks):
        log(f"{spk:>6}" + " ".join(f"{sim_matrix[i, j]:6.3f}" for j in range(k)))

    log(f"\nfile {uri}: inseparable pairs (centroid sim > {INSEPARABLE_SIM}):")
    pairs = [
        (spks[i], spks[j], float(sim_matrix[i, j]))
        for i in range(k)
        for j in range(i + 1, k)
    ]
    pairs.sort(key=lambda x: -x[2])
    found = False
    for spk_a, spk_b, sim in pairs:
        if sim <= INSEPARABLE_SIM:
            break
        ia, ib = spks.index(spk_a), spks.index(spk_b)
        log(
            f"  spk {spk_a} <-> spk {spk_b}: sim={sim:.3f} "
            f"(n={counts[ia]}/{counts[ib]}, dur={tot_dur[ia]:.1f}s/{tot_dur[ib]:.1f}s)"
        )
        # 最近质心误分配率：该说话人的段离对方质心更近的比例。
        for src, dst in ((ia, ib), (ib, ia)):
            m = labels == src
            if not m.any():
                continue
            s_self = embeddings[m] @ cents_arr[src]
            s_other = embeddings[m] @ cents_arr[dst]
            mis = float((s_other > s_self).mean())
            log(
                f"    {100 * mis:.1f}% of spk {spks[src]} obs "
                f"({int(m.sum())}) are closer to spk {spks[dst]}'s centroid"
            )
        found = True
    if not found:
        log("  (none)")
    log(
        f"file {uri}: top-5 highest cross-speaker centroid sims: "
        + ", ".join(f"{a}-{b}={v:.3f}" for a, b, v in pairs[:5])
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(k), spks)
    ax.set_yticks(range(k), spks)
    for i in range(k):
        for j in range(k):
            ax.text(
                j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if sim_matrix[i, j] < 0.7 else "black",
            )
    ax.set_title(f"file {uri}: ref-speaker centroid cosine similarity")
    fig.colorbar(image)
    fig.tight_layout()
    path = out_dir / f"centroid_matrix_{uri}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"saved {path}")


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(description="嵌入质量分析（chunks.npz + 参考 RTTM）")
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录（<uri>.rttm）")
    parser.add_argument("--output", required=True, help="图表与摘要输出目录")
    parser.add_argument(
        "--confusion-files",
        type=int,
        default=2,
        help="输出质心混淆矩阵的文件数（按 d-prime 升序取最差 N 个，0 表示全部跳过）",
    )
    return parser


def main() -> None:
    """CLI 入口：逐文件分析并产出图表与 summary.txt。"""

    args = build_parser().parse_args()
    input_dir, ref_dir, out_dir = Path(args.input), Path(args.ref), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.chunks.npz"))
    if not npz_files:
        raise SystemExit(f"no chunks.npz under {input_dir}")

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    results: dict[str, dict[str, object]] = {}
    uris: list[str] = []
    for npz_path in npz_files:
        uri = npz_path.name[: -len(".chunks.npz")]
        rttm_path = ref_dir / f"{uri}.rttm"
        if not rttm_path.is_file():
            log(f"skip {uri}: ref RTTM not found: {rttm_path}")
            continue
        result = analyze_file(npz_path, rttm_path)
        results[uri] = result
        uris.append(uri)

        quantiles = result["dur_quantiles"]  # type: ignore[assignment]
        log(f"=== File {uri} (ref speakers: {result['k']}) ===")
        log(
            f"  observations: {result['n_obs']} total, {result['n_emb']} with embedding, "
            f"{result['n_amb']} ambiguous-mapping "
            f"({100 * result['n_amb'] / max(result['n_emb'], 1):.1f}%)"  # type: ignore[arg-type]
        )
        log(f"  embedding dim: {result['dim']}")
        log(
            f"  L2 norm: min={result['norm_min']:.4f} mean={result['norm_mean']:.4f} "
            f"max={result['norm_max']:.4f} -> "
            f"{'normalized' if abs(result['norm_mean'] - 1) < 0.01 else 'NOT normalized'}"  # type: ignore[arg-type]
        )
        log(
            f"  duration(s): min={quantiles[0]:.2f} p25={quantiles[1]:.2f} "
            f"med={quantiles[2]:.2f} p75={quantiles[3]:.2f} max={quantiles[4]:.2f}"
        )
        within: np.ndarray = result["within"]  # type: ignore[assignment]
        between: np.ndarray = result["between"]  # type: ignore[assignment]
        log(f"  within:  n={len(within)} mean={within.mean():.3f} std={within.std():.3f}")
        log(f"  between: n={len(between)} mean={between.mean():.3f} std={between.std():.3f}")
        log(
            f"  d-prime={result['dp']:.3f}  best-threshold err="
            f"{100 * result['err']:.2f}% @thr={result['thr']:.3f}"  # type: ignore[arg-type]
        )
        log()

    if not results:
        raise SystemExit("no file analyzed (check --input/--ref pairing)")

    # ---------- 可分性直方图（多子图） ----------
    ncols = min(3, len(uris))
    nrows = (len(uris) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.2 * nrows))
    axes_flat = np.atleast_1d(axes).flat
    for ax in axes_flat[len(uris) :]:
        ax.axis("off")
    for ax, uri in zip(axes_flat, uris):
        result = results[uri]
        within: np.ndarray = result["within"]  # type: ignore[assignment]
        between: np.ndarray = result["between"]  # type: ignore[assignment]
        ax.hist(
            between, bins=60, range=(-0.2, 1.0), density=True, alpha=0.6,
            label=f"between (mu={between.mean():.2f})", color="tab:red",
        )
        ax.hist(
            within, bins=60, range=(-0.2, 1.0), density=True, alpha=0.6,
            label=f"within (mu={within.mean():.2f})", color="tab:blue",
        )
        ax.axvline(result["thr"], color="k", ls="--", lw=1)  # type: ignore[arg-type]
        ax.set_title(
            f"file {uri} ({result['k']} spk) d'={result['dp']:.2f} "
            f"err={100 * result['err']:.1f}%"  # type: ignore[arg-type]
        )
        ax.legend(fontsize=8)
    fig.suptitle("Cosine similarity: within-speaker vs between-speaker pairs")
    fig.tight_layout()
    path = out_dir / "separability_hist.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"saved {path}")

    # ---------- 汇总表（按 d-prime 降序） ----------
    uris_by_dp = sorted(uris, key=lambda u: -results[u]["dp"])  # type: ignore[arg-type]
    log("\n=== Separability summary (sorted by d-prime desc) ===")
    log(
        f"{'file':>4} {'n_spk':>5} {'n_emb':>6} {'amb%':>6} {'mu_w':>6} "
        f"{'mu_b':>6} {'d-prime':>8} {'best_err%':>9} {'thr':>6}"
    )
    for uri in uris_by_dp:
        result = results[uri]
        within: np.ndarray = result["within"]  # type: ignore[assignment]
        between: np.ndarray = result["between"]  # type: ignore[assignment]
        log(
            f"{uri:>4} {result['k']:>5} {result['n_emb']:>6} "
            f"{100 * result['n_amb'] / max(result['n_emb'], 1):6.1f} "  # type: ignore[arg-type]
            f"{within.mean():6.3f} {between.mean():6.3f} {result['dp']:8.3f} "
            f"{100 * result['err']:9.2f} {result['thr']:6.3f}"  # type: ignore[arg-type]
        )

    # ---------- 混淆矩阵（d-prime 最差的 N 个文件） ----------
    n_confusion = args.confusion_files
    for uri in list(reversed(uris_by_dp))[:n_confusion]:
        report_confusion(results[uri], out_dir, log)

    # ---------- 时长分桶效应 ----------
    log("\n=== Duration effect: within-speaker sim mean by duration bucket ===")
    log(f"{'file':>4} " + " ".join(f"{lb:>16}" for lb in DUR_LABELS))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(DUR_LABELS))
    width = 0.8 / max(len(uris), 1)
    for fi, uri in enumerate(uris):
        result = results[uri]
        embeddings: np.ndarray = result["embeddings"]  # type: ignore[assignment]
        labels: np.ndarray = result["labels"]  # type: ignore[assignment]
        dur: np.ndarray = result["dur"]  # type: ignore[assignment]
        row, means = [], []
        for lo, hi in DUR_BUCKETS:
            m = (dur >= lo) & (dur < hi)
            n = int((m & (labels >= 0)).sum())
            if n >= 4:
                w_bucket, _ = pair_sims(embeddings, labels, m)
                mu = float(w_bucket.mean()) if len(w_bucket) else float("nan")
            else:
                mu = float("nan")
            means.append(mu)
            row.append(f"{mu:6.3f}(n={n:<4})" if not np.isnan(mu) else f"  n/a (n={n:<4})")
        log(f"{uri:>4} " + " ".join(f"{c:>16}" for c in row))
        ax.bar(
            x + (fi - (len(uris) - 1) / 2) * width, means, width, label=f"file {uri}"
        )
    ax.set_xticks(x, DUR_LABELS)
    ax.set_ylabel("mean within-speaker cosine sim")
    ax.set_title("Duration effect on embedding reliability")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "duration_effect.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"saved {path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {summary_path}")


if __name__ == "__main__":
    main()
