#!/usr/bin/env python3
"""E6 分离段重聚类实验：对分离导出的语音段重提嵌入、AHC 重聚类并评测 DER。

验证假设："embedding 可分性差源于混合音频短段的串扰"。逐文件：

1. 读 `{segments_dir}/{uri}.segments.jsonl`（WavSegmentSink 产物），
   忽略 manifest 中管线聚类给出的 speaker_id，只取音频与时间段；
2. 每段分离 wav 用与管线相同的 ERes2NetV2 提一个整段 embedding；
3. 按参考 RTTM 映射参考说话人，算 d-prime / 最优阈值错误率，
   与混合短段嵌入（--mixture-input 的 chunks.npz，可选）并排对比；
4. 对段 embedding 做 AHC（cosine + average，与 cluster/backends/ahc.py
   同口径），扫 similarity 阈值网格，生成 RTTM 用 compute_der 评测。

产物（--output 目录）：`sweep.csv`（阈值 × 全局 DER 明细）、`summary.txt`
（d-prime 对比表 + 逐阈值 DER 表 + 逐文件最优 DER）。

解读注意（实验设计的固有混淆，勿过度解读绝对 DER）：
- 段时间线由分离工具的导出决策固定（gap 合并、min_segment_duration、
  归属失败回退混合音频等），因此阈值扫描中 MS/FA 为常数、只有 SER 随阈值
  变化——这正是本实验想隔离的部分；但与管线 refined RTTM 的 DER 绝对值
  不可直接相比（时间线不同）。
- "唯一变量是音频来源"只是近似：分离段为整段 embedding（可能数十秒），
  而对照的混合短段是 ≤4s 纯净区截断，时长与选段策略同时不同。
- 归属失败回退混合音频的段（含串音）无法从 manifest 区分，一并参与统计。

用法：
  .venv/bin/python tools/experiments/separated_recluster.py \
      --segments exp/experiments/task1-6_sep \
      --ref datasets/task1-6/rttm \
      --mixture-input exp/experiments/task1-6_extract/embeddings \
      --output exp/experiments/e6_separated_recluster
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analyze_embeddings import (  # noqa: E402
    best_threshold_err,
    dprime,
    load_rttm,
    map_to_ref,
    pair_sims,
)
from compute_der import compute_der_batch  # noqa: E402

from common.config import ChunkPipelineConfig  # noqa: E402
from common.models.embedding_infer import (  # noqa: E402
    NativeERes2NetV2SegmentEmbedder,
    load_embedding_model,
)
from common.utils.device import resolve_device  # noqa: E402
from common.speakerlab.process.processor import FBank  # noqa: E402

logger = logging.getLogger("separated_recluster")

# AHC 相似度阈值网格（与 ahc_similarity_threshold 同语义）。
AHC_GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 .. 0.70
# 短于该时长的分离段跳过 embedding（噪声大且对 DER 贡献小）。
MIN_SEGMENT_DURATION = 0.25


def load_base_config(config_path: str) -> ChunkPipelineConfig:
    """直接从 YAML 构造配置（与 tools/sweep_params.py 同口径）。

    Args:
        config_path: 配置 YAML 路径。

    Returns:
        基准 ChunkPipelineConfig。
    """

    import yaml

    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg_dict = yaml.safe_load(file_obj) or {}
    field_names = {f.name for f in dataclasses.fields(ChunkPipelineConfig)}
    return ChunkPipelineConfig(
        **{k: v for k, v in cfg_dict.items() if k in field_names}
    )


def load_manifest(segments_dir: Path, uri: str) -> list[dict[str, object]]:
    """读取 {uri}.segments.jsonl，返回段记录列表（按 start 排序）。

    Args:
        segments_dir: segments 根目录。
        uri: 会话标识。

    Returns:
        段记录列表（含 start/end/path 等字段）。
    """

    manifest_path = segments_dir / f"{uri}.segments.jsonl"
    entries = []
    with open(manifest_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: float(e["start"]))
    return entries


def embed_segments(
    entries: list[dict[str, object]],
    embedder: NativeERes2NetV2SegmentEmbedder,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """逐段提 embedding，返回 (embeddings[N,dim], starts, ends, n_skipped)。

    短于 MIN_SEGMENT_DURATION 的段（按时间段长或实际波形长判定）跳过。

    Args:
        entries: load_manifest 的段记录列表。
        embedder: 段级 embedder。
        sample_rate: 目标采样率（不一致时重采样，多声道强制混单声道）。

    Returns:
        (embeddings, starts, ends, 跳过的段数)；全部跳过时 embeddings 形状 (0, 0)。
    """

    embeddings, starts, ends = [], [], []
    n_skipped = 0
    for entry in entries:
        start, end = float(entry["start"]), float(entry["end"])
        if end - start < MIN_SEGMENT_DURATION:
            n_skipped += 1
            continue
        waveform, wav_sr = torchaudio.load(str(entry["path"]))
        if wav_sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, wav_sr, sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)  # 强制单声道
        if waveform.shape[1] < int(MIN_SEGMENT_DURATION * sample_rate):
            n_skipped += 1
            continue
        embeddings.append(embedder.embed_segment(waveform))
        starts.append(start)
        ends.append(end)
    return (
        np.stack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32),
        np.array(starts),
        np.array(ends),
        n_skipped,
    )


def evaluate_separability(
    embeddings: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    segs: list[tuple[float, float, str]],
    spk_index: dict[str, int],
) -> dict[str, float]:
    """分离段 embedding 的可分性指标（d-prime / 最优阈值错误率）。

    Args:
        embeddings: 分离段 embedding 矩阵 (N, dim)。
        starts: 各段起点（秒）。
        ends: 各段终点（秒）。
        segs: 参考语音段 [(start, end, spk)]。
        spk_index: 说话人名 -> 下标。

    Returns:
        {"n_amb", "mu_within", "mu_between", "dp", "err", "thr"} 指标字典。
    """

    labels = np.array(
        [map_to_ref(s, e, segs, spk_index) for s, e in zip(starts, ends)]
    )
    within, between = pair_sims(
        embeddings, labels, np.ones(len(embeddings), dtype=bool)
    )
    err, thr = best_threshold_err(within, between)
    return {
        "n_amb": float((labels == -1).sum()),
        "mu_within": float(within.mean()),
        "mu_between": float(between.mean()),
        "dp": dprime(within, between),
        "err": err,
        "thr": thr,
    }


def mixture_separability(npz_path: Path, rttm_path: Path) -> dict[str, float]:
    """混合短段 embedding（chunks.npz）的可分性指标，作为对照。

    Args:
        npz_path: chunks.npz 路径。
        rttm_path: 参考 RTTM 路径。

    Returns:
        {"dp", "err", "thr"} 指标字典。
    """

    data = np.load(npz_path, allow_pickle=False)
    segs, spks = load_rttm(rttm_path)
    spk_index = {s: i for i, s in enumerate(spks)}
    has = data["has_embedding"]
    embeddings = data["embeddings"][has].astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / np.maximum(norms, 1e-12)[:, None]
    labels = np.array(
        [
            map_to_ref(s, e, segs, spk_index)
            for s, e in zip(data["start"][has], data["end"][has])
        ]
    )
    within, between = pair_sims(
        embeddings, labels, np.ones(len(embeddings), dtype=bool)
    )
    err, thr = best_threshold_err(within, between)
    return {
        "dp": dprime(within, between),
        "err": err,
        "thr": thr,
    }


def ahc_labels(embeddings: np.ndarray, similarity_threshold: float) -> np.ndarray:
    """cosine + average linkage AHC，与 cluster/backends/ahc.py 同口径。

    Args:
        embeddings: embedding 矩阵 (N, dim)。
        similarity_threshold: 相似度阈值（距离阈值 = 1 - 阈值）。

    Returns:
        逐样本簇标签数组。
    """

    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
        n_clusters=None,
    )
    return model.fit_predict(embeddings)


def write_rttm(
    path: Path,
    uri: str,
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
) -> None:
    """把 (时间段, 簇标签) 写成标准 RTTM。

    Args:
        path: 输出 RTTM 路径。
        uri: 会话标识。
        starts: 各段起点（秒）。
        ends: 各段终点（秒）。
        labels: 各段簇标签。
    """

    with open(path, "w", encoding="utf-8") as file_obj:
        for start, end, label in zip(starts, ends, labels):
            file_obj.write(
                f"SPEAKER {uri} 0 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> {label} <NA> <NA>\n"
            )


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(description="分离段重提嵌入 + AHC 重聚类实验")
    parser.add_argument("--segments", required=True, help="segments 根目录（含 {uri}.segments.jsonl）")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录（{uri}.rttm）")
    parser.add_argument("--output", required=True, help="分析产物输出目录")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mixture-input",
        default=None,
        help="混合段 chunks.npz 目录（可选，用于 d-prime 对照）",
    )
    parser.add_argument("--device", default=None, help="设备覆盖（如 cuda/cpu）")
    parser.add_argument("--model_path", default=None, help="embedding checkpoint 覆盖")
    parser.add_argument(
        "--collar",
        type=float,
        default=0.0,
        help="DER 评估的宽容边界（秒），默认 0.0（与 tools/experiments 统一口径）",
    )
    return parser


def main() -> None:
    """CLI 入口：pass 1 提嵌入+可分性，pass 2 AHC 阈值扫描+DER，写产物。"""

    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    segments_dir, ref_dir = Path(args.segments), Path(args.ref)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_base_config(args.config)
    device = resolve_device(args.device or config.device)

    model = load_embedding_model(
        model_path=args.model_path,
        device=device,
        model_type=config.model_type,
        feat_dim=config.feat_dim,
        embedding_size=config.embedding_size,
        m_channels=config.m_channels,
    )
    embedder = NativeERes2NetV2SegmentEmbedder(
        embedding_model=model,
        feature_extractor=FBank(n_mels=config.feat_dim, sample_rate=config.sample_rate, mean_nor=True),
        sample_rate=config.sample_rate,
        normalize_embeddings=config.normalize_embeddings,
    )

    manifests = sorted(segments_dir.glob("*.segments.jsonl"))
    if not manifests:
        raise SystemExit(f"no *.segments.jsonl under {segments_dir}")

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ---------- pass 1：提嵌入 + 可分性 ----------
    per_uri: dict[str, dict[str, object]] = {}
    for manifest_path in manifests:
        uri = manifest_path.name[: -len(".segments.jsonl")]
        rttm_path = ref_dir / f"{uri}.rttm"
        if not rttm_path.is_file():
            log(f"skip {uri}: ref RTTM not found: {rttm_path}")
            continue
        entries = load_manifest(segments_dir, uri)
        embeddings, starts, ends, n_skipped = embed_segments(
            entries, embedder, config.sample_rate
        )
        if len(embeddings) == 0:
            log(f"skip {uri}: no segment embedded")
            continue
        segs, spks = load_rttm(rttm_path)
        spk_index = {s: i for i, s in enumerate(spks)}
        sep_stats = evaluate_separability(embeddings, starts, ends, segs, spk_index)

        durations = ends - starts
        log(f"=== File {uri} (ref speakers: {len(spks)}) ===")
        log(
            f"  segments: {len(entries)} in manifest, {len(embeddings)} embedded, "
            f"{n_skipped} skipped (<{MIN_SEGMENT_DURATION}s)"
        )
        qs = np.quantile(durations, [0, 0.25, 0.5, 0.75, 1.0])
        log(
            f"  segment duration(s): min={qs[0]:.2f} p25={qs[1]:.2f} "
            f"med={qs[2]:.2f} p75={qs[3]:.2f} max={qs[4]:.2f}"
        )
        log(
            f"  separated  : mu_within={sep_stats['mu_within']:.3f} "
            f"mu_between={sep_stats['mu_between']:.3f} d-prime={sep_stats['dp']:.3f} "
            f"best_err={100 * sep_stats['err']:.2f}% @thr={sep_stats['thr']:.3f} "
            f"(amb={int(sep_stats['n_amb'])})"
        )
        if args.mixture_input:
            npz_path = Path(args.mixture_input) / f"{uri}.chunks.npz"
            if npz_path.is_file():
                mix_stats = mixture_separability(npz_path, rttm_path)
                log(
                    f"  mixture    : d-prime={mix_stats['dp']:.3f} "
                    f"best_err={100 * mix_stats['err']:.2f}% @thr={mix_stats['thr']:.3f}"
                )
                sep_stats["mixture_dp"] = mix_stats["dp"]
                sep_stats["mixture_err"] = mix_stats["err"]
        log()
        per_uri[uri] = {
            "embeddings": embeddings,
            "starts": starts,
            "ends": ends,
            "sep": sep_stats,
        }

    if not per_uri:
        raise SystemExit("no file analyzed (check --segments/--ref pairing)")

    # ---------- pass 2：AHC 阈值扫描 + DER ----------
    ref_dir_str = str(ref_dir)
    log(f"=== AHC threshold sweep (collar={args.collar}s, 计入重叠) ===")
    sweep_rows: list[dict[str, object]] = []
    best_per_file: dict[str, dict[str, float]] = {}
    for thr in AHC_GRID:
        with tempfile.TemporaryDirectory(prefix="sep_ahc_") as tmp:
            tmp_path = Path(tmp)
            for uri, data in per_uri.items():
                labels = ahc_labels(data["embeddings"], thr)  # type: ignore[arg-type]
                write_rttm(
                    tmp_path / f"{uri}.rttm",
                    uri,
                    data["starts"],  # type: ignore[arg-type]
                    data["ends"],  # type: ignore[arg-type]
                    labels,
                )
            per_file, _, global_result = compute_der_batch(
                ref_path=ref_dir_str,
                sys_path=tmp,
                collar=args.collar,
                ignore_overlap=False,
                sys_suffix=".rttm",
                ref_suffix=".rttm",
            )
        row = {
            "threshold": thr,
            "ms": float(global_result["ms"]),
            "fa": float(global_result["fa"]),
            "ser": float(global_result["ser"]),
            "der": float(global_result["der"]),
        }
        sweep_rows.append(row)
        for item in per_file:
            # filename 形如 "{uri}.rttm"，剥掉后缀取 uri。
            uri = str(item["filename"])[: -len(".rttm")]
            der = float(item["der"])
            current = best_per_file.get(uri)
            if current is None or der < current["der"]:
                best_per_file[uri] = {"threshold": thr, "der": der}
        log(
            f"ahc={thr:.2f} -> MS={row['ms']:.2f} FA={row['fa']:.2f} "
            f"SER={row['ser']:.2f} DER={row['der']:.2f}"
        )

    csv_path = out_dir / "sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    best_global = min(sweep_rows, key=lambda r: r["der"])
    log(f"\nbest global: thr={best_global['threshold']:.2f} DER={best_global['der']:.2f}")
    log("\n=== 逐文件最优 DER（oracle 阈值，仅作参考） ===")
    for uri in sorted(best_per_file):
        best = best_per_file[uri]
        log(f"  {uri}: DER={best['der']:.2f} @thr={best['threshold']:.2f}")
    log(f"\nwrote {csv_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
