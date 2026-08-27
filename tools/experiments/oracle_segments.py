#!/usr/bin/env python3
"""E3 嵌入+聚类上限：真值分割（ref RTTM turn）+ 真实 ERes2NetV2 embedding。

回答"分割完美时，编码器判别力和聚类能做到多少"。分三个子命令：

- extract：对每个 ref RTTM，把每个 turn 从对应 wav 裁出（load_audio 重采样到
  16k 单声道），时长策略镜像管线：短于 min_segment_duration_for_embedding
  (0.3s) 丢弃，长于 max_segment_duration_for_embedding (4.0s) 从起点截断
  （同 track_builder 的截断方向）。逐段提 embedding，输出每文件一个 npz：
  embeddings (N×192, 已 L2 归一化) / labels（说话人名）/ starts / ends，
  落在 <work-dir>/embeddings/{uri}.npz。
- analyze：读 embeddings npz，逐文件与全局计算类内/类间余弦分布、d-prime、
  最优单阈值错误率（复用 tools/analyze_embeddings.py 的 pair_sims/dprime/
  best_threshold_err），报告写 <work-dir>/analyze_report.txt。
- cluster：两种分配各评一次 DER——
  1) streaming：按 config.hop_duration 网格把 turn 分组成伪 chunk，构造
     ChunkObservation 顺序喂给 StreamingChunkClusterer（用 config 里的真实
     阈值），按分配结果写 raw RTTM，再叠加 refined 级（merge 链修正 +
     post-merge）写 {uri}.rttm 到 <work-dir>/streaming_rttm/；
  2) ahc-oracle：对同一批 embedding 用 AgglomerativeClustering
     (metric=cosine, linkage=config.ahc_linkage, n_clusters=真值说话人数)
     分配，直接写 {uri}.rttm 到 <work-dir>/ahc_oracle_rttm/（不叠 post-merge，
     避免小真值说话人被并掉、混淆 oracle-k 口径）。
  两版 DER 都打印并写 <work-dir>/cluster_report.txt。

用法：
  .venv/bin/python tools/experiments/oracle_segments.py extract \
      --audio datasets/task1-6 --ref datasets/task1-6/rttm \
      --config config/config.yaml \
      --work-dir exp/experiments/e3_oracle_segments --device cuda
  .venv/bin/python tools/experiments/oracle_segments.py analyze \
      --ref datasets/task1-6/rttm --work-dir exp/experiments/e3_oracle_segments
  .venv/bin/python tools/experiments/oracle_segments.py cluster \
      --ref datasets/task1-6/rttm --config config/config.yaml \
      --work-dir exp/experiments/e3_oracle_segments
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analyze_embeddings import best_threshold_err, dprime, pair_sims  # noqa: E402
from compute_der import compute_der_batch  # noqa: E402
from rttm_utils import load_rttm  # noqa: E402

from common.pipeline_config import ChunkPipelineConfig  # noqa: E402
from common.schema import ChunkObservation  # noqa: E402
from diarization.cluster.post_process import write_refined_rttm  # noqa: E402


logger = logging.getLogger("oracle_segments")


def load_base_config(config_path: str) -> ChunkPipelineConfig:
    """直接从 YAML 构造配置（与 tools/sweep_params.py 同口径）。"""

    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg_dict = yaml.safe_load(file_obj) or {}
    field_names = {f.name for f in dataclasses.fields(ChunkPipelineConfig)}
    return ChunkPipelineConfig(
        **{k: v for k, v in cfg_dict.items() if k in field_names}
    )


def iter_ref_turns(ref_dir: Path) -> list[tuple[str, Path, dict]]:
    """列出 (uri, rttm_path, turns)，turns 为 load_rttm 的 {speaker: [(s,e)]}。"""

    return [
        (path.stem, path, load_rttm(path)) for path in sorted(ref_dir.glob("*.rttm"))
    ]


# ----------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from common.models.embedding_infer import (
        NativeERes2NetV2SegmentEmbedder,
        load_embedding_model,
    )
    from speakerlab.process.processor import FBank
    from speakerlab.utils.fileio import load_audio

    config = load_base_config(args.config)
    device = torch.device(args.device)
    embedding_model = load_embedding_model(
        model_path=None,
        device=device,
        model_type=config.model_type,
        feat_dim=config.feat_dim,
        embedding_size=config.embedding_size,
        m_channels=config.m_channels,
    )
    feature_extractor = FBank(
        n_mels=config.feat_dim,
        sample_rate=config.sample_rate,
        mean_nor=True,
    )
    embedder = NativeERes2NetV2SegmentEmbedder(
        embedding_model=embedding_model,
        feature_extractor=feature_extractor,
        sample_rate=config.sample_rate,
        normalize_embeddings=config.normalize_embeddings,
    )

    out_dir = Path(args.work_dir) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = config.sample_rate
    min_dur = float(config.min_segment_duration_for_embedding)
    max_dur = float(config.max_segment_duration_for_embedding)
    batch_size = max(1, int(config.segment_batch_size))

    for uri, _, turns in iter_ref_turns(Path(args.ref)):
        wav_path = Path(args.audio) / f"{uri}.wav"
        if not wav_path.is_file():
            logger.warning("skip %s: wav not found: %s", uri, wav_path)
            continue
        waveform = load_audio(str(wav_path), obj_fs=sample_rate)
        total_samples = int(waveform.shape[1])

        # 逐 turn 裁剪（短丢弃、长从起点截断），按时间顺序排列。
        segments: list[tuple[float, float, str]] = sorted(
            (start, end, spk)
            for spk, segs in turns.items()
            for start, end in segs
        )
        kept: list[tuple[float, float, str]] = []
        waveforms: list[torch.Tensor] = []
        for start, end, spk in segments:
            duration = end - start
            if duration < min_dur:
                continue
            # 截断只作用于喂给 embedding 的音频；kept 里保留完整 [start, end)，
            # 否则 cluster 步写 RTTM 时 turn 被截到 4s，会凭空贡献大量 MS。
            embed_end = min(end, start + max_dur)
            start_sample = max(0, min(int(round(start * sample_rate)), total_samples))
            end_sample = max(
                start_sample, min(int(round(embed_end * sample_rate)), total_samples)
            )
            if end_sample <= start_sample:
                continue
            kept.append((start, end, spk))
            waveforms.append(waveform[:, start_sample:end_sample])

        embeddings: list[np.ndarray] = []
        for offset in range(0, len(waveforms), batch_size):
            embeddings.extend(
                embedder.embed_segments(waveforms[offset : offset + batch_size])
            )

        np.savez(
            out_dir / f"{uri}.npz",
            uri=np.array(uri),
            embeddings=np.stack(embeddings)
            if embeddings
            else np.zeros((0, config.embedding_size), dtype=np.float32),
            labels=np.array([spk for _, _, spk in kept]),
            starts=np.array([s for s, _, _ in kept], dtype=np.float64),
            ends=np.array([e for _, e, _ in kept], dtype=np.float64),
        )
        print(f"[extract] {uri}: {len(kept)} turns embedded -> {out_dir}/{uri}.npz")


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------


def _load_embedding_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 npz，返回 (L2 归一化 embeddings, 整数 labels)（labels 按名字字典序）。"""

    data = np.load(npz_path, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / np.maximum(norms, 1e-12)[:, None]
    speakers = sorted(set(data["labels"].tolist()))
    spk_index = {spk: idx for idx, spk in enumerate(speakers)}
    labels = np.array([spk_index[spk] for spk in data["labels"].tolist()])
    return embeddings, labels


def cmd_analyze(args: argparse.Namespace) -> None:
    npz_dir = Path(args.work_dir) / "embeddings"
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"no npz under {npz_dir}（先跑 extract 子命令）")

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    all_within: list[np.ndarray] = []
    all_between: list[np.ndarray] = []
    log(f"{'file':<6} {'n_spk':>5} {'n_seg':>6} {'mu_w':>6} {'mu_b':>6} "
        f"{'d-prime':>8} {'best_err%':>9} {'thr':>6}")
    for npz_path in npz_files:
        embeddings, labels = _load_embedding_npz(npz_path)
        within, between = pair_sims(
            embeddings, labels, np.ones(len(embeddings), dtype=bool)
        )
        err, thr = best_threshold_err(within, between)
        dp = dprime(within, between)
        all_within.append(within)
        all_between.append(between)
        log(
            f"{npz_path.stem:<6} {len(set(labels.tolist())):>5} {len(labels):>6} "
            f"{within.mean():>6.3f} {between.mean():>6.3f} {dp:>8.3f} "
            f"{100 * err:>9.2f} {thr:>6.3f}"
        )

    within = np.concatenate(all_within)
    between = np.concatenate(all_between)
    err, thr = best_threshold_err(within, between)
    log("-" * 56)
    log(
        f"GLOBAL n_pairs={len(within)}+{len(between)} "
        f"mu_w={within.mean():.3f} mu_b={between.mean():.3f} "
        f"d-prime={dprime(within, between):.3f} "
        f"best_err={100 * err:.2f}% @thr={thr:.3f}"
    )

    report_path = Path(args.work_dir) / "analyze_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


# ----------------------------------------------------------------------
# cluster
# ----------------------------------------------------------------------


def _write_turns_rttm(
    path: Path,
    uri: str,
    turns: list[tuple[float, float, int]],
    header: str,
) -> None:
    """把 [(start, end, global_id)] 直接写成 RTTM（按 start 排序）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"# {header} for {uri}\n")
        for start, end, global_id in sorted(turns):
            file_obj.write(
                f"SPEAKER {uri} 0 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> {int(global_id)} <NA> <NA>\n"
            )


def _streaming_assign(
    uri: str,
    data: np.lib.npyio.NpzFile,
    config: ChunkPipelineConfig,
    out_dir: Path,
) -> int:
    """流式臂：伪 chunk 分组喂 StreamingChunkClusterer，raw + refined 两级输出。

    返回被聚类器丢弃（未分配 global id）的 turn 数。
    """

    from diarization.cluster.backends.streaming import StreamingChunkClusterer

    embeddings = data["embeddings"]
    starts = data["starts"].tolist()
    ends = data["ends"].tolist()
    if len(starts) == 0:
        _write_turns_rttm(out_dir / f"{uri}.rttm", uri, [], "streaming oracle-seg")
        return 0

    clusterer = StreamingChunkClusterer(config)
    hop = float(config.hop_duration)
    # 按 hop 网格分伪 chunk：chunk 内不同 local 受 Hungarian cannot-link 约束，
    # 与真实管线同口径；ref turn 无重叠，同 chunk 撞车概率低。
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    pseudo_chunks: dict[int, list[int]] = {}
    for i in order:
        pseudo_chunks.setdefault(int(starts[i] // hop), []).append(i)

    assigned: dict[int, int] = {}  # turn 下标 -> global id
    for _, turn_indices in sorted(pseudo_chunks.items()):
        observations = [
            ChunkObservation(
                local_idx=slot,
                start=starts[i],
                end=ends[i],
                duration=ends[i] - starts[i],
                embedding=embeddings[i],
                mean_activity=1.0,
                # 真值 turn 均为非重叠纯净段，允许 centroid 更新；
                # 时长门控由 clusterer 内部的 min_segment_duration_for_* 判定。
                allow_centroid_update=True,
                selection_mode="non_overlap",
            )
            for slot, i in enumerate(turn_indices)
        ]
        local_to_global, _ = clusterer.assign_chunk(observations)
        for slot, i in enumerate(turn_indices):
            if slot in local_to_global:
                assigned[i] = local_to_global[slot]

    dropped = len(starts) - len(assigned)
    raw_dir = out_dir / "raw"
    raw_path = raw_dir / f"{uri}.raw.rttm"
    _write_turns_rttm(
        raw_path,
        uri,
        [(starts[i], ends[i], gid) for i, gid in assigned.items()],
        "streaming oracle-seg raw",
    )
    # refined 级：merge 链修正 + 小样本 post-merge（与管线最终输出同口径）。
    write_refined_rttm(
        str(raw_path),
        str(out_dir / f"{uri}.rttm"),
        centroids=clusterer.centroids,
        merged_into=clusterer.merged_into,
        counts=clusterer.counts,
        min_count=int(config.post_merge_min_count),
        min_similarity=float(config.post_merge_min_similarity),
    )
    return dropped


def _ahc_oracle_assign(
    uri: str,
    data: np.lib.npyio.NpzFile,
    num_speakers: int,
    config: ChunkPipelineConfig,
    out_dir: Path,
) -> None:
    """离线臂：AgglomerativeClustering(n_clusters=真值说话人数) 直接写 RTTM。"""

    from sklearn.cluster import AgglomerativeClustering

    embeddings = data["embeddings"]
    starts = data["starts"].tolist()
    ends = data["ends"].tolist()
    if len(starts) == 0:
        _write_turns_rttm(out_dir / f"{uri}.rttm", uri, [], "ahc oracle-k")
        return

    n_clusters = min(num_speakers, len(starts))
    model = AgglomerativeClustering(
        metric="cosine",
        linkage=config.ahc_linkage,
        n_clusters=n_clusters,
    )
    labels = model.fit_predict(embeddings)
    _write_turns_rttm(
        out_dir / f"{uri}.rttm",
        uri,
        [(starts[i], ends[i], int(labels[i])) for i in range(len(starts))],
        "ahc oracle-k",
    )


def _eval_dir(ref_dir: str, sys_dir: Path, log) -> None:
    results, skipped, global_result = compute_der_batch(
        ref_path=ref_dir,
        sys_path=str(sys_dir),
        collar=0.0,
        ignore_overlap=False,
        sys_suffix=".rttm",
        ref_suffix=".rttm",
    )
    for row in results:
        log(
            f"  {Path(row['sys_rttm']).stem:<6} ms={row['ms']:>6.2f} "
            f"fa={row['fa']:>6.2f} ser={row['ser']:>6.2f} der={row['der']:>6.2f}"
        )
    for filename, reason in skipped:
        log(f"  skip {filename}: {reason}")
    if global_result is not None:
        log(
            f"  GLOBAL ms={global_result['ms']:.2f} fa={global_result['fa']:.2f} "
            f"ser={global_result['ser']:.2f} der={global_result['der']:.2f} "
            f"({global_result['files']} files)"
        )


def cmd_cluster(args: argparse.Namespace) -> None:
    config = load_base_config(args.config)
    work_dir = Path(args.work_dir)
    npz_dir = work_dir / "embeddings"
    streaming_dir = work_dir / "streaming_rttm"
    ahc_dir = work_dir / "ahc_oracle_rttm"

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    ref_turns = {uri: turns for uri, _, turns in iter_ref_turns(Path(args.ref))}
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"no npz under {npz_dir}（先跑 extract 子命令）")

    for npz_path in npz_files:
        uri = npz_path.stem
        if uri not in ref_turns:
            log(f"skip {uri}: ref RTTM not found")
            continue
        data = np.load(npz_path, allow_pickle=False)
        dropped = _streaming_assign(uri, data, config, streaming_dir)
        _ahc_oracle_assign(uri, data, len(ref_turns[uri]), config, ahc_dir)
        log(f"[cluster] {uri}: streaming dropped {dropped} turn(s)")

    log()
    log("== streaming (oracle segments -> StreamingChunkClusterer, refined) ==")
    _eval_dir(args.ref, streaming_dir, log)
    log()
    log("== ahc oracle-k (n_clusters = 真值说话人数, raw) ==")
    _eval_dir(args.ref, ahc_dir, log)

    report_path = work_dir / "cluster_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--ref", required=True, help="参考 RTTM 目录")
        sub.add_argument("--work-dir", required=True, help="实验产物根目录")
        sub.add_argument("--config", default="config/config.yaml")

    sub = subparsers.add_parser("extract", help="真值 turn 裁剪 + embedding 提取")
    add_common(sub)
    sub.add_argument("--audio", required=True, help="wav 目录（<uri>.wav）")
    sub.add_argument("--device", default="cuda")
    sub.set_defaults(func=cmd_extract)

    sub = subparsers.add_parser("analyze", help="类内/类间余弦分布与 d-prime")
    add_common(sub)
    sub.set_defaults(func=cmd_analyze)

    sub = subparsers.add_parser("cluster", help="streaming 与 ahc oracle-k 两版聚类评 DER")
    add_common(sub)
    sub.set_defaults(func=cmd_cluster)

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
