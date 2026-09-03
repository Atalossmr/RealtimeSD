#!/usr/bin/env python3
"""E2 分割层上限：用真值给每个 chunk 的每个 local slot 打 oracle 标签，跳过聚类。

逐文件 load_chunks，镜像 writer 的帧判定逻辑（iter_commit_frames：提交区
[commit_start, commit_end) 内 score > 0.0 即活跃，跨界帧裁剪到提交区边界），
对每个 local slot 统计其活跃帧落在各参考说话人区间内的帧数，取覆盖帧数最多
的说话人作为该 local 的 oracle global id（ref 说话人名按字典序映射为
0..k-1）。无活跃帧或活跃帧完全无真值覆盖的 local 不放入 local_to_global
（writer 对缺失 local 的帧直接丢弃，见 common/contracts/frames.py）。随后按
chunk 顺序
重放 AppendOnlyRTTMWriter（consume_chunk + close_inactive + finalize），
得到的 DER 即"当前分割帧决策 + 完美说话人分配"的下限（分割层天花板）。

用法：
  .venv/bin/python tools/experiments/oracle_tracks.py \
      --input exp/experiments/task1-6_extract/embeddings \
      --ref datasets/task1-6/rttm \
      --output-dir exp/experiments/e2_oracle_tracks/rttm \
      --report exp/experiments/e2_oracle_tracks/report.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from compute_der import compute_der_batch  # noqa: E402
from rttm_utils import load_rttm  # noqa: E402

from common.config import ChunkPipelineConfig  # noqa: E402
from common.contracts import ChunkArtifacts  # noqa: E402
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("oracle_tracks")


def load_base_config(config_path: str) -> ChunkPipelineConfig:
    """直接从 YAML 构造配置（与 tools/sweep_params.py 同口径）。

    Args:
        config_path: 配置 YAML 路径。

    Returns:
        基准 ChunkPipelineConfig。
    """

    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg_dict = yaml.safe_load(file_obj) or {}
    field_names = {f.name for f in dataclasses.fields(ChunkPipelineConfig)}
    return ChunkPipelineConfig(
        **{k: v for k, v in cfg_dict.items() if k in field_names}
    )


def build_ref_timeline(
    turns: dict[str, list[tuple[float, float]]],
) -> list[tuple[float, float, int]]:
    """把 {speaker: [(start, end)]} 展平为按 start 排序的 [(start, end, global_id)]。

    ref 说话人名按字典序映射为 0..k-1 的整数 global id。

    Args:
        turns: load_rttm 的返回。

    Returns:
        按 start 排序的 [(start, end, global_id)]。
    """

    spk_index = {spk: idx for idx, spk in enumerate(sorted(turns))}
    return sorted(
        (start, end, spk_index[spk])
        for spk, segments in turns.items()
        for start, end in segments
    )


def speaker_at(
    segments: list[tuple[float, float, int]], time_point: float
) -> int | None:
    """查 time_point 落在哪个参考说话人区间内；无覆盖返回 None（ref 无重叠）。

    Args:
        segments: build_ref_timeline 的返回（按 start 排序）。
        time_point: 查询时刻（秒）。

    Returns:
        覆盖该时刻的 global id；无覆盖返回 None。
    """

    for start, end, global_id in segments:
        if start > time_point:
            break
        if start <= time_point < end:
            return global_id
    return None


def oracle_local_to_global(
    chunk: ChunkArtifacts, segments: list[tuple[float, float, int]]
) -> dict[int, int]:
    """对 chunk 的每个 local slot 求 oracle global id。

    帧判定严格镜像 iter_commit_frames：只取与提交区 [commit_start, commit_end)
    相交的帧（跨界裁剪），score > 0.0 即活跃。以帧中点查真值说话人，取覆盖
    帧数最多者；无活跃帧或无覆盖的 local 不进 mapping（writer 丢弃其帧）。

    Args:
        chunk: 单个 chunk 的产物（seg_scores/frame_step/提交区边界）。
        segments: build_ref_timeline 的返回。

    Returns:
        local_idx -> oracle global id 的映射。
    """

    mapping: dict[int, int] = {}
    num_frames, num_locals = chunk.seg_scores.shape
    frame_step = max(1e-6, float(chunk.frame_step))
    for local_idx in range(num_locals):
        counts: dict[int, int] = {}
        for frame_idx in range(num_frames):
            if chunk.seg_scores[frame_idx, local_idx] <= 0.0:
                continue
            frame_start = chunk.chunk_start + frame_idx * frame_step
            frame_end = frame_start + frame_step
            if frame_end <= chunk.commit_start + 1e-9:
                continue
            if frame_start >= chunk.commit_end - 1e-9:
                break
            frame_start = max(frame_start, chunk.commit_start)
            frame_end = min(frame_end, chunk.commit_end)
            global_id = speaker_at(segments, 0.5 * (frame_start + frame_end))
            if global_id is not None:
                counts[global_id] = counts.get(global_id, 0) + 1
        if counts:
            mapping[local_idx] = max(counts, key=lambda gid: counts[gid])
    return mapping


def replay_file(
    chunks_path: Path,
    ref_dir: Path,
    output_dir: Path,
    config: ChunkPipelineConfig,
) -> dict[str, int]:
    """单文件：oracle 打标签 + 重放 writer，返回 local 统计。

    Args:
        chunks_path: chunks.npz 路径。
        ref_dir: 参考 RTTM 目录。
        output_dir: 系统 RTTM 输出目录（{uri}.rttm）。
        config: 配置。

    Returns:
        {"locals_total", "locals_assigned"} 统计。

    Raises:
        SystemExit: 对应参考 RTTM 不存在。
    """

    uri, artifacts = load_chunks(str(chunks_path))
    ref_path = ref_dir / f"{uri}.rttm"
    if not ref_path.is_file():
        raise SystemExit(f"ref RTTM not found for {uri}: {ref_path}")
    segments = build_ref_timeline(load_rttm(ref_path))

    writer = AppendOnlyRTTMWriter(
        str(output_dir / f"{uri}.rttm"),
        uri,
        config.min_segment_duration,
        config.segment_merge_gap,
        False,
    )
    locals_total = 0
    locals_assigned = 0
    for chunk in artifacts:
        local_to_global = oracle_local_to_global(chunk, segments)
        locals_total += int(chunk.seg_scores.shape[1])
        locals_assigned += len(local_to_global)
        writer.consume_chunk(
            chunk.seg_scores,
            chunk.frame_step,
            chunk.chunk_start,
            chunk.commit_start,
            chunk.commit_end,
            local_to_global,
        )
        writer.close_inactive(chunk.commit_end)
    writer.finalize()
    return {"locals_total": locals_total, "locals_assigned": locals_assigned}


def main() -> None:
    """CLI 入口：逐文件 oracle 打标签重放并评估 DER、写报告。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--output-dir", required=True, help="系统 RTTM 输出目录")
    parser.add_argument("--report", required=True, help="结果报告 txt 路径")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    config = load_base_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    for chunks_path in chunks_files:
        stats = replay_file(chunks_path, Path(args.ref), output_dir, config)
        uri = chunks_path.name[: -len(".chunks.npz")]
        log(
            f"[replay] {uri}: locals {stats['locals_assigned']}/"
            f"{stats['locals_total']} assigned oracle id"
        )

    results, skipped, global_result = compute_der_batch(
        ref_path=args.ref,
        sys_path=str(output_dir),
        collar=0.0,
        ignore_overlap=False,
        sys_suffix=".rttm",
        ref_suffix=".rttm",
    )
    log()
    log(f"{'file':<6} {'MS%':>7} {'FA%':>7} {'SER%':>7} {'DER%':>7}")
    for row in results:
        log(
            f"{Path(row['sys_rttm']).stem:<6} {row['ms']:>7.2f} {row['fa']:>7.2f} "
            f"{row['ser']:>7.2f} {row['der']:>7.2f}"
        )
    for filename, reason in skipped:
        log(f"skip {filename}: {reason}")
    if global_result is not None:
        log("-" * 40)
        log(
            f"GLOBAL ms={global_result['ms']:.2f} fa={global_result['fa']:.2f} "
            f"ser={global_result['ser']:.2f} der={global_result['der']:.2f} "
            f"({global_result['files']} files)"
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
