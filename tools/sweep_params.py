#!/usr/bin/env python3
"""参数网格扫描：复用 chunks.npz，对任意配置字段做网格实验并算 DER。

只跑聚类阶段（不加载任何模型）：每个参数组合重放全部 chunks.npz 生成
RTTM（经 refined 级逻辑：流式 merge 链修正 + 小样本 post-merge，与管线最终
输出同口径），再用本目录 compute_der.py 的批量评测算 global DER。

扫描维度通过 --set 以字符串传入，可重复、取笛卡尔积：

    --set global_match_threshold=0.45,0.50,0.55
    --set new_speaker_threshold=0.50,0.55
    --set post_merge_min_count=0,6,8,12
    --set clustering_backend=ahc --set ahc_similarity_threshold=0.35,0.40,0.45

字段名必须是 ChunkPipelineConfig 的已有字段；数值按该字段当前值的类型解析
（int/float/str/bool）。每个组合都完整重放一次聚类，逻辑直白、无特殊字段
处理。

用法：
  .venv/bin/python tools/sweep_params.py \
      --input exp/sweep_task1-6/embeddings \
      --ref datasets/task1-6/rttm \
      --config config/config.yaml \
      --set global_match_threshold=0.40,0.45,0.50,0.55,0.60 \
      --set new_speaker_threshold=0.40,0.45,0.50,0.55,0.60 \
      --output exp/sweep_task1-6/thresholds.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import logging
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_der import compute_der_batch  # noqa: E402

from common.config import ChunkPipelineConfig  # noqa: E402
from diarization.cluster.backends import build_clusterer  # noqa: E402
from diarization.cluster.post_process import write_refined_rttm  # noqa: E402
from diarization.cluster.runner import run_clustering  # noqa: E402
from diarization.cluster.rttm_writer import AppendOnlyRTTMWriter  # noqa: E402
from diarization.utils.chunk_io import load_chunks  # noqa: E402


logger = logging.getLogger("sweep_params")


def load_base_config(config_path: str) -> ChunkPipelineConfig:
    """直接从 YAML 构造配置（跳过 CLI 合并，只取 dataclass 字段）。

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


def parse_set_spec(spec: str, base_config: ChunkPipelineConfig) -> tuple[str, list]:
    """解析 "field=v1,v2,..."，按字段当前值的类型转换各取值。

    Args:
        spec: 扫描维度描述，形如 field=v1,v2,...。
        base_config: 基准配置（字段存在性与类型解析的依据）。

    Returns:
        (字段名, 取值列表)。

    Raises:
        SystemExit: 格式错误、字段未知、取值类型不匹配或为空。
    """

    if "=" not in spec:
        raise SystemExit(f"--set 格式应为 field=v1,v2,...，收到: {spec!r}")
    field, raw_values = spec.split("=", 1)
    field = field.strip()
    field_names = {f.name for f in dataclasses.fields(ChunkPipelineConfig)}
    if field not in field_names:
        raise SystemExit(
            f"未知字段 {field!r}（须为 ChunkPipelineConfig 字段），"
            f"可选: {', '.join(sorted(field_names))}"
        )
    values = [v.strip() for v in raw_values.split(",") if v.strip()]
    if not values:
        raise SystemExit(f"--set {field}: 至少需要一个取值")

    base_value = getattr(base_config, field)

    def convert(raw: str):
        if isinstance(base_value, bool):
            if raw.lower() in ("true", "1", "yes"):
                return True
            if raw.lower() in ("false", "0", "no"):
                return False
            raise SystemExit(f"--set {field}: {raw!r} 不是合法布尔值")
        if isinstance(base_value, int):
            try:
                return int(raw)
            except ValueError:
                raise SystemExit(f"--set {field}: {raw!r} 不是整数") from None
        if isinstance(base_value, float):
            try:
                return float(raw)
            except ValueError:
                raise SystemExit(f"--set {field}: {raw!r} 不是数值") from None
        return raw  # str 及 Optional[str] 等按字符串处理

    return field, [convert(v) for v in values]


def replay_files(
    config: ChunkPipelineConfig,
    chunks_files: list[Path],
    work_dir: Path,
) -> dict[str, dict[str, object]]:
    """逐文件重放聚类写 raw RTTM，返回各文件的 clusterer 终态。

    Args:
        config: 本组合的完整配置。
        chunks_files: chunks.npz 文件列表。
        work_dir: raw RTTM 输出目录（{uri}.raw.rttm）。

    Returns:
        {uri: {"centroids": ..., "counts": ..., "merged_into": ...}}。
    """

    states: dict[str, dict[str, object]] = {}
    for chunks_path in chunks_files:
        uri, artifacts = load_chunks(str(chunks_path))
        clusterer = build_clusterer(config)
        writer = AppendOnlyRTTMWriter(
            str(work_dir / f"{uri}.raw.rttm"),
            uri,
            config.min_segment_duration,
            config.segment_merge_gap,
            False,
        )
        run_clustering(artifacts, clusterer, writer)
        states[uri] = {
            "centroids": dict(clusterer.centroids),
            "counts": dict(clusterer.counts),
            "merged_into": dict(clusterer.merged_into),
        }
    return states


def evaluate_combo(
    states: dict[str, dict[str, object]],
    raw_dir: Path,
    eval_dir: Path,
    config: ChunkPipelineConfig,
) -> dict[str, float]:
    """在重放终态上叠加 refined 级逻辑（merge 链修正 + post-merge）并算 DER。

    Args:
        states: replay_files 的返回（uri -> clusterer 终态）。
        raw_dir: raw RTTM 目录（{uri}.raw.rttm）。
        eval_dir: refined RTTM 输出目录（{uri}.rttm）。
        config: 本组合的完整配置。

    Returns:
        {"ms", "fa", "ser", "der"} 全局指标（百分比）。
    """

    for uri, state in states.items():
        write_refined_rttm(
            str(raw_dir / f"{uri}.raw.rttm"),
            str(eval_dir / f"{uri}.rttm"),
            centroids=state["centroids"],  # type: ignore[arg-type]
            merged_into=state["merged_into"],  # type: ignore[arg-type]
            counts=state["counts"],  # type: ignore[arg-type]
            min_count=int(config.post_merge_min_count),
            min_similarity=float(config.post_merge_min_similarity),
        )
    _, _, global_result = compute_der_batch(
        ref_path=str(REF_DIR),
        sys_path=str(eval_dir),
        collar=0.0,
        ignore_overlap=False,
        sys_suffix=".rttm",
        ref_suffix=".rttm",
    )
    assert global_result is not None
    return {
        "ms": float(global_result["ms"]),
        "fa": float(global_result["fa"]),
        "ser": float(global_result["ser"]),
        "der": float(global_result["der"]),
    }


REF_DIR: Path


def main() -> None:
    """CLI 入口：按 --set 笛卡尔积逐组合重放聚类、评 DER 并写 CSV。"""

    global REF_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chunks.npz 目录")
    parser.add_argument("--ref", required=True, help="参考 RTTM 目录")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="FIELD=V1,V2,...",
        help="扫描维度（可重复，取笛卡尔积）；不给则按配置单点评估",
    )
    parser.add_argument("--output", required=True, help="结果 CSV 路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    REF_DIR = Path(args.ref)

    chunks_files = sorted(Path(args.input).glob("*.chunks.npz"))
    if not chunks_files:
        raise SystemExit(f"no chunks.npz under {args.input}")
    base_config = load_base_config(args.config)

    dimensions = [parse_set_spec(spec, base_config) for spec in args.sets]
    swept_fields = [field for field, _ in dimensions]
    if len(set(swept_fields)) != len(swept_fields):
        raise SystemExit(f"--set 字段重复: {swept_fields}")
    combos = [
        dict(zip(swept_fields, values))
        for values in itertools.product(*[vals for _, vals in dimensions])
    ] or [{}]

    rows: list[dict[str, object]] = []
    for combo in combos:
        config = dataclasses.replace(base_config, **combo)
        with tempfile.TemporaryDirectory(prefix="sweep_raw_") as raw_tmp:
            states = replay_files(config, chunks_files, Path(raw_tmp))
            with tempfile.TemporaryDirectory(prefix="sweep_eval_") as eval_tmp:
                metrics = evaluate_combo(states, Path(raw_tmp), Path(eval_tmp), config)
        row = {**combo, **metrics}
        rows.append(row)
        print(
            " ".join(f"{f}={combo[f]}" for f in swept_fields)
            + f" -> MS={metrics['ms']:.2f} FA={metrics['fa']:.2f} "
            f"SER={metrics['ser']:.2f} DER={metrics['der']:.2f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = swept_fields + ["ms", "fa", "ser", "der"]
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: r["der"])
    print(f"best: {best}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
