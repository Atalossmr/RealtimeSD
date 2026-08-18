"""管线的配置定义、YAML 加载与 CLI 构建。

约定：

- YAML 是全部调参项的唯一来源；
- CLI 仅保留运行时输入、模型/环境参数与少量开关；
- YAML 键名会校验，合法键 = `ChunkPipelineConfig` 字段名 ∪ CLI 参数名。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Optional

import yaml

from .constants import BASE_DIR


DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def _parser_dest_set(parser: argparse.ArgumentParser) -> set[str]:
    """收集 argparse 中定义过的参数名，用于校验 YAML 键名。"""

    return {
        action.dest
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help"}
    }


def _extract_provided_dests(
    parser: argparse.ArgumentParser, argv: list[str] | None
) -> set[str]:
    """根据原始命令行，判断用户显式传了哪些参数。"""

    if argv is None:
        return set()

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    provided: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            provided.add(dest)
    return provided


def _load_yaml_config(config_path: str, explicit: bool) -> dict[str, object]:
    """读取 YAML 配置文件。"""

    path = Path(config_path)
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}

    with open(path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at top level")
    return data


def validate_runtime_args(args: argparse.Namespace) -> None:
    """校验运行实时脚本所必需的输入参数。"""

    missing: list[str] = []
    for field_name in ("wav", "output_dir"):
        value = getattr(args, field_name, None)
        if value in {None, ""}:
            missing.append(field_name)
    if missing:
        raise ValueError(
            "Missing required runtime arguments after merging CLI and YAML: "
            + ", ".join(missing)
        )


@dataclass
class ChunkPipelineConfig:
    """chunk 管线的统一配置（字段按模块分组：环境 / extract / cluster / 输出）。"""

    # ---- 运行环境 ----
    device: str = "cpu"
    sample_rate: int = 16000
    hf_token: Optional[str] = None
    hf_cache_dir: str = str(BASE_DIR / "pretrained" / "huggingface")

    # ---- extract 阶段：模型 ----
    # ERes2NetV2 说话人 embedding。
    model_type: str = "eres2netv2"
    embedding_size: int = 192
    feat_dim: int = 80
    m_channels: int = 64
    normalize_embeddings: bool = True
    # pyannote segmentation-3.0 局部识别。
    segmentation_model: str = "pyannote/segmentation-3.0"
    segmentation_batch_size: int = 1

    # ---- extract 阶段：chunk 调度 ----
    chunk_duration: float = 10.0
    # 窗口推进步长（秒）。hop == chunk_duration 时退化为非重叠模式；
    # hop < chunk_duration 时按重叠滑窗运行，每个窗口只提交中段 hop 秒。
    hop_duration: float = 5.0

    # ---- extract 阶段：local track 构造与 embedding 提取 ----
    min_local_activity_duration: float = 0.30
    min_segment_duration_for_embedding: float = 0.30
    max_segment_duration_for_embedding: float = 4.0
    # 纯净区选取优先级：commit（提交区内片段优先，不足再从两侧 margin 补齐）
    # / latest（最新优先，旧默认行为）。
    region_priority: str = "commit"
    segment_batch_size: int = 8

    # ---- cluster 阶段：后端选择与通用 ----
    clustering_backend: str = "streaming"
    save_embeddings: bool = False

    # ---- cluster 阶段：streaming 后端（全局 speaker 匹配与 centroid 维护） ----
    max_speakers: int = 50
    new_speaker_threshold: float = 0.50
    global_match_threshold: float = 0.55
    min_segment_duration_for_new_speaker: float = 0.50
    min_segment_duration_for_centroid_update: float = 1.50
    # 每次加入新片段后，若最相似的一对 centroid 相似度 ≥ 该阈值则合并
    # （count 小者并入大者）；raw RTTM 已写出行不受影响（历史行由 refined 级
    # 修正），被合并者退出后续聚类。
    merge_threshold: float = 0.70
    # new-speaker hold：某 chunk 新建 speaker 后，缓存该 chunk 及后续最多
    # N 个 chunk 的输出（RTTM 与 exporter 同步等待），缓刑 speaker 被全部
    # merge 或满 N 个 chunk 时经 merged_into 重映射后一起输出；0 = 关闭。
    new_speaker_hold_chunks: int = 0
    # 开启后，已存活过缓冲期（new_speaker_hold_chunks）的 speaker 不允许被
    # merge 掉：只有缓刑期内的 speaker 可以作为 absorbed 方。配合调低
    # merge_threshold 可在不误并资深 speaker 的前提下尽早修掉 false split。
    merge_protect_established: bool = False

    # ---- cluster 阶段：ahc 后端（离线层次聚类） ----
    ahc_similarity_threshold: float = 0.50
    ahc_linkage: str = "average"

    # ---- cluster 阶段：后处理（小样本簇强制合并，ahc / streaming refined 共用） ----
    # 总发声时长低于该值的簇/speaker 视为小样本：ahc 在 finalize 统一并入质心
    # 最相似的达标簇；streaming 由 refined 级在 EOF 最终刷新时叠加合并
    # （raw RTTM 保持 append-only 不动），speaker 未达标期间在前端标记为
    # uncertain（见 speakers.json sidecar）。0 = 关闭。
    post_merge_min_speech_duration: float = 0.0
    # 强制合并的相似度下限：小样本簇与目标簇质心余弦相似度低于该值时保留
    # 原身份不并（防止把说得少的真实独立 speaker 错并）。
    post_merge_min_similarity: float = 0.0

    # ---- 分段音频导出（接流式 ASR；仅 streaming 后端生效） ----
    # 开启后构造 exporter 导出逐 speaker 音频段（wav + manifest）：每个 commit
    # 区检测重叠帧，无重叠直接按 speaker 切片输出，有重叠则用 TIGER 分离整个
    # commit 区（能量门控 + embedding 匹配归属）。pipeline 内不做 ASR，转写由
    # python -m asr.app 读取输出目录独立完成，转写调参项在 config/asr.yaml。
    separation_enabled: bool = False
    # TIGER 分离模型（Hugging Face，固定 2 路输出、16kHz）。
    separation_model: str = "JusperLee/TIGER-speech"
    # 能量门控：分离音轨在重叠帧区间的 RMS / 混合音频同区间 RMS，
    # 低于该比值判为伪源（OSD 疑似误报），整窗回退为原始音频。
    separation_energy_ratio: float = 0.10
    # 2x2 匹配结果的最小余弦相似度，低于则判分离质量不可靠，回退原始音频。
    # 注意：aishell4 全量标定（exp/sep_export_full）表明真/假重叠窗的 min_sim
    # 分布高度重合（中位数 0.406 vs 0.388），该阈值无法区分 OSD 误报，
    # 仅用于防极端分离崩溃，不宜调高（0.30 时误伤 27.7% 真重叠窗）。
    separation_min_match_similarity: float = 0.10
    # 分离音轨归属匹配的参照 embedding：
    # observation（默认）：用本 chunk 的观测 embedding，与分离音轨同时间窗、
    #   域最接近（按 assigner 契约，候选 speaker 在本 chunk 必有观测）；
    # centroid：一律用全局质心，不受本 chunk 观测质量影响，更稳但分数偏低。
    separation_match_reference: str = "observation"

    # ---- RTTM 输出 ----
    min_segment_duration: float = 0.30
    streaming_merge_gap: float = 0.25
    output_dir_for_streaming: Optional[str] = None
    show_rttm: bool = False

    # ---- 调试 ----
    debug: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器（仅运行时/模型/开关参数）。"""

    parser = argparse.ArgumentParser(
        description="chunk 级局部识别 + 增量聚类的实时说话人分离脚本"
    )
    parser.add_argument(
        "--wav",
        default=None,
        help="输入单个 wav/flac/mp3、目录，或每行一个路径的文本文件",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录，每个文件会写一个 .raw.rttm（流式后端另有 .refined.rttm）",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径；全部调参项以该文件为唯一来源",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="预训练 ERes2NetV2 checkpoint 路径；未提供时会尝试从 ModelScope 下载默认模型",
    )
    parser.add_argument("--device", default="auto", help="运行设备，如 auto/cpu/cuda:0")
    parser.add_argument(
        "--model_type",
        default="eres2netv2",
        choices=["eres2netv2"],
        help="speaker encoder 类型，当前仅支持 eres2netv2",
    )
    parser.add_argument(
        "--segmentation_model",
        default="pyannote/segmentation-3.0",
        help="pyannote segmentation 模型名",
    )
    parser.add_argument("--hf_token", default=None, help="Hugging Face token")
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Hugging Face 模型缓存目录，默认使用仓库内 pretrained/huggingface",
    )
    parser.add_argument(
        "--show_rttm",
        action="store_true",
        help="把新生成的 RTTM 行同步输出到控制台",
    )
    parser.add_argument("--debug", action="store_true", help="输出 chunk 级 debug 信息")
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    return parser


def merge_args_with_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """把 YAML 配置和 CLI 参数合并成最终运行参数。

    合并优先级：argparse 默认值 < YAML 配置 < 显式 CLI 参数。
    调参项不在 CLI 上，因此它们的生效值始终以 YAML 为准。
    """

    provided_dests = _extract_provided_dests(parser, argv)
    args_dict = vars(args)

    config_path = str(args_dict.get("config", DEFAULT_CONFIG_PATH))
    explicit_config = "config" in provided_dests
    yaml_config = _load_yaml_config(config_path, explicit=explicit_config)

    valid_keys = _parser_dest_set(parser) | {
        field.name for field in dataclass_fields(ChunkPipelineConfig)
    }
    unknown_keys = sorted(set(yaml_config.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown keys in YAML config: {', '.join(unknown_keys)}")

    merged = dict(args_dict)
    merged.update(yaml_config)

    for dest in provided_dests:
        if dest == "config":
            continue
        merged[dest] = args_dict[dest]

    merged["config"] = config_path
    return argparse.Namespace(**merged)


def _merged_value(merged_args: argparse.Namespace, name: str, default):
    """从合并后的参数对象中读取字段，缺失时回退到给定默认值。"""

    return getattr(merged_args, name, default)


def config_from_args(args: argparse.Namespace) -> ChunkPipelineConfig:
    """把合并后的参数转换为 `ChunkPipelineConfig`。"""

    config = ChunkPipelineConfig(
        device=_merged_value(args, "device", "auto"),
        model_type=_merged_value(args, "model_type", "eres2netv2"),
        segmentation_model=_merged_value(
            args, "segmentation_model", "pyannote/segmentation-3.0"
        ),
        hf_token=_merged_value(args, "hf_token", None),
        chunk_duration=float(_merged_value(args, "chunk_duration", 10.0)),
        hop_duration=float(_merged_value(args, "hop_duration", 5.0)),
        min_local_activity_duration=float(
            _merged_value(args, "min_local_activity_duration", 0.30)
        ),
        min_segment_duration_for_embedding=float(
            _merged_value(args, "min_segment_duration_for_embedding", 0.30)
        ),
        max_segment_duration_for_embedding=float(
            _merged_value(args, "max_segment_duration_for_embedding", 4.0)
        ),
        region_priority=str(_merged_value(args, "region_priority", "commit")),
        segment_batch_size=int(_merged_value(args, "segment_batch_size", 8)),
        max_speakers=int(_merged_value(args, "max_speakers", 50)),
        new_speaker_threshold=float(_merged_value(args, "new_speaker_threshold", 0.50)),
        global_match_threshold=float(
            _merged_value(args, "global_match_threshold", 0.55)
        ),
        min_segment_duration_for_new_speaker=float(
            _merged_value(args, "min_segment_duration_for_new_speaker", 0.50)
        ),
        min_segment_duration_for_centroid_update=float(
            _merged_value(args, "min_segment_duration_for_centroid_update", 1.50)
        ),
        merge_threshold=float(_merged_value(args, "merge_threshold", 0.70)),
        new_speaker_hold_chunks=int(_merged_value(args, "new_speaker_hold_chunks", 0)),
        merge_protect_established=bool(
            _merged_value(args, "merge_protect_established", False)
        ),
        clustering_backend=str(_merged_value(args, "clustering_backend", "streaming")),
        ahc_similarity_threshold=float(
            _merged_value(args, "ahc_similarity_threshold", 0.50)
        ),
        ahc_linkage=str(_merged_value(args, "ahc_linkage", "average")),
        post_merge_min_speech_duration=float(
            _merged_value(args, "post_merge_min_speech_duration", 0.0)
        ),
        post_merge_min_similarity=float(
            _merged_value(args, "post_merge_min_similarity", 0.0)
        ),
        separation_enabled=bool(_merged_value(args, "separation_enabled", False)),
        separation_model=str(
            _merged_value(args, "separation_model", "JusperLee/TIGER-speech")
        ),
        separation_energy_ratio=float(
            _merged_value(args, "separation_energy_ratio", 0.10)
        ),
        separation_min_match_similarity=float(
            _merged_value(args, "separation_min_match_similarity", 0.10)
        ),
        separation_match_reference=str(
            _merged_value(args, "separation_match_reference", "observation")
        ),
        save_embeddings=bool(_merged_value(args, "save_embeddings", False)),
        min_segment_duration=float(_merged_value(args, "min_segment_duration", 0.30)),
        streaming_merge_gap=float(_merged_value(args, "streaming_merge_gap", 0.25)),
        output_dir_for_streaming=_merged_value(args, "output_dir", None),
        show_rttm=bool(_merged_value(args, "show_rttm", False)),
        debug=bool(_merged_value(args, "debug", False)),
    )
    if config.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be > 0")
    if config.new_speaker_hold_chunks < 0:
        raise ValueError("new_speaker_hold_chunks must be >= 0")
    if config.post_merge_min_speech_duration < 0:
        raise ValueError("post_merge_min_speech_duration must be >= 0")
    if config.post_merge_min_similarity < 0:
        raise ValueError("post_merge_min_similarity must be >= 0")
    if config.hop_duration <= 0.0 or config.hop_duration > config.chunk_duration:
        raise ValueError("hop_duration must be in (0, chunk_duration]")
    if config.region_priority not in {"latest", "commit"}:
        raise ValueError(
            f"region_priority must be 'latest' or 'commit', got {config.region_priority!r}"
        )
    if config.separation_match_reference not in {"centroid", "observation"}:
        raise ValueError(
            "separation_match_reference must be 'centroid' or 'observation', "
            f"got {config.separation_match_reference!r}"
        )

    hf_cache_dir = _merged_value(args, "hf_cache_dir", None)
    if hf_cache_dir:
        # YAML 中的相对路径（如 ./pretrained/huggingface）按仓库根目录解析，
        # 避免从其他工作目录运行时缓存失效或重复下载。
        hf_path = Path(str(hf_cache_dir)).expanduser()
        if not hf_path.is_absolute():
            hf_path = BASE_DIR / hf_path
        config.hf_cache_dir = str(hf_path)
    return config


__all__ = [
    "ChunkPipelineConfig",
    "DEFAULT_CONFIG_PATH",
    "build_arg_parser",
    "merge_args_with_config",
    "config_from_args",
    "validate_runtime_args",
]
