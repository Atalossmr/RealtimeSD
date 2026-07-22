"""chunk 管线的配置定义、YAML 加载与 CLI 构建。

与主 `pipeline/cli.py` 同一约定：

- YAML 是全部调参项的唯一来源；
- CLI 仅保留运行时输入、模型/环境参数与少量开关；
- YAML 键名会校验，合法键 = `ChunkPipelineConfig` 字段名 ∪ CLI 参数名。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields as dataclass_fields
from typing import Optional

from ..cli import (
    _extract_provided_dests,
    _load_yaml_config,
    _parser_dest_set,
    validate_runtime_args,
)
from ..constants import BASE_DIR


DEFAULT_CHUNK_CONFIG_PATH = BASE_DIR / "config_chunk.yaml"


@dataclass
class ChunkPipelineConfig:
    """chunk 管线的统一配置。"""

    # 音频与调度。
    sample_rate: int = 16000
    chunk_duration: float = 10.0

    # ERes2NetV2 配置。
    model_type: str = "eres2netv2"
    embedding_size: int = 192
    feat_dim: int = 80
    m_channels: int = 64
    normalize_embeddings: bool = True

    # segmentation-3.0 配置。
    segmentation_model: str = "pyannote/segmentation-3.0"
    segmentation_batch_size: int = 1
    hf_token: Optional[str] = None
    hf_cache_dir: str = str(BASE_DIR / "pretrained" / "huggingface")
    device: str = "cpu"

    # chunk 内 local track 构造。
    min_local_activity_duration: float = 0.30
    min_segment_duration_for_embedding: float = 0.30
    max_segment_duration_for_embedding: float = 4.0
    segment_batch_size: int = 8

    # 全局 speaker 匹配与维护。
    max_speakers: int = 50
    new_speaker_threshold: float = 0.40
    global_match_threshold: float = 0.50
    absorb_threshold: float = 0.60
    min_segment_duration_for_new_speaker: float = 1.00
    min_segment_duration_for_centroid_update: float = 1.50
    update_segment_overlap_threshold: float = 0.5
    weak_update_similarity_margin: float = 0.15
    weak_update_weight_multiplier: float = 1.00
    probation_confirm_duration: float = 3.0

    # RTTM 输出。
    min_segment_duration: float = 0.30
    streaming_merge_gap: float = 0.75
    output_dir_for_streaming: Optional[str] = None
    show_rttm: bool = False
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
        help="输出目录，每个文件会写一个 .streaming.rttm",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CHUNK_CONFIG_PATH),
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

    config_path = str(args_dict.get("config", DEFAULT_CHUNK_CONFIG_PATH))
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
        min_local_activity_duration=float(
            _merged_value(args, "min_local_activity_duration", 0.30)
        ),
        min_segment_duration_for_embedding=float(
            _merged_value(args, "min_segment_duration_for_embedding", 0.30)
        ),
        max_segment_duration_for_embedding=float(
            _merged_value(args, "max_segment_duration_for_embedding", 4.0)
        ),
        segment_batch_size=int(_merged_value(args, "segment_batch_size", 8)),
        max_speakers=int(_merged_value(args, "max_speakers", 50)),
        new_speaker_threshold=float(
            _merged_value(args, "new_speaker_threshold", 0.40)
        ),
        global_match_threshold=float(
            _merged_value(args, "global_match_threshold", 0.50)
        ),
        absorb_threshold=float(_merged_value(args, "absorb_threshold", 0.60)),
        min_segment_duration_for_new_speaker=float(
            _merged_value(args, "min_segment_duration_for_new_speaker", 1.00)
        ),
        min_segment_duration_for_centroid_update=float(
            _merged_value(args, "min_segment_duration_for_centroid_update", 1.50)
        ),
        update_segment_overlap_threshold=float(
            _merged_value(args, "update_segment_overlap_threshold", 0.5)
        ),
        weak_update_similarity_margin=float(
            _merged_value(args, "weak_update_similarity_margin", 0.15)
        ),
        weak_update_weight_multiplier=float(
            _merged_value(args, "weak_update_weight_multiplier", 1.00)
        ),
        probation_confirm_duration=float(
            _merged_value(args, "probation_confirm_duration", 3.0)
        ),
        min_segment_duration=float(_merged_value(args, "min_segment_duration", 0.30)),
        streaming_merge_gap=float(_merged_value(args, "streaming_merge_gap", 0.75)),
        output_dir_for_streaming=_merged_value(args, "output_dir", None),
        show_rttm=bool(_merged_value(args, "show_rttm", False)),
        debug=bool(_merged_value(args, "debug", False)),
    )
    if config.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be > 0")

    hf_cache_dir = _merged_value(args, "hf_cache_dir", None)
    if hf_cache_dir:
        config.hf_cache_dir = hf_cache_dir
    return config


__all__ = [
    "ChunkPipelineConfig",
    "DEFAULT_CHUNK_CONFIG_PATH",
    "build_arg_parser",
    "merge_args_with_config",
    "config_from_args",
    "validate_runtime_args",
]
