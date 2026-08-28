"""管线的配置定义、YAML 加载与 CLI 构建。

约定：

- YAML 是全部调参项的唯一来源；
- CLI 仅保留运行时输入、模型/环境参数与少量开关；
- YAML 键名会校验，合法键 = `ChunkPipelineConfig` 字段名 ∪ CLI 参数名。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..constants import BASE_DIR
from . import base as common_config


DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


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
    """chunk 管线的统一配置（字段按模块分组：环境 / extract / cluster / 输出）。

    逐项调参与标定依据见 `config/README.md`。
    """

    # ---- 运行环境 ----
    device: str = "auto"
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
    # 窗口推进步长（秒）；hop < chunk 为重叠滑窗，每窗只提交中段 hop 秒。
    hop_duration: float = 5.0

    # ---- extract 阶段：local track 构造与 embedding 提取 ----
    min_local_activity_duration: float = 0.30
    min_segment_duration_for_embedding: float = 0.30
    max_segment_duration_for_embedding: float = 4.0
    segment_batch_size: int = 4

    # ---- cluster 阶段：后端选择与通用 ----
    clustering_backend: str = "streaming"

    # ---- cluster 阶段：streaming 后端（全局 speaker 匹配与 centroid 维护） ----
    max_speakers: int = 50
    new_speaker_threshold: float = 0.50
    global_match_threshold: float = 0.55
    min_segment_duration_for_new_speaker: float = 1.00
    min_segment_duration_for_centroid_update: float = 1.50
    merge_threshold: float = 0.70

    # ---- cluster 阶段：ahc 后端（离线层次聚类） ----
    ahc_similarity_threshold: float = 0.50
    ahc_linkage: str = "average"

    # ---- cluster 阶段：后处理 ----
    post_merge_min_count: int = 8
    post_merge_min_similarity: float = 0.0

    # ---- 分段音频导出（接流式 ASR；仅 streaming 后端生效） ----
    # 逐 speaker 音频段导出（wav + manifest）：无重叠纯切片，有重叠走 TIGER 语音分离
    separation_enabled: bool = True
    # TIGER 分离模型（Hugging Face，固定 2 路输出、16kHz）。
    separation_model: str = "JusperLee/TIGER-speech"
    # 能量门控：音轨 RMS / 混合 RMS 低于该比值判伪源，整窗回退原始音频。
    separation_energy_ratio: float = 0.05
    # 2x2 匹配的最小余弦相似度，低于则回退原始音频（仅防分离崩溃，不宜调高）。
    separation_min_match_similarity: float = 0.10

    # ---- RTTM 输出 ----
    min_segment_duration: float = 0.30
    segment_merge_gap: float = 0.25
    output_dir: Optional[str] = None
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
        help="输出目录；RTTM 写入其 rttm/ 子目录（raw 级 .raw.rttm，"
        "最终输出 .refined.rttm + .speakers.json），"
        "embeddings npz 在 embeddings/，分段音频在 segments/，日志在 logs/",
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

    return common_config.merge_args_with_config(
        parser,
        args,
        argv,
        default_config_path=DEFAULT_CONFIG_PATH,
        config_type=ChunkPipelineConfig,
    )


def _validate_and_normalize(config: ChunkPipelineConfig) -> ChunkPipelineConfig:
    """数值校验 + hf_cache_dir 相对路径按仓库根解析（config_from_args 与
    load_config 共用，规则只维护这一份）。"""

    if config.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be > 0")
    if config.post_merge_min_count < 0:
        raise ValueError("post_merge_min_count must be >= 0")
    if config.post_merge_min_similarity < 0:
        raise ValueError("post_merge_min_similarity must be >= 0")
    if config.hop_duration <= 0.0 or config.hop_duration > config.chunk_duration:
        raise ValueError("hop_duration must be in (0, chunk_duration]")

    if config.hf_cache_dir:
        # YAML 中的相对路径（如 ./pretrained/huggingface）按仓库根目录解析，
        # 避免从其他工作目录运行时缓存失效或重复下载。
        hf_path = Path(config.hf_cache_dir).expanduser()
        if not hf_path.is_absolute():
            hf_path = BASE_DIR / hf_path
        config.hf_cache_dir = str(hf_path)
    return config


def config_from_args(args: argparse.Namespace) -> ChunkPipelineConfig:
    """把合并后的参数转换为 `ChunkPipelineConfig`。

    字段值取自合并后的 args，缺失时落回 dataclass 字段默认值（唯一兜底来源）。
    """

    return _validate_and_normalize(
        common_config.dataclass_from_args(args, ChunkPipelineConfig)
    )


# YAML 允许的运行时键：不属于 ChunkPipelineConfig 字段（CLI 下是 argparse
# 参数），库式加载时单独捕获返回，由调用方消费。
RUNTIME_KEYS = ("model_path",)


def load_config(
    config_path: str | Path | None = None,
    overrides: Optional[dict[str, object]] = None,
) -> ChunkPipelineConfig:
    """库式调用的配置构建入口（不经过 argparse）。

    config_path 缺省为仓库自带 `config/config.yaml`；合并语义为
    YAML < overrides 中非 None 值 < 字段默认值兜底（与 CLI 的
    "YAML < 显式 CLI"对应）。YAML 中的运行时键（RUNTIME_KEYS）被忽略，
    需要时请改用 `load_config_and_runtime`。
    """

    config, _ = load_config_and_runtime(config_path, overrides)
    return config


def load_config_and_runtime(
    config_path: str | Path | None = None,
    overrides: Optional[dict[str, object]] = None,
) -> tuple[ChunkPipelineConfig, dict[str, object]]:
    """`load_config` + 捕获 YAML 中实际生效的运行时键（当前仅 model_path）。"""

    path = str(config_path) if config_path is not None else str(DEFAULT_CONFIG_PATH)
    config, extras = common_config.config_from_yaml(
        path, ChunkPipelineConfig, overrides, extra_keys=RUNTIME_KEYS
    )
    return _validate_and_normalize(config), extras


__all__ = [
    "ChunkPipelineConfig",
    "DEFAULT_CONFIG_PATH",
    "RUNTIME_KEYS",
    "build_arg_parser",
    "merge_args_with_config",
    "config_from_args",
    "load_config",
    "load_config_and_runtime",
    "validate_runtime_args",
]
