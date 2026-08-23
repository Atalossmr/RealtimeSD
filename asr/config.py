"""ASR 转写的配置定义、YAML 加载与 CLI 构建。

约定与 diarization 侧一致（公共实现收敛在 common.config）：

- YAML（默认 `config/asr.yaml`）是全部调参项的唯一来源；
- CLI 仅保留运行时输入与少量开关；
- YAML 键名会校验，合法键 = `AsrConfig` 字段名 ∪ CLI 参数名。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from common import config as common_config

from .constants import BASE_DIR


DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "asr.yaml"


@dataclass
class AsrConfig:
    """ASR 转写配置（Fun-ASR-Nano 段级推理）。"""

    # 输入音频段的采样率（须与 pipeline 导出一致）。
    sample_rate: int = 16000
    # ASR 推理设备（auto/cuda:0/cpu 等）；跟随模式下与 diarization 同时占卡，
    # 显存紧张时可单独放到别的卡或 CPU。
    device: str = "auto"
    # Fun-ASR-Nano 模型：本地目录 / ModelScope id（缓存到 pretrained/modelscope）。
    model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    # 长段窗切续写时 prev_text 的 token 预算（取本段已累计文本尾部）。
    # 注意 prev_text 语义是"本段音频前缀的转写"，仅长段窗切内部使用；
    # 跨段上下文不成立（文本与音频对不上时模型会判转写完成而提前 EOS）。
    # 预算过大（prev 估计覆盖 ≥ 窗口音频总长）同样会提前 EOS；0 = 自动
    # （≈ 窗口重叠时长 × 4 token/s，刚好覆盖重叠区）。
    prev_text_max_tokens: int = 0
    # 超过该时长的段按窗口滑窗推理：每次推进（窗口 - 重叠）秒，
    # prev = 本段已累计文本尾部（token 预算封顶），单次推理成本有界。
    max_segment_duration: float = 30.0
    # 长段滑窗的重叠时长（秒）：窗口头部带这段已转写音频供 prev 对齐，
    # 必须小于 max_segment_duration。
    window_overlap: float = 10.0


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器（仅运行时/开关参数；调参项都在 YAML）。"""

    parser = argparse.ArgumentParser(description="exporter 音频段目录的 ASR 转写")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径；全部调参项以该文件为唯一来源",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="transcript 输出目录，缺省与 segments_dir 相同",
    )
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    return parser


def merge_args_with_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """把 YAML 配置和 CLI 参数合并成最终运行参数（实现见 common.config）。

    合并优先级：argparse 默认值 < YAML 配置 < 显式 CLI 参数。
    调参项不在 CLI 上，因此它们的生效值始终以 YAML 为准。
    """

    return common_config.merge_args_with_config(
        parser,
        args,
        argv,
        default_config_path=DEFAULT_CONFIG_PATH,
        config_type=AsrConfig,
    )


def config_from_args(args: argparse.Namespace) -> AsrConfig:
    """把合并后的参数转换为 `AsrConfig` 并做合法性校验。

    字段值取自合并后的 args，缺失时落回 dataclass 字段默认值（唯一兜底来源）。
    """

    config = common_config.dataclass_from_args(args, AsrConfig)
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if config.prev_text_max_tokens < 0:
        raise ValueError("prev_text_max_tokens must be >= 0")
    if config.max_segment_duration <= 0.0:
        raise ValueError("max_segment_duration must be > 0")
    if not 0.0 <= config.window_overlap < config.max_segment_duration:
        raise ValueError(
            "window_overlap must be in [0, max_segment_duration), "
            f"got {config.window_overlap!r}"
        )
    return config


__all__ = [
    "AsrConfig",
    "DEFAULT_CONFIG_PATH",
    "build_arg_parser",
    "merge_args_with_config",
    "config_from_args",
]
