"""ASR 转写的配置定义、YAML 加载与 CLI 构建（独立于 diarization.config）。

约定与 diarization 侧一致：

- YAML（默认 `config/asr.yaml`）是全部调参项的唯一来源；
- CLI 仅保留运行时输入与少量开关；
- YAML 键名会校验，合法键 = `AsrConfig` 字段名 ∪ CLI 参数名。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path

import yaml

from .constants import BASE_DIR


DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "asr.yaml"


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
        field.name for field in dataclass_fields(AsrConfig)
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


def config_from_args(args: argparse.Namespace) -> AsrConfig:
    """把合并后的参数转换为 `AsrConfig` 并做合法性校验。"""

    config = AsrConfig(
        sample_rate=int(getattr(args, "sample_rate", 16000)),
        device=str(getattr(args, "device", "auto")),
        model=str(getattr(args, "model", "FunAudioLLM/Fun-ASR-Nano-2512")),
        prev_text_max_tokens=int(getattr(args, "prev_text_max_tokens", 0)),
        max_segment_duration=float(getattr(args, "max_segment_duration", 30.0)),
        window_overlap=float(getattr(args, "window_overlap", 10.0)),
    )
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
