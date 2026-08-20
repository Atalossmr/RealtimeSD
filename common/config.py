"""YAML 配置与 CLI 参数合并的公共实现（asr / diarization 共用）。

约定：

- YAML 是全部调参项的唯一来源；
- 合并优先级：argparse 默认值 < YAML 配置 < 显式 CLI 参数；
- dataclass 字段默认值是配置项的唯一兜底来源，不再手写第二份字面量。
"""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional, TypeVar

import yaml

T = TypeVar("T")


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


def merge_args_with_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
    *,
    default_config_path: Path,
    config_type: type,
) -> argparse.Namespace:
    """把 YAML 配置和 CLI 参数合并成最终运行参数。

    合并优先级：argparse 默认值 < YAML 配置 < 显式 CLI 参数。
    合法 YAML 键 = parser 参数名 ∪ config_type 的 dataclass 字段名。
    """

    provided_dests = _extract_provided_dests(parser, argv)
    args_dict = vars(args)

    config_path = str(args_dict.get("config", default_config_path))
    explicit_config = "config" in provided_dests
    yaml_config = _load_yaml_config(config_path, explicit=explicit_config)

    valid_keys = _parser_dest_set(parser) | {
        field.name for field in dataclass_fields(config_type)
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


def dataclass_from_args(
    args: argparse.Namespace,
    config_type: type[T],
    attr_overrides: Optional[dict[str, str]] = None,
) -> T:
    """按 dataclass 字段从合并后的参数构造配置。

    每个字段的值取自 args（缺失或为 None 时落回字段默认值），标量字段按
    默认值类型做 int/float/str 转换（bool 字段对字符串按内容解析，
    'true'/'false' 等大小写均可）；字段默认值是唯一兜底来源。
    attr_overrides 处理字段名与 args 属性名不一致的情况
    （如 output_dir_for_streaming <- output_dir）。
    """

    attr_overrides = attr_overrides or {}
    kwargs: dict[str, object] = {}
    for field in dataclass_fields(config_type):
        attr = attr_overrides.get(field.name, field.name)
        value = getattr(args, attr, None)
        if value is None:
            value = field.default
        elif isinstance(field.default, bool) and isinstance(value, str):
            # bool("false") is True：YAML 里带引号的 'false' 会静默反转语义，
            # 必须按字符串内容解析。
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                value = True
            elif lowered in ("0", "false", "no", "off"):
                value = False
            else:
                raise ValueError(
                    f"配置项 {field.name} 需要布尔值，收到无法解析的字符串: {value!r}"
                )
        elif field.default is not None:
            try:
                value = type(field.default)(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"配置项 {field.name} 无法转换为 {type(field.default).__name__}: "
                    f"{value!r}"
                ) from exc
        kwargs[field.name] = value
    return config_type(**kwargs)


__all__ = ["merge_args_with_config", "dataclass_from_args"]
