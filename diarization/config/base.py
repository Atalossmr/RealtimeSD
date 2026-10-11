"""YAML 配置与 CLI 参数合并的公共实现。

约定：

- YAML 是全部调参项的唯一来源；
- 合并优先级：argparse 默认值 < YAML 配置 < 显式 CLI 参数；
  YAML 中值为 null 的键视为"未配置"，不参与覆盖（保留环境变量缺省）。
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
    """收集 argparse 中定义过的参数名，用于校验 YAML 键名。

    Args:
        parser: 待检查的参数解析器。

    Returns:
        解析器中全部参数的 dest 集合（排除 help）。
    """

    return {
        action.dest
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help"}
    }


def _extract_provided_dests(
    parser: argparse.ArgumentParser, argv: list[str] | None
) -> set[str]:
    """根据原始命令行，判断用户显式传了哪些参数。

    Args:
        parser: 参数解析器，用于建立选项字符串到 dest 的映射。
        argv: 原始命令行 token 列表；为 None 时视为未显式传参。

    Returns:
        用户显式传入参数的 dest 集合。
    """

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
    """读取 YAML 配置文件。

    Args:
        config_path: YAML 文件路径。
        explicit: 是否为用户显式指定的配置；True 时文件不存在直接报错，
            False 时文件不存在返回空配置。

    Returns:
        YAML 顶层 mapping；文件不存在且非显式指定时返回空 dict。

    Raises:
        FileNotFoundError: explicit 为 True 且文件不存在。
        ValueError: YAML 顶层不是 mapping。
    """

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

    Args:
        parser: 命令行参数解析器。
        args: argparse 已解析的参数。
        argv: 原始命令行 token 列表，用于识别显式 CLI 参数；None 表示无。
        default_config_path: 未显式指定 --config 时使用的默认配置路径。
        config_type: 配置 dataclass 类型，其字段名计入合法 YAML 键。

    Returns:
        合并后的参数 Namespace，config 字段记录实际生效的配置路径。

    Raises:
        FileNotFoundError: 显式指定的配置文件不存在。
        ValueError: YAML 中出现未知键。
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
    # YAML 中值为 null 的键视为"未配置"，不参与覆盖——否则 config.yaml 里的
    # `model_path:`（空值）会把 MODEL_PATH 等环境变量缺省静默吞掉。
    merged.update({key: value for key, value in yaml_config.items() if value is not None})

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

    Args:
        args: 合并后的参数 Namespace。
        config_type: 目标配置 dataclass 类型。
        attr_overrides: 字段名到 args 属性名的映射，处理两者不一致的情况。

    Returns:
        构造完成的配置实例。

    Raises:
        ValueError: bool 字段收到无法解析的字符串，或字段值无法转换为
            字段默认值的类型。
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


def config_from_yaml(
    config_path: str,
    config_type: type[T],
    overrides: Optional[dict[str, object]] = None,
    *,
    extra_keys: tuple[str, ...] = (),
) -> tuple[T, dict[str, object]]:
    """无 argparse 的配置构建（库式调用入口）。

    合并优先级：YAML 配置 < overrides 中非 None 的值（即调用方显式给定）；
    字段默认值是唯一兜底来源。YAML 中值为 null 的键视为"未配置"，不参与合并；
    YAML 键与 overrides 键必须是 config_type 的合法字段名（或 extra_keys）。

    Args:
        config_path: YAML 配置文件路径，必须存在。
        config_type: 目标配置 dataclass 类型。
        overrides: 调用方显式给定的覆盖值，值为 None 的键不参与覆盖。
        extra_keys: 额外允许的键；这些键不属于 config 字段（如 model_path
            等运行时参数），单独捕获返回，由调用方自行消费。

    Returns:
        (config, extras) 二元组：config 为构建的配置实例，extras 为
        extra_keys 中实际生效的键值。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 顶层不是 mapping，或 YAML / overrides 中出现未知键。
    """

    yaml_config = _load_yaml_config(config_path, explicit=True)
    field_keys = {field.name for field in dataclass_fields(config_type)}
    valid_keys = field_keys | set(extra_keys)
    unknown_keys = sorted(set(yaml_config.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown keys in YAML config: {', '.join(unknown_keys)}")

    overrides = overrides or {}
    unknown_overrides = sorted(set(overrides.keys()) - valid_keys)
    if unknown_overrides:
        raise ValueError(f"Unknown config overrides: {', '.join(unknown_overrides)}")

    merged: dict[str, object] = {
        key: value for key, value in yaml_config.items() if value is not None
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    extras = {key: merged.pop(key) for key in extra_keys if key in merged}
    return dataclass_from_args(argparse.Namespace(**merged), config_type), extras


__all__ = ["merge_args_with_config", "dataclass_from_args", "config_from_yaml"]
