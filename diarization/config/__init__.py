"""配置体系：通用 YAML/CLI 合并机制（base）与管线配置（pipeline）。

`merge_args_with_config` 在包级别指向管线版（3 参便捷封装）；通用机制版
（带 `default_config_path` / `config_type` 关键字参数）经 `diarization.config.base`
取用。
"""

from .base import config_from_yaml, dataclass_from_args
from .pipeline import (
    DEFAULT_CONFIG_PATH,
    RUNTIME_KEYS,
    ChunkPipelineConfig,
    build_arg_parser,
    config_from_args,
    load_config_and_runtime,
    merge_args_with_config,
    validate_runtime_args,
)

__all__ = [
    "dataclass_from_args",
    "config_from_yaml",
    "ChunkPipelineConfig",
    "DEFAULT_CONFIG_PATH",
    "RUNTIME_KEYS",
    "build_arg_parser",
    "merge_args_with_config",
    "config_from_args",
    "load_config_and_runtime",
    "validate_runtime_args",
]
