"""CLI 参数定义与配置构建逻辑。

职责：

- 定义命令行参数；
- 将 YAML 配置与 CLI 参数合并；
- 构造运行时的 `PipelineConfig`。

"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .constants import BASE_DIR
from .schema import PipelineConfig


DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"


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
) -> argparse.Namespace:
    """把 YAML 配置和 CLI 参数合并成最终运行参数。"""

    provided_dests = _extract_provided_dests(parser, argv)
    args_dict = vars(args)

    config_path = str(args_dict.get("config", DEFAULT_CONFIG_PATH))
    explicit_config = "config" in provided_dests
    yaml_config = _load_yaml_config(config_path, explicit=explicit_config)

    valid_dests = _parser_dest_set(parser)
    unknown_keys = sorted(set(yaml_config.keys()) - valid_dests)
    if unknown_keys:
        raise ValueError(f"Unknown keys in YAML config: {', '.join(unknown_keys)}")

    # 合并优先级：argparse 默认值 < YAML 配置 < 显式 CLI 参数。
    # 这样可以保证：
    # - CLI 明确传入的参数总是优先；
    # - CLI 未传时优先使用 config.yaml；
    # - config.yaml 缺失项再回退到 argparse 默认值。
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


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="基于 segmentation-3.0 和原生 ERes2NetV2 的实时说话人分离脚本"
    )
    parser.add_argument(
        "--wav",
        default=None,
        help="输入单个 wav/flac/mp3、目录，或每行一个路径的文本文件",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="预训练 ERes2NetV2 checkpoint 路径；未提供时会尝试从 ModelScope 下载默认模型",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录，每个文件会写一个 append-only 的 .streaming.rttm",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径；未显式传参的选项会优先从这里读取",
    )
    parser.add_argument("--device", default="auto", help="运行设备，如 auto/cpu/cuda:0")
    parser.add_argument(
        "--model_type",
        default="eres2netv2",
        choices=["eres2netv2"],
        help="speaker encoder 类型，当前仅支持 eres2netv2",
    )

    parser.add_argument(
        "--context_left_duration",
        type=float,
        default=5.0,
        help="目标帧左侧使用多少秒历史上下文，单位秒",
    )
    parser.add_argument(
        "--context_right_duration",
        type=float,
        default=5.0,
        help="目标帧右侧使用多少秒未来上下文，单位秒",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=None,
        help="兼容旧参数；若设置，则表示总上下文时长，会平均拆到左右两侧",
    )
    parser.add_argument(
        "--advance_step",
        type=float,
        default=0.5,
        help="时间轴推进步长，同时也是每帧决策覆盖时长，单位秒",
    )
    parser.add_argument(
        "--target_activity_window_duration",
        type=float,
        default=0.5,
        help="select_target_local_indices 的统计窗口时长，单位秒",
    )
    parser.add_argument(
        "--target_min_duration",
        type=float,
        default=0.08,
        help="目标窗口内 local slot 的最小累计活跃时长阈值，低于该值视为 inactive",
    )

    parser.add_argument(
        "--new_speaker_threshold",
        type=float,
        default=0.68,
        help="低于该相似度时更倾向于新建 speaker",
    )
    parser.add_argument(
        "--max_speakers", type=int, default=10, help="允许维护的最大全局 speaker 数"
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
        "--min_segment_duration",
        type=float,
        default=0.35,
        help="streaming RTTM 允许写出的最短稳定片段时长",
    )
    parser.add_argument(
        "--min_segment_duration_for_embedding",
        type=float,
        default=0.8,
        help="允许提 embedding 的最短片段时长",
    )
    parser.add_argument(
        "--max_segment_duration_for_embedding",
        type=float,
        default=2.5,
        help="允许提 embedding 的最长片段时长",
    )
    parser.add_argument(
        "--max_segment_shift_from_center",
        type=float,
        default=1.5,
        help="候选片段中心离目标帧允许的最大偏移",
    )
    parser.add_argument(
        "--segment_batch_size",
        type=int,
        default=8,
        help="批量提取 segment embedding 时的 batch size",
    )

    parser.add_argument(
        "--global_match_threshold",
        type=float,
        default=0.7,
        help="observation 与 global centroid 主匹配阈值",
    )
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.8,
        help="全局 speaker 在更新前触发合并的相似度阈值",
    )
    parser.add_argument(
        "--min_segment_duration_for_new_speaker",
        type=float,
        default=0.6,
        help="仅当 observation 片段时长达到该阈值时才允许新建 speaker",
    )
    parser.add_argument(
        "--min_segment_duration_for_centroid_update",
        type=float,
        default=0.45,
        help="仅当 observation 片段时长达到该阈值时才允许更新 centroid",
    )
    parser.add_argument(
        "--disable_ema_update",
        action="store_true",
        help="关闭 EMA 更新，centroid 全程使用增量均值更新",
    )
    parser.add_argument(
        "--centroid_warmup_window",
        type=int,
        default=5,
        help="centroid 预热窗口大小（EMA 开启时超过该值后切换到 EMA）",
    )
    parser.add_argument(
        "--stable_update_count_threshold",
        type=int,
        default=10,
        help="speaker 被更新达到该次数后判定为 stable",
    )
    parser.add_argument(
        "--update_segment_overlap_threshold",
        type=float,
        default=0.8,
        help="连续两次用于更新 centroid 的片段允许的最大重合度",
    )
    parser.add_argument(
        "--weak_update_similarity_margin",
        type=float,
        default=0.15,
        help="overlap fallback 触发弱更新时，相对 `global_match_threshold` 还需额外超过的相似度裕量",
    )
    parser.add_argument(
        "--weak_update_weight_multiplier",
        type=float,
        default=0.25,
        help="overlap fallback 高置信度弱更新时使用的衰减权重倍率",
    )

    parser.add_argument(
        "--max_frame_speakers",
        type=int,
        default=2,
        help="每帧最多输出几个 speaker，overlap 版本默认 2",
    )
    parser.add_argument(
        "--streaming_flush_interval",
        type=float,
        default=2.0,
        help="streaming RTTM 稳定前缀的最小刷盘时长",
    )
    parser.add_argument(
        "--streaming_merge_gap",
        type=float,
        default=0.75,
        help="streaming RTTM 中同 speaker 相邻片段允许自动合并的最大间隔",
    )
    parser.add_argument(
        "--delay_short_speaker_output",
        action="store_true",
        help="开启后，未达到 stable 条件的 speaker turn 先缓存，不立即写出 RTTM",
    )
    parser.add_argument(
        "--show_rttm",
        action="store_true",
        help="在持续写入 output_dir 下 RTTM 文件的同时，把新生成的 RTTM 行同步输出到控制台",
    )
    parser.add_argument("--debug", action="store_true", help="输出窗口级 debug 信息")
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    # 说话人音轨转录与重叠分离相关参数
    parser.add_argument(
        "--enable_speech_separation",
        action="store_true",
        help="是否启用说话人音轨转录（开启后可在重叠段执行分离覆盖）",
    )
    parser.add_argument(
        "--separation_model",
        type=str,
        default="JusperLee/TIGER-speech",
        help="TIGER语音分离模型名",
    )
    parser.add_argument(
        "--min_overlap_duration_to_process",
        type=float,
        default=0.3,
        help="触发分离的最小重叠段时长（秒）",
    )
    parser.add_argument(
        "--separation_required_duration",
        type=float,
        default=3.0,
        help="TIGER需要的输入时长（秒）",
    )
    parser.add_argument(
        "--max_overlap_process_interval",
        type=float,
        default=3.0,
        help="长重叠的最大处理间隔（秒）",
    )
    parser.add_argument(
        "--export_uncertain_speaker_audio",
        action="store_true",
        help="是否导出不确定说话人的音轨",
    )
    parser.add_argument(
        "--speaker_audio_sample_rate", type=int, default=16000, help="导出音频的采样率"
    )
    parser.add_argument(
        "--speaker_audio_format", type=str, default="wav", help="导出音频的格式"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """把 CLI 参数转换为 `PipelineConfig`。"""

    # 注意：旧参数 step 已废弃并移除；frame_duration 已并入 advance_step。
    advance_step = float(_merged_value(args, "advance_step", 0.5))
    target_activity_window_duration = float(
        _merged_value(args, "target_activity_window_duration", 0.5)
    )

    new_speaker_threshold = _merged_value(args, "new_speaker_threshold", 0.68)
    centroid_warmup_window = _merged_value(args, "centroid_warmup_window", 5)

    config = PipelineConfig(
        context_left_duration=5.0,
        context_right_duration=5.0,
        advance_step=float(advance_step),
        target_activity_window_duration=float(target_activity_window_duration),
        target_min_duration=float(_merged_value(args, "target_min_duration", 0.08)),
        new_speaker_threshold=float(new_speaker_threshold),
        max_speakers=_merged_value(args, "max_speakers", 10),
        model_type=_merged_value(args, "model_type", "eres2netv2"),
        segmentation_model=_merged_value(
            args, "segmentation_model", "pyannote/segmentation-3.0"
        ),
        hf_token=_merged_value(args, "hf_token", None),
        device=_merged_value(args, "device", "auto"),
        min_segment_duration=_merged_value(args, "min_segment_duration", 0.35),
        min_segment_duration_for_embedding=_merged_value(
            args, "min_segment_duration_for_embedding", 0.8
        ),
        max_segment_duration_for_embedding=_merged_value(
            args, "max_segment_duration_for_embedding", 2.5
        ),
        max_segment_shift_from_center=_merged_value(
            args, "max_segment_shift_from_center", 1.5
        ),
        segment_batch_size=_merged_value(args, "segment_batch_size", 8),
        global_match_threshold=_merged_value(args, "global_match_threshold", 0.7),
        merge_threshold=_merged_value(args, "merge_threshold", 0.8),
        min_segment_duration_for_new_speaker=float(
            _merged_value(args, "min_segment_duration_for_new_speaker", 0.6)
        ),
        min_segment_duration_for_centroid_update=float(
            _merged_value(args, "min_segment_duration_for_centroid_update", 0.45)
        ),
        enable_ema_update=not bool(_merged_value(args, "disable_ema_update", False)),
        centroid_warmup_window=max(1, int(centroid_warmup_window)),
        stable_update_count_threshold=max(
            1, int(_merged_value(args, "stable_update_count_threshold", 10))
        ),
        update_segment_overlap_threshold=float(
            _merged_value(args, "update_segment_overlap_threshold", 0.8)
        ),
        weak_update_similarity_margin=float(
            _merged_value(args, "weak_update_similarity_margin", 0.15)
        ),
        weak_update_weight_multiplier=float(
            _merged_value(args, "weak_update_weight_multiplier", 0.25)
        ),
        max_frame_speakers=_merged_value(args, "max_frame_speakers", 2),
        streaming_flush_interval=_merged_value(args, "streaming_flush_interval", 2.0),
        streaming_merge_gap=_merged_value(args, "streaming_merge_gap", 0.75),
        delay_short_speaker_output=bool(
            _merged_value(args, "delay_short_speaker_output", False)
        ),
        output_dir_for_streaming=_merged_value(args, "output_dir", None),
        show_rttm=bool(_merged_value(args, "show_rttm", False)),
        debug=bool(_merged_value(args, "debug", False)),
        enable_speech_separation=bool(
            _merged_value(args, "enable_speech_separation", False)
        ),
        separation_model=_merged_value(
            args, "separation_model", "JusperLee/TIGER-speech"
        ),
        min_overlap_duration_to_process=float(
            _merged_value(args, "min_overlap_duration_to_process", 0.3)
        ),
        separation_required_duration=float(
            _merged_value(args, "separation_required_duration", 3.0)
        ),
        max_overlap_process_interval=float(
            _merged_value(args, "max_overlap_process_interval", 3.0)
        ),
        export_uncertain_speaker_audio=bool(
            _merged_value(args, "export_uncertain_speaker_audio", False)
        ),
        speaker_audio_sample_rate=int(
            _merged_value(args, "speaker_audio_sample_rate", 16000)
        ),
        speaker_audio_format=_merged_value(args, "speaker_audio_format", "wav"),
    )
    chunk_duration = _merged_value(args, "chunk_duration", None)
    if chunk_duration is not None:
        half = max(0.0, float(chunk_duration)) / 2.0
        config.context_left_duration = half
        config.context_right_duration = half
    else:
        config.context_left_duration = max(
            0.0, float(_merged_value(args, "context_left_duration", 5.0))
        )
        config.context_right_duration = max(
            0.0, float(_merged_value(args, "context_right_duration", 5.0))
        )
    if config.chunk_duration <= 0.0:
        raise ValueError("context_left_duration + context_right_duration must be > 0")
    if config.advance_step <= 0.0:
        raise ValueError("advance_step must be > 0")
    if config.target_activity_window_duration <= 0.0:
        raise ValueError("target_activity_window_duration must be > 0")

    hf_cache_dir = _merged_value(args, "hf_cache_dir", None)
    if hf_cache_dir:
        config.hf_cache_dir = hf_cache_dir
    return config
