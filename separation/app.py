"""separation 离线 CLI：整段音频 + RTTM -> 各 speaker 的纯净音频段。

用法：

    python -m separation.app --wav <音频> --rttm <RTTM> --output_dir <目录> [--config ...]

按 RTTM 把含混叠的整条音频拆成各 speaker 的纯净音频段（重叠区经 TIGER
分离 + embedding 归属匹配，失败回退原始音频），wav 落在
`{output_dir}/{uri}/`，manifest 为 `{output_dir}/{uri}.segments.jsonl`。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from common.config import DEFAULT_CONFIG_PATH, load_config_and_runtime
from common.models.embedding_infer import (
    NativeERes2NetV2SegmentEmbedder,
    load_embedding_model,
)
from common.speakerlab.process.processor import FBank
from common.speakerlab.utils.fileio import load_audio
from common.utils import setup_logger
from common.utils.audio import resample_waveform_if_needed
from common.utils.device import resolve_device

from .offline import separate_by_rttm


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好全部参数的 ArgumentParser。
    """

    parser = argparse.ArgumentParser(
        description="离线 RTTM 驱动的分离切分：把含混叠的整条音频按 RTTM "
        "拆成各 speaker 的纯净音频段",
        allow_abbrev=False,
    )
    parser.add_argument("--wav", required=True, help="输入音频文件（wav/flac/mp3）")
    parser.add_argument(
        "--rttm", required=True, help="驱动切分的 RTTM（如 refined RTTM）"
    )
    parser.add_argument("--output_dir", required=True, help="segments 输出目录")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径；全部调参项以该文件为唯一来源",
    )
    parser.add_argument(
        "--device", default=None, help="运行设备，如 auto/cpu/cuda:0（默认取 YAML）"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="预训练 ERes2NetV2 checkpoint 路径（覆盖 YAML 的 model_path）",
    )
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    return parser


def _prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """重采样到目标采样率、转单声道 float32。

    与 diarization/extract/extractor.py 的 prepare_waveform 同逻辑；
    load_audio(obj_fs=sample_rate) 已重采样，此处 resample 为兜底 no-op。

    Args:
        waveform: 原始波形，一维或 (channels, samples)。
        sample_rate: 波形当前采样率（= 目标采样率 config.sample_rate）。

    Returns:
        形状 (1, samples) 的 float32 单声道波形。
    """

    waveform = resample_waveform_if_needed(waveform, sample_rate, sample_rate)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(torch.float32)


def _first_rttm_uri(rttm_path: Path) -> Optional[str]:
    """读取 RTTM 第一条 SPEAKER 行的 uri（第二列）；没有 SPEAKER 行返回 None。

    Args:
        rttm_path: RTTM 文件路径。

    Returns:
        第一条 SPEAKER 行的 uri；文件无 SPEAKER 行时返回 None。
    """

    with open(rttm_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                return parts[1]
    return None


def main() -> None:
    """CLI 入口。"""

    parser = build_parser()
    args = parser.parse_args()

    audio = Path(args.wav)
    if not audio.is_file():
        parser.error(f"音频输入不存在: {audio}")
    rttm = Path(args.rttm)
    if not rttm.is_file():
        parser.error(f"RTTM 不存在: {rttm}")

    config, extras = load_config_and_runtime(
        args.config, overrides={"device": args.device}
    )
    model_path = args.model_path or extras.get("model_path")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = str(output_dir / "logs" / "run.log")
    setup_logger(bool(args.verbose), run_log_path)
    logger.info("Run log is written to %s", run_log_path)

    uri = audio.stem
    rttm_uri = _first_rttm_uri(rttm)
    if rttm_uri is not None and rttm_uri != uri:
        logger.warning(
            "RTTM uri (%s) 与音频文件名 stem (%s) 不一致，继续处理", rttm_uri, uri
        )

    device = resolve_device(config.device)
    embedding_model = load_embedding_model(
        model_path=model_path,
        device=device,
        model_type=config.model_type,
        feat_dim=config.feat_dim,
        embedding_size=config.embedding_size,
        m_channels=config.m_channels,
    )
    feature_extractor = FBank(
        n_mels=config.feat_dim,
        sample_rate=config.sample_rate,
        mean_nor=True,
    )
    embedder = NativeERes2NetV2SegmentEmbedder(
        embedding_model=embedding_model,
        feature_extractor=feature_extractor,
        sample_rate=config.sample_rate,
        normalize_embeddings=config.normalize_embeddings,
    )

    logger.info("Processing %s with RTTM %s", audio, rttm)
    waveform = load_audio(str(audio), obj_fs=config.sample_rate)
    waveform = _prepare_waveform(waveform, config.sample_rate)

    manifest_path = separate_by_rttm(
        config, waveform, rttm, embedder, uri, str(output_dir)
    )
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        num_segments = sum(1 for line in file_obj if line.strip())
    logger.info("Wrote %d segments, manifest: %s", num_segments, manifest_path)
    print(f"manifest: {manifest_path} ({num_segments} segments)")


if __name__ == "__main__":
    main()
