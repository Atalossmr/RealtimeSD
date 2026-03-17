"""说话人音轨写入工具。"""

from __future__ import annotations

import torch

from ..schema import PipelineConfig, StreamingFrameDecision


def apply_frame_decisions_to_speaker_buffers(
    *,
    decisions: list[StreamingFrameDecision],
    waveform: torch.Tensor,
    speaker_buffers,
    config: PipelineConfig,
    total_duration: float,
) -> None:
    """把帧级聚类决策直接写入说话人音轨。"""

    if not config.enable_speech_separation:
        return
    for decision in decisions:
        start = max(0.0, float(decision.start))
        end = min(total_duration, float(decision.end))
        if end <= start:
            continue

        start_sample = int(round(start * config.sample_rate))
        end_sample = int(round(end * config.sample_rate))
        if end_sample <= start_sample:
            continue
        frame_audio = waveform[0, start_sample:end_sample].cpu()
        for speaker_id in sorted({int(speaker_id) for speaker_id in decision.speakers}):
            speaker_buffers.append(
                int(speaker_id),
                frame_audio,
                start,
                overwrite=False,
            )
