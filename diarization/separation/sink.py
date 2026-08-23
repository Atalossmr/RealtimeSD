"""音频段输出 sink：wav 落盘 + manifest.jsonl 追加。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torchaudio

# ASR 侧接收音频段的回调签名：(uri, speaker_id, start, end, waveform[1, T])。
SegmentCallback = Callable[[str, int, float, float, torch.Tensor], None]


class WavSegmentSink:
    """默认段输出 sink：wav 落盘 + manifest.jsonl 追加。

    output_dir 即 segments 根目录：wav 在 `{output_dir}/{uri}/`，
    manifest 为 `{output_dir}/{uri}.segments.jsonl`。
    """

    def __init__(self, output_dir: str, uri: str, sample_rate: int):
        self.sample_rate = sample_rate
        self.segments_dir = Path(output_dir) / uri
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(output_dir) / f"{uri}.segments.jsonl"

    def __call__(
        self,
        uri: str,
        speaker_id: int,
        start: float,
        end: float,
        waveform: torch.Tensor,
    ) -> None:
        filename = f"spk{speaker_id}_{start:.3f}_{end:.3f}.wav"
        path = self.segments_dir / filename
        torchaudio.save(str(path), waveform.cpu(), self.sample_rate)
        with open(self.manifest_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(
                json.dumps(
                    {
                        "uri": uri,
                        "speaker_id": int(speaker_id),
                        "start": round(float(start), 3),
                        "end": round(float(end), 3),
                        "path": str(path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
