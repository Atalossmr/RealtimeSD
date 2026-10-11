"""音频段输出 sink：wav 落盘 + manifest.jsonl 追加。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torchaudio

# 段消费侧接收音频段的回调签名：(uri, speaker_id, start, end, waveform[1, T])。
SegmentCallback = Callable[[str, int, float, float, torch.Tensor], None]


class WavSegmentSink:
    """默认段输出 sink：wav 落盘 + manifest.jsonl 追加。

    output_dir 即 segments 根目录：wav 在 `{output_dir}/{uri}/`，
    manifest 为 `{output_dir}/{uri}.segments.jsonl`。

    Args:
        output_dir: segments 输出根目录。
        uri: 音频标识（决定 wav 子目录与 manifest 文件名）。
        sample_rate: 落盘 wav 的采样率。

    Attributes:
        sample_rate: 落盘 wav 的采样率。
        segments_dir: wav 输出目录（构造时创建）。
        manifest_path: manifest.jsonl 路径（追加写）。
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
        """保存一个音频段：wav 落盘并向 manifest 追加一行记录。

        Args:
            uri: 音频标识。
            speaker_id: speaker id（RTTM label 按字典序映射的 int id，0 起）。
            start: 段起始绝对时刻（秒）。
            end: 段结束绝对时刻（秒）。
            waveform: 段音频（[1, T]，落盘前转到 CPU）。
        """
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
