"""音频缓存与拼接模块。

本模块用于“说话人音轨导出”场景：

- 以 speaker_id 为键缓存若干段音频写入操作（可覆盖/可填充）；
- 运行结束时把所有片段渲染到与原音频对齐的完整时轴，并按 stable/uncertain 分组导出。

注意：

- speaker merge 发生时，上层会调用 `merge_speaker_audio` 把被合并 speaker 的音频片段转移到目标 speaker。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


class SpeakerAudioBuffer:
    """每个全局说话人独立的音频缓冲区。

    设计：
        - `buffers[speaker_id]` 是一个片段列表，每个片段为 (start_sample, audio, overwrite)。
        - 基础音轨来自帧级聚类决策（`overwrite=False`）。
        - overwrite=True 表示该片段应覆盖已写区域（例如重叠分离结果）。
        - overwrite=False 表示“仅填充未写区域”（用于基础音轨写入，避免重复覆盖）。
    """

    def __init__(self, config):
        """功能：初始化说话人音频缓冲状态。

        参数：
            config: PipelineConfig，提供导出采样率等运行参数。
        """
        self.config = config
        self.sample_rate = config.speaker_audio_sample_rate
        self.total_duration: float = 0.0
        self.total_samples: int = 0
        self.buffers: dict[int, list[tuple[int, torch.Tensor, bool]]] = {}
        self.fade_samples = int(0.005 * self.sample_rate)  # 5ms淡入淡出

    def set_total_duration(self, duration_seconds: float) -> None:
        """功能：设置整段音频总时长并同步总采样点。

        参数：
            duration_seconds: 音频总时长（秒）。
        """
        self.total_duration = max(0.0, float(duration_seconds))
        self.total_samples = int(round(self.total_duration * self.sample_rate))

    def append(
        self,
        speaker_id: int,
        audio: torch.Tensor,
        start_time: float,
        overwrite: bool = False,
    ) -> None:
        """追加音频片段到指定说话人的缓冲区。

        参数:
            speaker_id: 目标全局 speaker id
            audio: 单声道音频 (T,) 或 (1, T)
            start_time: 该片段在全局时间轴上的起点（秒）
            overwrite: 是否覆盖已写区域
        """
        if audio.ndim > 1:
            audio = audio.squeeze(0)
        audio = audio.detach().cpu().to(torch.float32)
        if audio.numel() == 0:
            return

        start_sample = int(round(float(start_time) * self.sample_rate))
        if self.total_samples > 0:
            if start_sample >= self.total_samples:
                return
            max_len = self.total_samples - start_sample
            if max_len <= 0:
                return
            if audio.shape[-1] > max_len:
                audio = audio[:max_len]

        if speaker_id not in self.buffers:
            self.buffers[speaker_id] = []
        self.buffers[speaker_id].append((start_sample, audio, bool(overwrite)))

    def erase_range(self, speaker_id: int, start_time: float, end_time: float) -> None:
        """擦除指定说话人在给定时间区间的音频。

        用途：
            streaming RTTM 在帧集合变化且有重叠时，会对旧帧尾部进行“回收”。
            这里通过删除/裁剪 speaker 的片段列表实现。
        """

        start_sec = max(0.0, float(start_time))
        end_sec = max(start_sec, float(end_time))
        start_sample = int(round(start_sec * self.sample_rate))
        end_sample = int(round(end_sec * self.sample_rate))
        if self.total_samples > 0:
            start_sample = max(0, min(start_sample, self.total_samples))
            end_sample = max(0, min(end_sample, self.total_samples))
        if end_sample <= start_sample:
            return

        segments = self.buffers.get(int(speaker_id), [])
        if not segments:
            return

        kept: list[tuple[int, torch.Tensor, bool]] = []
        for seg_start, seg_audio, overwrite in segments:
            seg_end = seg_start + int(seg_audio.shape[-1])
            if seg_end <= start_sample or seg_start >= end_sample:
                kept.append((seg_start, seg_audio, overwrite))
                continue

            left_keep_end = max(seg_start, min(seg_end, start_sample))
            if left_keep_end > seg_start:
                left_len = left_keep_end - seg_start
                kept.append((seg_start, seg_audio[:left_len], overwrite))

            right_keep_start = min(seg_end, max(seg_start, end_sample))
            if seg_end > right_keep_start:
                right_offset = right_keep_start - seg_start
                kept.append((right_keep_start, seg_audio[right_offset:], overwrite))

        self.buffers[int(speaker_id)] = kept

    def _apply_fade(
        self, audio: torch.Tensor, fade_in: bool = True, fade_out: bool = True
    ) -> torch.Tensor:
        """应用淡入淡出处理。

        说明：
            片段级别的淡入淡出可以减少拼接点击声。
        """
        audio = audio.clone()
        len_audio = audio.shape[-1]
        if fade_in and len_audio > self.fade_samples:
            fade_in_curve = torch.linspace(0, 1, self.fade_samples)
            audio[..., : self.fade_samples] *= fade_in_curve
        if fade_out and len_audio > self.fade_samples:
            fade_out_curve = torch.linspace(1, 0, self.fade_samples)
            audio[..., -self.fade_samples :] *= fade_out_curve
        return audio

    def _render_speaker_audio(self, speaker_id: int) -> torch.Tensor | None:
        """渲染指定说话人的完整时轴音频。

        渲染规则：
            - overwrite=True：无条件覆盖对应区间；
            - overwrite=False：只填充未写区域，避免重复写入导致叠加/重复。
        """

        segments = self.buffers.get(int(speaker_id), [])
        if not segments:
            return None

        full_audio = torch.zeros(self.total_samples, dtype=torch.float32)
        written_mask = torch.zeros(self.total_samples, dtype=torch.bool)

        for start_sample, seg, overwrite in sorted(segments, key=lambda item: item[0]):
            seg = self._apply_fade(seg, fade_in=True, fade_out=True)
            end_sample = min(self.total_samples, start_sample + seg.shape[-1])
            if end_sample <= start_sample:
                continue

            target = full_audio[start_sample:end_sample]
            seg_part = seg[: end_sample - start_sample]
            mask_slice = written_mask[start_sample:end_sample]

            if overwrite:
                target.copy_(seg_part)
                mask_slice.fill_(True)
                continue

            fill_mask = ~mask_slice
            if fill_mask.any():
                target[fill_mask] = seg_part[fill_mask]
                mask_slice[fill_mask] = True

        return full_audio

    def merge_speaker_audio(self, from_speaker_id: int, to_speaker_id: int) -> None:
        """把一个 speaker 的所有音频片段合并到另一个 speaker。

        需求对应：
            当 speaker merge 时，如果被合并 speaker 在 merge 前不属于 stable（即 RTTM 未写出），
            需要把它的音频也合并到目标 speaker，保证最终导出的 stable 音轨包含完整语音。

        注意：
            - 这里不做时间上的再切分，直接迁移片段列表；
            - 后续渲染阶段会统一处理 overwrite/fill 的行为。
        """

        from_speaker_id = int(from_speaker_id)
        to_speaker_id = int(to_speaker_id)
        if from_speaker_id == to_speaker_id:
            return

        segments = list(self.buffers.get(from_speaker_id, []))
        self.buffers.pop(from_speaker_id, None)
        if not segments:
            return
        self.buffers.setdefault(to_speaker_id, []).extend(segments)

    def export_grouped(
        self,
        uri: str,
        export_root: Path,
        speaker_meta: dict[int, dict[str, int | float | bool]],
        export_uncertain: bool,
    ) -> None:
        """按稳定/不确定分组导出说话人音轨。"""

        stable_dir = export_root / "stable"
        uncertain_dir = export_root / "uncertain"
        stable_dir.mkdir(parents=True, exist_ok=True)
        uncertain_dir.mkdir(parents=True, exist_ok=True)

        # 清理同一 URI 的历史导出文件，避免旧命名残留造成误解。
        for pattern in (
            f"{uri}_speaker_*.{self.config.speaker_audio_format}",
            f"{uri}_spk_*_stable.{self.config.speaker_audio_format}",
            f"{uri}_spk_*_uncertain.{self.config.speaker_audio_format}",
            f"{uri}_spk_internal_*_uncertain.{self.config.speaker_audio_format}",
        ):
            for old_path in stable_dir.glob(pattern):
                old_path.unlink(missing_ok=True)
            for old_path in uncertain_dir.glob(pattern):
                old_path.unlink(missing_ok=True)

        if self.total_samples <= 0:
            return

        for speaker_id in sorted(self.buffers.keys()):
            full_audio = self._render_speaker_audio(speaker_id)
            if full_audio is None:
                continue

            meta = speaker_meta.get(int(speaker_id), {})
            is_stable = bool(meta.get("is_stable", False))
            output_speaker_id = int(meta.get("output_speaker_id", int(speaker_id)))

            if is_stable:
                output_path = (
                    stable_dir
                    / f"{uri}_spk_{output_speaker_id}_stable.{self.config.speaker_audio_format}"
                )
            elif export_uncertain:
                internal_speaker_id = int(meta.get("speaker_id", int(speaker_id)))
                output_path = (
                    uncertain_dir
                    / f"{uri}_spk_internal_{internal_speaker_id}_uncertain.{self.config.speaker_audio_format}"
                )
            else:
                continue

            torchaudio.save(str(output_path), full_audio.unsqueeze(0), self.sample_rate)
