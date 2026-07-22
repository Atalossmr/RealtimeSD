"""实时说话人识别主流程模块。

该文件负责把各子模块串成一条可运行的实时管线：

- segmentation: 运行 pyannote segmentation，挑选活跃 local slot，并构造 observation
- clustering: local -> global speaker 分配、centroid 更新、以及 speaker merge 事件产生
- streaming: 把离散的逐帧决策写为 RTTM turn
- (可选) separation: 对重叠语音片段触发分离，覆盖写入说话人音轨

注意：

- speaker merge 只允许合并 unstable speaker，并按稳定性变化决定 RTTM 的补写策略；
- merge 时若 small speaker 不稳定，则其音频片段也会合并到目标 speaker。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from speakerlab.process.processor import FBank
from speakerlab.utils.fileio import load_audio

from ..audio import SpeakerAudioBuffer
from ..clustering import IncrementalCentroidClusterer
from ..models import (
    NativeERes2NetV2SegmentEmbedder,
    PyannoteStreamingSegmentation,
    TIGERSeparator,
    load_embedding_model,
)
from .merge_ops import handle_pending_speaker_merges
from .rewrite_ops import apply_frame_decisions_to_speaker_buffers
from .window_ops import (
    slice_window,
    target_frame_index,
    target_frame_speakers,
)
from ..separation.overlap_processor import process_overlap_segment
from ..schema import (
    PipelineConfig,
    SegmentObservation,
    StreamingFrameDecision,
    WindowDebugInfo,
)
from ..segmentation import SegmentBuilder
from ..streaming import StreamingRTTMWriter, quantize_decision_time
from ..utils import resample_waveform_if_needed, resolve_device


logger = logging.getLogger(__name__)


class NativeOnlineSpeakerDiarization:
    """实时主控类。

    当前实现的主要流程为：
    1. 读音频；
    2. 切上下文；
    3. 跑 segmentation；
    4. 取目标帧活跃 local slot；
    5. 生成 observation；
    6. 做 local -> global 映射；
    7. 每个窗口解析后立即进入后续流程；
    8. 写 streaming RTTM。
    """

    def __init__(self, config: PipelineConfig, embedding_model_path: Optional[str]):
        """功能：初始化实时说话人识别主流程对象。

        参数：
            config: 全链路运行配置。
            embedding_model_path: 说话人 embedding 模型路径，可为空。
        """
        # 保存配置，并先解析最终运行设备。
        self.config = config
        self.device = resolve_device(config.device)

        # 加载说话人 embedding 模型。
        self.embedding_model = load_embedding_model(
            model_path=embedding_model_path,
            device=self.device,
            model_type=config.model_type,
            feat_dim=config.feat_dim,
            embedding_size=config.embedding_size,
            m_channels=config.m_channels,
        )

        # 构建 FBank 特征提取器，供 ERes2NetV2 使用。
        self.feature_extractor = FBank(
            n_mels=config.feat_dim,
            sample_rate=config.sample_rate,
            mean_nor=True,
        )

        # 构建 segmentation 推理器。
        # 这里使用总上下文时长作为 segmentation 输入长度。
        self.segmentation = PyannoteStreamingSegmentation(
            model_name=config.segmentation_model,
            duration=config.chunk_duration,
            batch_size=config.segmentation_batch_size,
            device=self.device,
            cache_dir=config.hf_cache_dir,
            use_auth_token=config.hf_token,
        )

        # 构建片段 embedding 提取器。
        self.embedder = NativeERes2NetV2SegmentEmbedder(
            embedding_model=self.embedding_model,
            feature_extractor=self.feature_extractor,
            sample_rate=config.sample_rate,
            normalize_embeddings=config.normalize_embeddings,
        )

        # 构建 observation 生成器。
        self.segment_builder = SegmentBuilder(config, self.embedder)

        # 为当前音频初始化全局 speaker 分配器。
        self._reset_clusterer()

        # 初始化语音分离相关组件。
        self.speaker_buffers = SpeakerAudioBuffer(config)
        self.is_overlapping = False
        self.overlap_start = 0.0
        self.overlap_duration = 0.0
        self.prev_frame_speakers: set[int] = set()
        self.total_duration = 0.0
        self._separation_waveform: Optional[torch.Tensor] = None
        self._speaker_export_meta: dict[int, dict[str, int | float | bool]] = {}

        # 初始化TIGER分离模型
        self.separator = None
        if config.enable_speech_separation:
            self.separator = TIGERSeparator(
                model_name=config.separation_model,
                cache_dir=config.hf_cache_dir,
                device=self.device,
            )
            logger.info(
                "[separation] enabled model=%s cache_dir=%s",
                config.separation_model,
                config.hf_cache_dir,
            )
        else:
            logger.info("[separation] disabled")

    def _reset_clusterer(self) -> None:
        """为新音频重置实时聚类状态。"""

        self.clusterer = IncrementalCentroidClusterer(
            new_speaker_threshold=self.config.new_speaker_threshold,
            max_speakers=self.config.max_speakers,
            global_match_threshold=self.config.global_match_threshold,
            merge_threshold=self.config.merge_threshold,
            min_segment_duration_for_new_speaker=self.config.min_segment_duration_for_new_speaker,
            min_segment_duration_for_centroid_update=self.config.min_segment_duration_for_centroid_update,
            enable_ema_update=self.config.enable_ema_update,
            centroid_warmup_window=self.config.centroid_warmup_window,
            update_segment_overlap_threshold=self.config.update_segment_overlap_threshold,
            weak_update_similarity_margin=self.config.weak_update_similarity_margin,
            weak_update_weight_multiplier=self.config.weak_update_weight_multiplier,
        )

    def reset(self) -> None:
        """对外暴露的状态清理接口。"""

        self._reset_clusterer()
        self.speaker_buffers = SpeakerAudioBuffer(self.config)
        self.is_overlapping = False
        self.overlap_start = 0.0
        self.overlap_duration = 0.0
        self.prev_frame_speakers = set()
        self.total_duration = 0.0
        self._separation_waveform = None
        self._speaker_export_meta = {}

    def _handle_pending_speaker_merges(
        self,
        *,
        streaming_logger: StreamingRTTMWriter,
        commit_time: float,
    ) -> None:
        """消费 clusterer 累计的 speaker merge 事件，并同步到 streaming/audio。

        关键点：
            - 只允许 unstable speaker 被合并：如果 small 在 merge 前已经 stable，则跳过（不合并）。
            - small 不稳定时：
                - 合并其音频到 large（保证最终 stable 音轨不丢语音）
                - 让 streaming 层按规则决定 RTTM 的补写方式
        """

        handle_pending_speaker_merges(
            clusterer=self.clusterer,
            config=self.config,
            speaker_buffers=self.speaker_buffers,
            streaming_logger=streaming_logger,
            commit_time=commit_time,
        )

    def _slice_waveform_by_time(self, start: float, end: float) -> torch.Tensor:
        """按绝对时间从当前处理波形中裁剪单声道音频。

        这里直接从当前整段波形取片段，避免滑窗重复缓存带来的时间错位。
        """

        if self._separation_waveform is None:
            return torch.zeros(1, 0, dtype=torch.float32)
        start_sec = max(0.0, float(start))
        end_sec = max(start_sec, float(end))
        start_sample = int(round(start_sec * self.config.sample_rate))
        end_sample = int(round(end_sec * self.config.sample_rate))
        start_sample = max(0, min(start_sample, self._separation_waveform.shape[1]))
        end_sample = max(0, min(end_sample, self._separation_waveform.shape[1]))
        if end_sample <= start_sample:
            return torch.zeros(1, 0, dtype=torch.float32)
        return self._separation_waveform[:, start_sample:end_sample].detach().cpu()

    def _format_log_payload(self, payload: object) -> str:
        """把结构化对象转成多行 JSON 风格文本，方便阅读 debug 日志。"""

        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _log_structured(
        self,
        level: int,
        prefix: str,
        title: str,
        payload: object,
    ) -> None:
        """统一输出多行结构化日志。"""

        logger.log(
            level, "%s %s:\n%s", prefix, title, self._format_log_payload(payload)
        )

    def _slice_window(
        self,
        waveform: torch.Tensor,
        target_time: float,
    ) -> tuple[torch.Tensor, float]:
        """围绕目标帧截取固定长度上下文，并在边界处补零。

        这里始终返回固定长度 chunk，原因是 segmentation 模型按固定长度输入工作。
        """

        return slice_window(
            config=self.config, waveform=waveform, target_time=target_time
        )

    def _target_frame_index(
        self,
        absolute_centers: np.ndarray,
        target_time: float,
    ) -> Optional[int]:
        """找到最接近目标时刻的 segmentation 帧索引。"""

        return target_frame_index(absolute_centers, target_time)

    def _target_frame_speakers(
        self,
        segmentation_scores: np.ndarray,
        absolute_centers: np.ndarray,
        target_time: float,
        target_frame_idx: Optional[int],
        local_to_global: dict[int, int],
    ) -> list[int]:
        """把目标时间附近活跃的 local slot 映射成最终 global speaker。

        overlap 版本这里不再只看 target 对应单帧的瞬时分数，
        而是围绕 target_time 做一个和 `advance_step` 对齐的小时间窗汇总。

        这样做有两个目的：
        - 避免第二说话人只因为某一帧瞬时分数稍低就被忽略；
        - 让 overlap 输出更依赖“持续活跃时长”，而不是 17ms 单帧波动。
        """

        return target_frame_speakers(
            config=self.config,
            segment_builder=self.segment_builder,
            segmentation_scores=segmentation_scores,
            absolute_centers=absolute_centers,
            target_time=target_time,
            target_frame_idx=target_frame_idx,
            local_to_global=local_to_global,
        )

    def _process_overlap_segment(self, start: float, end: float):
        """处理重叠段：补全音频、分离、匹配，并覆盖写入说话人音轨。"""
        process_overlap_segment(
            start=start,
            end=end,
            config=self.config,
            separator=self.separator,
            embedder=self.embedder,
            clusterer=self.clusterer,
            speaker_buffers=self.speaker_buffers,
            slice_waveform_by_time=self._slice_waveform_by_time,
            total_duration=self.total_duration,
            logger=logger,
        )

    def _flush_overlap_segment(self, end_time: float) -> bool:
        """按当前重叠状态尝试触发一次分离并切分重叠段。"""

        if not self.is_overlapping:
            return False

        segment_start = float(self.overlap_start)
        segment_end = float(end_time)
        segment_duration = segment_end - segment_start
        if segment_duration < self.config.min_overlap_duration_to_process:
            return False

        self._process_overlap_segment(segment_start, segment_end)
        self.overlap_start = segment_end
        self.overlap_duration = 0.0
        return True

    def _log_debug_window(
        self,
        *,
        window_end_sample: int,
        target_time: float,
        chunk_start_time: float,
        seg_scores: np.ndarray,
        observations: list[SegmentObservation],
        local_to_global: dict[int, int],
        debug_info: WindowDebugInfo,
        absolute_centers: np.ndarray,
        emitted_count: int,
    ) -> None:
        """输出窗口级调试信息。"""

        target_frame_idx = self._target_frame_index(absolute_centers, target_time)
        target_frame_scores = (
            [round(float(score), 6) for score in seg_scores[target_frame_idx].tolist()]
            if target_frame_idx is not None and seg_scores.size > 0
            else []
        )

        debug_summary = {
            "window_end_sec": round(
                float(window_end_sample / self.config.sample_rate), 3
            ),
            "target_time": round(float(target_time), 3),
            "chunk": {
                "start": round(float(chunk_start_time), 3),
                "end": round(float(chunk_start_time + self.config.chunk_duration), 3),
            },
            "segmentation_summary": {
                "shape": [int(dim) for dim in seg_scores.shape],
                "min": round(float(np.min(seg_scores)), 6),
                "max": round(float(np.max(seg_scores)), 6),
                "mean": round(float(np.mean(seg_scores)), 6),
                "target_frame_idx": (
                    int(target_frame_idx) if target_frame_idx is not None else None
                ),
                "target_frame_scores": target_frame_scores,
            },
            "window_state": {
                "observations": int(len(observations)),
                "embedded": int(
                    debug_info.get("num_embedded_observations", len(observations))
                ),
                "emitted": int(emitted_count),
            },
            "assignment": {
                "local_to_global": {
                    str(local_idx): int(global_id)
                    for local_idx, global_id in sorted(local_to_global.items())
                },
                "target_local_activity": self.segment_builder.summarize_target_local_activity(
                    seg_scores,
                    absolute_centers,
                    target_time,
                ),
                "local_assignments": debug_info.get("local_assignments", []),
            },
            "centroids": {
                # 改成直接索引后，Pylance 能保留 `int` 精确类型，不会退化成 `object`。
                "before": int(debug_info["num_centroids_before"]),
                "after": int(debug_info["num_centroids_after"]),
            },
        }
        self._log_structured(logging.DEBUG, "[debug]", "window_summary", debug_summary)

        if observations:
            self._log_structured(
                logging.DEBUG,
                "[debug]",
                "observations",
                [
                    {
                        "local": int(obs.local_idx),
                        "start": round(float(obs.start), 3),
                        "end": round(float(obs.end), 3),
                        "duration": round(float(obs.duration), 3),
                        "score_at_target": round(float(obs.score_at_target), 4),
                        "mean_activity": round(float(obs.mean_activity), 4),
                        "speech_ratio": round(float(obs.speech_ratio), 4),
                        "allow_centroid_update": bool(obs.allow_centroid_update),
                        "selection_mode": obs.selection_mode,
                    }
                    for obs in observations
                ],
            )

        if debug_info["new_speakers"]:
            self._log_structured(
                logging.DEBUG,
                "[debug]",
                "new_speakers",
                debug_info["new_speakers"],
            )
        if debug_info["merged_speakers"]:
            self._log_structured(
                logging.DEBUG,
                "[debug]",
                "merged_speakers",
                debug_info["merged_speakers"],
            )
        if debug_info["updated_speakers"]:
            self._log_structured(
                logging.DEBUG,
                "[debug]",
                "updated_speakers",
                debug_info["updated_speakers"],
            )
        if debug_info["skipped_updates"]:
            self._log_structured(
                logging.DEBUG,
                "[debug]",
                "skipped_updates",
                debug_info["skipped_updates"],
            )
        if debug_info["global_speakers"]:
            self._log_structured(
                logging.INFO,
                "[runtime]",
                "current_global_speakers",
                debug_info["global_speakers"],
            )
        if debug_info["local_assignments"]:
            self._log_structured(
                logging.INFO,
                "[runtime]",
                "frame_assignments",
                debug_info["local_assignments"],
            )

    def _apply_frame_decisions_to_speaker_buffers(
        self,
        decisions: list[StreamingFrameDecision],
        waveform: torch.Tensor,
    ) -> None:
        """把帧级聚类决策直接写入说话人音轨。"""
        apply_frame_decisions_to_speaker_buffers(
            decisions=decisions,
            waveform=waveform,
            speaker_buffers=self.speaker_buffers,
            config=self.config,
            total_duration=self.total_duration,
        )

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        streaming_log_path: str,
        uri: Optional[str] = None,
    ) -> None:
        """按实时方式处理整段波形并持续写出 RTTM。

        注意：
        - 模型看到的是“目标帧附近的一整段上下文”；
        - 但真正输出的只有这个目标帧对应的说话人结果；
        - 上下文只是为了帮助当前帧做判断，不会整段直接输出。
        """

        # 先把输入统一到 pipeline 约定的采样率和声道数。
        waveform = resample_waveform_if_needed(
            waveform, sample_rate, self.config.sample_rate
        )
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(torch.float32)

        self.total_duration = waveform.shape[1] / self.config.sample_rate
        self._separation_waveform = waveform
        total_samples = waveform.shape[1]
        step_samples = max(
            1, int(round(self.config.advance_step * self.config.sample_rate))
        )
        if self.config.enable_speech_separation:
            # 先固定输出总时长，导出时会自动在非说话区填充静音。
            self.speaker_buffers.set_total_duration(self.total_duration)

        # `emitted_buckets` 用于保证同一个目标时间桶只输出一次。
        emitted_buckets: set[float] = set()

        streaming_logger = StreamingRTTMWriter(
            streaming_log_path,
            uri or "unknown",
            self.config.min_segment_duration,
            self.config.streaming_flush_interval,
            self.config.streaming_merge_gap,
            self.config.delay_short_speaker_output,
            self.config.show_rttm,
            stable_speaker_ids_provider=lambda: self.clusterer.stable_speaker_ids(
                self.config.stable_update_count_threshold
            ),
        )

        # 按 advance_step 构造所有窗口结束点。
        window_ends = list(
            range(step_samples, total_samples + step_samples, step_samples)
        )
        if not window_ends or window_ends[-1] != total_samples:
            window_ends.append(total_samples)

        def process_window(window_end_sample: int) -> None:
            # `window_time` 表示当前实时推进位置。
            window_time = (
                min(window_end_sample, total_samples) / self.config.sample_rate
            )

            # `target_time` 才是当前真正要判定的时刻。
            # 这里保留“落在当前推进点前半个 advance_step”的简单策略。
            target_time = min(
                self.total_duration,
                max(0.0, window_time - 0.5 * self.config.advance_step),
            )

            # 第一步：围绕目标帧切出固定上下文。
            chunk, chunk_start_time = self._slice_window(waveform, target_time)

            # 第二步：对整个上下文跑 segmentation。
            seg_scores, centers = self.segmentation(chunk, self.config.sample_rate)
            if seg_scores.size == 0:
                return
            absolute_centers = chunk_start_time + centers

            # 第三步：在 target_time 附近若干帧内统计各 local slot 的活跃总时长。
            target_local_indices = self.segment_builder.select_target_local_indices(
                seg_scores,
                absolute_centers,
                target_time,
            )
            # if target_time == 12.25:
            #     print("local speaker:", target_local_indices)

            # 第四步：构造候选片段并批量提取 embedding。
            candidates = self.segment_builder.build_candidates(
                window_id=self.clusterer.window_counter,
                segmentation=seg_scores,
                absolute_centers=absolute_centers,
                target_local_indices=target_local_indices,
                reference_center=target_time,
            )

            observations = self.segment_builder.embed_candidates(
                chunk=chunk,
                chunk_start_time=chunk_start_time,
                candidates=candidates,
            )

            # 第五步：把当前目标帧窗口送入全局 speaker 分配器。
            # stable 判定改为“被 centroid 更新达到固定次数”，
            # 以此约束 merge 仅可合并 unstable speaker。
            self.clusterer.set_stable_speakers(
                self.clusterer.stable_speaker_ids(
                    self.config.stable_update_count_threshold
                )
            )

            window = self.clusterer.start_window(
                target_time=target_time,
                target_local_indices=target_local_indices,
                chunk_start_time=chunk_start_time,
                segmentation=seg_scores,
                absolute_centers=absolute_centers,
                observations=observations,
            )
            resolved = self.clusterer.push_window(window)

            # 这里不提前写音频；统一在 resolved frame 决策后写入，
            # 这样能确保写入的 speaker id 已经是最终 local->global 映射结果。

            # 第六步：当前窗口一旦解析完成，立即进入后续流程。
            commit_time = float(window_time)

            # 处理 speaker merge（必须在输出决策前同步，避免后续写错 speaker id）。
            self._handle_pending_speaker_merges(
                streaming_logger=streaming_logger,
                commit_time=commit_time,
            )

            # 第七步：只读取已解析窗口中“目标帧位置”的最终 speaker 决策。
            resolved_target_frame_idx = self._target_frame_index(
                resolved.window.absolute_centers,
                resolved.window.target_time,
            )
            speakers = self._target_frame_speakers(
                resolved.window.segmentation,
                resolved.window.absolute_centers,
                resolved.window.target_time,
                resolved_target_frame_idx,
                resolved.local_to_global,
            )

            self._log_structured(
                logging.INFO,
                "[runtime]",
                "frame_decision",
                {
                    "target_time": round(float(resolved.window.target_time), 3),
                    "local_to_global": {
                        str(local_idx): int(global_id)
                        for local_idx, global_id in sorted(
                            resolved.local_to_global.items()
                        )
                    },
                    "frame_speakers": [int(speaker_id) for speaker_id in speakers],
                },
            )

            # 第八步：把目标时刻量化到统一时间桶，避免重复写同一帧。
            quantized_time = quantize_decision_time(
                self.config.advance_step,
                resolved.window.target_time,
            )
            if quantized_time in emitted_buckets:
                return
            emitted_buckets.add(quantized_time)

            # 第九步：把该目标帧 speaker 决策构造成统一帧对象。
            half_duration = 0.5 * self.config.advance_step
            frame_start = max(0.0, quantized_time - half_duration)
            frame_end = min(self.total_duration, quantized_time + half_duration)
            decisions = [
                StreamingFrameDecision(
                    time=quantized_time,
                    start=frame_start,
                    end=frame_end,
                    speakers=speakers,
                )
            ]

            # 第十步：先按帧级聚类结果写基础音轨，再处理分离覆盖。
            self._apply_frame_decisions_to_speaker_buffers(decisions, waveform)

            # 第十一步 语音分离流程（覆盖写入）。
            if self.config.enable_speech_separation:
                detection_time = target_time - 0.5 * self.config.advance_step

                # 按聚类后帧决策是否出现多个 speaker 来判断 overlap
                current_frame_speakers = {int(speaker_id) for speaker_id in speakers}
                has_overlap = len(current_frame_speakers) >= 2
                prev_frame_speakers = set(self.prev_frame_speakers)
                prev_frame_had_overlap = len(prev_frame_speakers) >= 2
                speakers_changed_from_prev = (
                    bool(prev_frame_speakers)
                    and current_frame_speakers != prev_frame_speakers
                )

                # 情况 1：首次进入重叠，记录重叠起点。
                if has_overlap and not self.is_overlapping:
                    self.is_overlapping = True
                    self.overlap_start = detection_time

                    logger.debug(
                        "[separation] overlap_start target_time=%.3f",
                        target_time,
                    )

                if self.is_overlapping:
                    self.overlap_duration = max(
                        0.0,
                        detection_time - self.overlap_start,
                    )

                    # 情况 2/3：在重叠期内优先按“说话人变化”切段，否则按固定间隔切段。
                    flush_reason: Optional[str] = None
                    if prev_frame_had_overlap and speakers_changed_from_prev:
                        flush_reason = "speaker_switch"
                    elif (
                        has_overlap
                        and self.overlap_duration
                        > self.config.max_overlap_process_interval
                    ):
                        flush_reason = "interval"

                    if flush_reason and self._flush_overlap_segment(
                        end_time=detection_time
                    ):
                        if flush_reason == "speaker_switch":
                            logger.debug(
                                "[separation] overlap_speaker_switch prev=%s curr=%s split_end=%.3f",
                                sorted(prev_frame_speakers),
                                sorted(current_frame_speakers),
                                detection_time - self.config.advance_step,
                            )
                        else:
                            logger.debug(
                                "[separation] overlap_chunked next_start=%.3f",
                                detection_time - self.config.advance_step,
                            )

                    # 情况 4：重叠结束仅复位状态，不再单独触发分离。
                    if not has_overlap:
                        logger.debug(
                            "[separation] overlap_end target_time=%.3f",
                            detection_time,
                        )
                        self.is_overlapping = False
                        self.overlap_duration = 0.0

                self.prev_frame_speakers = set(current_frame_speakers)

            # 第十二步：帧决策继续喂给 streaming，独立产出 RTTM。
            streaming_logger.consume(
                decisions,
                stable_until=max(0.0, commit_time),
            )

            # consume 过程中可能结束了某些 speaker 的 active turn，从而触发 merge 延后补写；
            # 因此这里再跑一次 merge 事件消费，避免 merge 累积到下个窗口才生效。
            self._handle_pending_speaker_merges(
                streaming_logger=streaming_logger,
                commit_time=commit_time,
            )

            # 如果打开 debug，这里顺手把当前窗口的核心上下文、observation 和分配结果打出来。
            if self.config.debug:
                self._log_debug_window(
                    window_end_sample=window_end_sample,
                    target_time=resolved.window.target_time,
                    chunk_start_time=resolved.window.chunk_start_time,
                    seg_scores=resolved.window.segmentation,
                    observations=resolved.window.observations,
                    local_to_global=resolved.local_to_global,
                    debug_info=resolved.debug_info,
                    absolute_centers=resolved.window.absolute_centers,
                    emitted_count=len(decisions),
                )

        # 按实时推进顺序处理全部窗口（每个窗口解析后立即提交后续流程）。
        for window_end_sample in window_ends:
            process_window(window_end_sample)

        # 最后把所有还没写完的 turn 刷到 RTTM 文件中。
        streaming_logger.finalize()
        self._speaker_export_meta = streaming_logger.speaker_export_metadata()
        self._separation_waveform = None

    def process_file(self, wav_path: str) -> str:
        """处理单个音频文件并返回生成的 RTTM 路径。"""

        self.reset()

        # 读原始音频。
        waveform = load_audio(wav_path, obj_fs=self.config.sample_rate)

        # 约定每个输入音频对应一个 streaming RTTM。
        if self.config.output_dir_for_streaming is None:
            raise ValueError(
                "output_dir_for_streaming must be set before processing audio"
            )
        streaming_log_path = str(
            Path(self.config.output_dir_for_streaming)
            / f"{Path(wav_path).stem}.streaming.rttm"
        )

        self.process_waveform(
            waveform,
            self.config.sample_rate,
            streaming_log_path=streaming_log_path,
            uri=Path(wav_path).stem,
        )
        # 导出所有说话人的完整音频
        if self.config.enable_speech_separation:
            audio_export_root = (
                Path(self.config.output_dir_for_streaming) / Path(wav_path).stem
            )
            self.speaker_buffers.export_grouped(
                uri=Path(wav_path).stem,
                export_root=audio_export_root,
                speaker_meta=self._speaker_export_meta,
                export_uncertain=self.config.export_uncertain_speaker_audio,
            )
        return streaming_log_path
