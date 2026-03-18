"""实时说话人识别主流程模块。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from speakerlab.process.processor import FBank
from speakerlab.utils.fileio import load_audio

from .clustering import IncrementalCentroidClusterer
from .models import (
    NativeERes2NetV2SegmentEmbedder,
    PyannoteStreamingSegmentation,
    load_embedding_model,
)
from .schema import (
    PipelineConfig,
    SegmentObservation,
    StreamingFrameDecision,
    WindowDebugInfo,
)
from .segmentation import SegmentBuilder
from .streaming import StreamingRTTMWriter, quantize_decision_time
from .utils import resample_waveform_if_needed, resolve_device


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
    7. 按简单顺序提交；
    8. 写 streaming RTTM。
    """

    def __init__(self, config: PipelineConfig, embedding_model_path: Optional[str]):
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

    def _reset_clusterer(self) -> None:
        """为新音频重置实时聚类状态。"""

        self.clusterer = IncrementalCentroidClusterer(
            delta_new=self.config.delta_new,
            max_speakers=self.config.max_speakers,
            global_match_threshold=self.config.global_match_threshold,
            merge_threshold=self.config.merge_threshold,
            sma_window=self.config.sma_window,
            update_segment_overlap_threshold=self.config.update_segment_overlap_threshold,
            weak_update_similarity_margin=self.config.weak_update_similarity_margin,
            weak_update_weight_multiplier=self.config.weak_update_weight_multiplier,
        )

    def reset(self) -> None:
        """对外暴露的状态清理接口。"""

        self._reset_clusterer()

    def _segmentation_dump_path(self, wav_path: str) -> Path:
        """返回保存 segmentation 概率矩阵的文件路径。"""

        # 运行入口会保证输出目录已提供，这里再做一次显式非空收窄：
        # 一方面能在调用顺序错误时尽早报出清晰异常，另一方面也能消除 Optional 路径的类型告警。
        if self.config.output_dir_for_streaming is None:
            raise ValueError(
                "output_dir_for_streaming must be set before processing audio"
            )
        return (
            Path(self.config.output_dir_for_streaming)
            / f"{Path(wav_path).stem}.segmentation_scores.jsonl"
        )

    def _append_segmentation_scores(
        self,
        *,
        dump_path: Path,
        window_id: int,
        target_time: float,
        chunk_start_time: float,
        seg_scores: np.ndarray,
        centers: np.ndarray,
        absolute_centers: np.ndarray,
    ) -> None:
        """把当前窗口的完整 segmentation 概率矩阵追加保存到文件。

        这里使用 JSONL 的原因是：
        - 每个窗口一行，天然适合流式追加；
        - 后续做离线分析时，容易逐行读取，不用一次性加载整个大文件。
        """

        payload = {
            "window_id": int(window_id),
            "target_time": float(target_time),
            "chunk_start_time": float(chunk_start_time),
            "chunk_end_time": float(chunk_start_time + self.config.chunk_duration),
            "shape": [int(dim) for dim in seg_scores.shape],
            "centers": centers.tolist(),
            "absolute_centers": absolute_centers.tolist(),
            "scores": seg_scores.tolist(),
        }
        with dump_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False))
            file_obj.write("\n")

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

        chunk_samples = int(round(self.config.chunk_duration * self.config.sample_rate))
        start_time = target_time - self.config.context_left_duration
        end_time = target_time + self.config.context_right_duration

        total_samples = int(waveform.shape[1])
        start_sample = int(np.floor(start_time * self.config.sample_rate))
        end_sample = int(np.ceil(end_time * self.config.sample_rate))

        left_pad = max(0, -start_sample)
        right_pad = max(0, end_sample - total_samples)
        valid_start = max(0, start_sample)
        valid_end = min(total_samples, end_sample)
        chunk = waveform[:, valid_start:valid_end]

        if left_pad > 0 or right_pad > 0:
            chunk = F.pad(chunk, (left_pad, right_pad))
        if chunk.shape[1] < chunk_samples:
            chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[1]))
        elif chunk.shape[1] > chunk_samples:
            chunk = chunk[:, :chunk_samples]

        return chunk, start_time

    def _target_frame_index(
        self,
        absolute_centers: np.ndarray,
        target_time: float,
    ) -> Optional[int]:
        """找到最接近目标时刻的 segmentation 帧索引。"""

        if absolute_centers.size == 0:
            return None
        return int(np.argmin(np.abs(absolute_centers - target_time)))

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
        而是围绕 target_time 做一个和 `step` 对齐的小时间窗汇总。

        这样做有两个目的：
        - 避免第二说话人只因为某一帧瞬时分数稍低就被忽略；
        - 让 overlap 输出更依赖“持续活跃时长”，而不是 17ms 单帧波动。
        """

        if target_frame_idx is None or segmentation_scores.size == 0:
            return []

        local_activity_summary = self.segment_builder.summarize_target_local_activity(
            segmentation_scores,
            absolute_centers,
            target_time,
        )
        summary_by_local = {int(item["local"]): item for item in local_activity_summary}

        # 同一个 global speaker 可能对应多个 local slot。
        # overlap 版本里，我们把同一个 global speaker 在目标时间窗内的证据合并起来，
        # 并按“活跃总时长优先、平均分数次之、target 单帧分数再次之”来排序。
        aggregate_by_global: dict[int, dict[str, float]] = {}
        frame_scores = segmentation_scores[target_frame_idx]

        for local_idx, global_id in local_to_global.items():
            if local_idx >= len(frame_scores):
                continue
            local_summary = summary_by_local.get(int(local_idx))
            if local_summary is None:
                continue
            entry = aggregate_by_global.setdefault(
                int(global_id),
                {
                    "active_duration": 0.0,
                    "mean_score": 0.0,
                    "target_score": 0.0,
                    "num_locals": 0.0,
                },
            )
            prev_duration = entry["active_duration"]
            new_duration = float(local_summary["active_duration"])
            if prev_duration + new_duration > 0:
                entry["mean_score"] = (
                    entry["mean_score"] * prev_duration
                    + float(local_summary["mean_score"]) * new_duration
                ) / (prev_duration + new_duration)
            entry["active_duration"] += new_duration

            entry["target_score"] = max(
                float(entry["target_score"]),
                float(frame_scores[local_idx]),
            )
            entry["num_locals"] += 1.0

        scored_globals = [
            (
                float(values["active_duration"]),
                float(values["mean_score"]),
                float(values["target_score"]),
                int(global_id),
            )
            for global_id, values in aggregate_by_global.items()
            if values["active_duration"] >= self.config.target_overlap_min_duration
        ]

        scored_globals.sort(reverse=True)

        if not scored_globals:
            return []

        if scored_globals[0][0] < self.config.target_primary_min_duration:
            return []

        return [
            global_id
            for _, _, _, global_id in scored_globals[: self.config.max_frame_speakers]
        ]

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
        force_flush: bool,
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
                "emitted": int(emitted_count),
                "force_flush": bool(force_flush),
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

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        streaming_log_path: str,
        uri: Optional[str] = None,
        segmentation_dump_path: Optional[str] = None,
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

        total_duration = waveform.shape[1] / self.config.sample_rate
        total_samples = waveform.shape[1]
        step_samples = max(1, int(round(self.config.step * self.config.sample_rate)))

        # `emitted_buckets` 用于保证同一个目标时间桶只输出一次。
        emitted_buckets: set[float] = set()
        last_emitted_time = -1e9

        streaming_logger = StreamingRTTMWriter(
            streaming_log_path,
            uri or "unknown",
            self.config.min_segment_duration,
            self.config.streaming_flush_interval,
            self.config.streaming_merge_gap,
            self.config.delay_short_speaker_output,
            self.config.speaker_min_total_duration_to_emit,
            self.config.show_rttm,
        )
        segmentation_dump = (
            Path(segmentation_dump_path) if segmentation_dump_path else None
        )

        # 按 step 构造所有窗口结束点。
        window_ends = list(
            range(step_samples, total_samples + step_samples, step_samples)
        )
        if not window_ends or window_ends[-1] != total_samples:
            window_ends.append(total_samples)

        def process_window(window_end_sample: int, force_flush: bool) -> None:
            nonlocal last_emitted_time

            # `current_time` 表示当前实时推进位置。
            current_time = (
                min(window_end_sample, total_samples) / self.config.sample_rate
            )

            # `target_time` 才是当前真正要判定的时刻。
            # 这里保留“落在当前推进点前半个 step”的简单策略。
            target_time = min(
                total_duration,
                max(0.0, current_time - 0.5 * self.config.step),
            )

            # 第一步：围绕目标帧切出固定上下文。
            chunk, chunk_start_time = self._slice_window(waveform, target_time)

            # 第二步：对整个上下文跑 segmentation。
            seg_scores, centers = self.segmentation(chunk, self.config.sample_rate)
            if seg_scores.size == 0:
                return
            absolute_centers = chunk_start_time + centers

            # 如果用户要求保存完整 segmentation 概率矩阵，这里就把它追加到 JSONL 文件里。
            if segmentation_dump is not None:
                self._append_segmentation_scores(
                    dump_path=segmentation_dump,
                    window_id=self.clusterer.window_counter,
                    target_time=target_time,
                    chunk_start_time=chunk_start_time,
                    seg_scores=seg_scores,
                    centers=centers,
                    absolute_centers=absolute_centers,
                )

            # 第三步：不再只看 target_time 对应的单帧，
            # 而是在 target_time 附近若干帧内统计各 local slot 的活跃总时长。
            target_local_indices = self.segment_builder.select_target_local_indices(
                seg_scores,
                absolute_centers,
                target_time,
            )

            # 第四步：只围绕这些活跃时长足够长的 local slot 构造 observation。
            observations = self.segment_builder.build_observations(
                window_id=self.clusterer.window_counter,
                chunk=chunk,
                chunk_start_time=chunk_start_time,
                segmentation=seg_scores,
                absolute_centers=absolute_centers,
                target_local_indices=target_local_indices,
                reference_center=target_time,
            )

            # 第五步：把当前目标帧窗口送入全局 speaker 分配器。
            window = self.clusterer.start_window(
                target_time=target_time,
                target_local_indices=target_local_indices,
                chunk_start_time=chunk_start_time,
                segmentation=seg_scores,
                absolute_centers=absolute_centers,
                observations=observations,
            )
            self.clusterer.push_window(
                window,
                force_flush=force_flush,
                return_debug=self.config.debug,
            )

            # 第六步：直接把已经解析好的窗口按时间顺序提交。
            commit_limit = total_duration if force_flush else current_time
            if commit_limit <= last_emitted_time:
                return

            resolved_windows = self.clusterer.pop_committable_windows(
                commit_limit,
                force_flush=force_flush,
            )
            if not resolved_windows:
                return

            max_emitted = last_emitted_time
            for resolved in resolved_windows:
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
                    self.config.step,
                    resolved.window.target_time,
                )
                if quantized_time in emitted_buckets:
                    continue
                emitted_buckets.add(quantized_time)

                # 第九步：把该目标帧 speaker 决策交给 streaming 输出模块。
                decisions = [
                    StreamingFrameDecision(
                        time=quantized_time,
                        speakers=speakers,
                    )
                ]
                max_emitted = max(max_emitted, quantized_time)
                streaming_logger.consume(
                    decisions,
                    self.config.step,
                    stable_until=max(0.0, commit_limit),
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
                        force_flush=force_flush,
                    )

            last_emitted_time = max_emitted

        # 先按正常实时模式处理所有窗口，再在结尾做一次强制 flush。
        for window_end_sample in window_ends:
            process_window(window_end_sample, force_flush=False)
        process_window(total_samples, force_flush=True)

        # 最后把所有还没写完的 turn 刷到 RTTM 文件中。
        streaming_logger.finalize()

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

        # 如果用户要求保存 segmentation 概率矩阵，则在每次处理该音频前先清空旧文件。
        segmentation_dump_path = None
        if self.config.save_segmentation_scores:
            segmentation_dump = self._segmentation_dump_path(wav_path)
            segmentation_dump.unlink(missing_ok=True)
            segmentation_dump_path = str(segmentation_dump)

        self.process_waveform(
            waveform,
            self.config.sample_rate,
            streaming_log_path=streaming_log_path,
            uri=Path(wav_path).stem,
            segmentation_dump_path=segmentation_dump_path,
        )
        return streaming_log_path
