"""把 segmentation 结果转成可用于全局分配的 observation。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .models import NativeERes2NetV2SegmentEmbedder
from .schema import PipelineConfig, SegmentObservation


class SegmentBuilder:
    """从 segmentation 帧结果中构造最小可用的 observation。

    这次重构的目标是把逻辑压到最简单：
    - 先在目标帧上找活跃的 local slot；
    - 再在整个上下文里为每个活跃 slot 找连续活跃区间；
    - 对每个 slot 只取“离目标帧最近、且长度合法”的那一段；
    - 然后直接提 embedding。

    这样做虽然比复杂排序策略更朴素，但更容易理解、排查和维护。
    """

    def __init__(
        self,
        config: PipelineConfig,
        embedder: NativeERes2NetV2SegmentEmbedder,
    ):
        self.config = config
        self.embedder = embedder

    def _extract_segment_waveform(
        self,
        chunk: torch.Tensor,
        chunk_start_time: float,
        seg_start: float,
        seg_end: float,
    ) -> torch.Tensor:
        """根据绝对时间，从当前上下文音频中裁出候选片段。

        这里之所以用“绝对时间 -> 相对时间 -> 样本下标”的方式，是为了统一处理：
        - 正常窗口；
        - 音频开头左侧补零；
        - 音频结尾右侧补零。
        """

        rel_start = max(0.0, seg_start - chunk_start_time)
        rel_end = min(self.config.chunk_duration, seg_end - chunk_start_time)
        start_sample = int(round(rel_start * self.config.sample_rate))
        end_sample = int(round(rel_end * self.config.sample_rate))
        end_sample = max(end_sample, start_sample + 1)
        return chunk[:, start_sample:end_sample]

    def _frame_step(self, absolute_centers: np.ndarray) -> float:
        """根据 segmentation 帧中心序列估计单帧时间步长。"""

        if absolute_centers.size <= 1:
            return float(self.config.step)
        return float(np.median(np.diff(absolute_centers)))

    def _connected_regions(self, active_mask: np.ndarray) -> list[tuple[int, int]]:
        """把连续为真的帧区间合并成若干个连续 region。"""

        regions: list[tuple[int, int]] = []
        start = None
        for idx, value in enumerate(active_mask.tolist()):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                regions.append((start, idx))
                start = None
        if start is not None:
            regions.append((start, len(active_mask)))
        return regions

    def _non_overlap_mask(self, segmentation: np.ndarray, local_idx: int) -> np.ndarray:
        """构造“只保留当前说话人单独活跃帧”的掩码。"""

        local_active = segmentation[:, local_idx] >= self.config.tau_active
        if segmentation.shape[1] <= 1:
            return local_active
        all_active = segmentation >= self.config.tau_active
        overlap_active = np.sum(all_active, axis=1) >= 2
        return np.logical_and(local_active, np.logical_not(overlap_active))

    def _region_to_times(
        self,
        absolute_centers: np.ndarray,
        region: tuple[int, int],
        frame_step: float,
    ) -> tuple[float, float]:
        """把帧索引区间换算成真实时间区间。"""

        start_idx, end_idx = region
        seg_start = max(0.0, float(absolute_centers[start_idx] - frame_step / 2))
        seg_end = float(absolute_centers[end_idx - 1] + frame_step / 2)
        return seg_start, seg_end

    def _clip_segment_around_center(
        self,
        seg_start: float,
        seg_end: float,
        reference_center: float,
    ) -> Optional[tuple[float, float]]:
        """对候选片段做最基础的长度和位置约束。

        规则尽量简单：
        - 先检查片段中心离目标帧是否太远；
        - 再过滤过短片段；
        - 若片段太长，就以片段中心为轴裁成最大允许长度。
        """

        raw_center = 0.5 * (seg_start + seg_end)
        if (
            abs(raw_center - reference_center)
            > self.config.max_segment_shift_from_center
        ):
            return None

        duration = seg_end - seg_start
        if duration < self.config.min_segment_duration_for_embedding:
            return None

        if duration > self.config.max_segment_duration_for_embedding:
            half = self.config.max_segment_duration_for_embedding / 2.0
            seg_start = raw_center - half
            seg_end = raw_center + half

        return seg_start, seg_end

    def _pick_best_region(
        self,
        segmentation: np.ndarray,
        local_idx: int,
        candidate_regions: list[tuple[int, int]],
        absolute_centers: np.ndarray,
        frame_step: float,
        reference_center: float,
        target_frame_idx: int,
    ) -> Optional[tuple[float, float, float, float]]:
        """为一个 local slot 选出最合适的单个片段。

        overlap 版本里，这里的排序规则改成：
        - 优先选择离目标时间更近的 region；
        - 距离接近时，再看平均激活是否更高；
        - 最后再看长度是否更长。

        这样做的原因是：
        - 在线场景里，离当前目标时刻更近的证据更不容易跨到别的说话人；
        - 纯度更高的片段通常更适合拿来提 embedding；
        - 长度仍然重要，但不应压过时序相关性和纯度。
        """

        best_item: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float, float]]
        ] = None

        for region in candidate_regions:
            seg_start, seg_end = self._region_to_times(
                absolute_centers, region, frame_step
            )
            clipped = self._clip_segment_around_center(
                seg_start, seg_end, reference_center
            )
            if clipped is None:
                continue
            seg_start, seg_end = clipped

            start_idx, end_idx = region
            local_scores = segmentation[start_idx:end_idx, local_idx]
            mean_activity = float(np.mean(local_scores)) if local_scores.size else 0.0
            speech_ratio = (
                float(np.mean(local_scores >= self.config.tau_active))
                if local_scores.size
                else 0.0
            )
            duration = float(seg_end - seg_start)
            region_center = 0.5 * (seg_start + seg_end)
            distance = abs(region_center - reference_center)

            rank = (-distance, mean_activity, duration)
            value = (seg_start, seg_end, mean_activity, speech_ratio)
            if best_item is None or rank > best_item[0]:
                best_item = (rank, value)

        if best_item is None:
            return None
        return best_item[1]

    def _window_frame_mask(
        self,
        absolute_centers: np.ndarray,
        target_time: float,
    ) -> tuple[np.ndarray, float]:
        """返回 target_time 附近的统计窗口及对应的 frame_step。

        overlap 版本里，目标 speaker 的选择不再只看一个 17ms 左右的单帧，
        而是看一个和在线输出步长 `step` 对齐的小时间窗。
        """

        frame_step = self._frame_step(absolute_centers)
        half_window = max(frame_step, 0.5 * self.config.step)
        window_start = target_time - half_window
        window_end = target_time + half_window
        frame_mask = np.logical_and(
            absolute_centers >= window_start,
            absolute_centers <= window_end,
        )
        if not np.any(frame_mask):
            target_frame_idx = int(np.argmin(np.abs(absolute_centers - target_time)))
            frame_mask[target_frame_idx] = True
        return frame_mask, frame_step

    def select_target_local_indices(
        self,
        segmentation: np.ndarray,
        absolute_centers: np.ndarray,
        target_time: float,
    ) -> list[int]:
        """围绕 target_time 在多帧范围内挑选真正要跟踪的 local speaker。

        背景是：
        - segmentation 的一帧只有十几毫秒；
        - 在线系统一次真正输出的决策粒度却可能是 0.5 秒甚至更大；
        - 因此只看 target_time 对应的单帧很容易因为瞬时波动而漏掉说话人。

        当前策略是：
        1. 用 `step` 定义一个围绕 target_time 的统计窗口；
        2. 统计每个 local slot 在这个窗口中的活跃总时长；
        3. 只保留活跃总时长超过阈值 `target_speaker_min_duration` 的说话人；
        4. 再按活跃时长从大到小排序。
        """

        if segmentation.size == 0 or absolute_centers.size == 0:
            return []

        num_frames, num_locals = segmentation.shape
        if num_frames == 0 or num_locals == 0:
            return []

        frame_mask, frame_step = self._window_frame_mask(absolute_centers, target_time)

        active_scores = segmentation[frame_mask]
        active_binary = active_scores >= self.config.tau_active
        active_durations = np.sum(active_binary, axis=0).astype(np.float32) * float(
            frame_step
        )

        selected: list[tuple[float, int]] = []
        for local_idx, duration in enumerate(active_durations.tolist()):
            if duration < self.config.target_overlap_min_duration:
                continue
            selected.append((float(duration), int(local_idx)))

        selected.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        if not selected:
            return []

        # 必须至少有一个说话人达到 primary 阈值，才认为这一帧真的有说话人
        if selected[0][0] < self.config.target_primary_min_duration:
            return []

        return [local_idx for _, local_idx in selected]

    def summarize_target_local_activity(
        self,
        segmentation: np.ndarray,
        absolute_centers: np.ndarray,
        target_time: float,
    ) -> list[dict[str, float]]:
        """统计 target_time 附近每个 local slot 的活跃情况。

        这个函数主要服务于 overlap 版本的 debug 日志：
        - 帮助观察哪些 local slot 真正持续说话；
        - 也方便判断第二说话人为什么会被保留或被过滤。
        """

        if segmentation.size == 0 or absolute_centers.size == 0:
            return []

        frame_mask, frame_step = self._window_frame_mask(absolute_centers, target_time)
        if not np.any(frame_mask):
            return []

        window_scores = segmentation[frame_mask]
        window_binary = window_scores >= self.config.tau_active
        active_durations = np.sum(window_binary, axis=0).astype(np.float32) * frame_step
        mean_scores = np.mean(window_scores, axis=0)
        max_scores = np.max(window_scores, axis=0)

        summary: list[dict[str, float]] = []
        for local_idx in range(window_scores.shape[1]):
            summary.append(
                {
                    "local": int(local_idx),
                    "active_duration": float(active_durations[local_idx]),
                    "mean_score": float(mean_scores[local_idx]),
                    "max_score": float(max_scores[local_idx]),
                }
            )

        summary.sort(
            key=lambda item: (item["active_duration"], item["mean_score"]),
            reverse=True,
        )
        return summary

    def _select_region_for_local(
        self,
        segmentation: np.ndarray,
        local_idx: int,
        absolute_centers: np.ndarray,
        frame_step: float,
        reference_center: float,
        target_frame_idx: int,
    ) -> Optional[tuple[float, float, float, float, bool, str]]:
        """实现“先去重叠，再回退到原始掩码”的 observation 选择逻辑。"""

        primary_mask = self._non_overlap_mask(segmentation, local_idx)
        if np.any(primary_mask):
            primary_regions = self._connected_regions(primary_mask)
            primary_region = self._pick_best_region(
                segmentation,
                local_idx,
                primary_regions,
                absolute_centers,
                frame_step,
                reference_center,
                target_frame_idx,
            )
            if primary_region is not None:
                return (*primary_region, True, "non_overlap")

        fallback_mask = segmentation[:, local_idx] >= self.config.tau_active
        if not np.any(fallback_mask):
            return None
        fallback_regions = self._connected_regions(fallback_mask)
        fallback_region = self._pick_best_region(
            segmentation,
            local_idx,
            fallback_regions,
            absolute_centers,
            frame_step,
            reference_center,
            target_frame_idx,
        )
        if fallback_region is None:
            return None
        return (*fallback_region, False, "overlap_fallback")

    def build_observations(
        self,
        *,
        window_id: int,
        chunk: torch.Tensor,
        chunk_start_time: float,
        segmentation: np.ndarray,
        absolute_centers: np.ndarray,
        target_local_indices: Optional[list[int]] = None,
        reference_center: Optional[float] = None,
    ) -> list[SegmentObservation]:
        """围绕目标帧活跃 speaker 构造 observation 列表。

        这一步是整条链路的“取证”阶段。
        当前实现的思路非常直接：
        - 每个活跃 local slot 最多生成一条 observation；
        - observation 对应一个合法活跃片段；
        - 这样后面的全局 speaker 分配器只需要面对少量、直观的证据。
        """

        if segmentation.size == 0:
            return []

        num_frames, num_locals = segmentation.shape
        if num_frames == 0:
            return []

        frame_step = (
            float(np.median(np.diff(absolute_centers)))
            if num_frames > 1
            else self.config.step
        )

        if reference_center is None:
            reference_center = (
                0.5 * (float(absolute_centers[0]) + float(absolute_centers[-1]))
                if absolute_centers.size
                else chunk_start_time + 0.5 * self.config.chunk_duration
            )

        target_frame_idx = int(np.argmin(np.abs(absolute_centers - reference_center)))
        local_indices = (
            sorted({int(local_idx) for local_idx in target_local_indices})
            if target_local_indices is not None
            else list(range(num_locals))
        )

        observations: list[SegmentObservation] = []
        waveforms: list[torch.Tensor] = []
        pending_meta: list[
            tuple[int, float, float, float, float, float, float, bool, str]
        ] = []

        for local_idx in local_indices:
            if local_idx < 0 or local_idx >= num_locals:
                continue

            # 第一步：只看该 local slot 自己的帧级分数。
            local_scores = segmentation[:, local_idx]

            # 第二步：先尝试只在“非重叠帧”里找 observation；
            # 如果失败，再放开到原始活跃掩码。
            best_region = self._select_region_for_local(
                segmentation,
                local_idx,
                absolute_centers,
                frame_step,
                reference_center,
                target_frame_idx,
            )
            if best_region is None:
                continue

            (
                seg_start,
                seg_end,
                mean_activity,
                speech_ratio,
                allow_centroid_update,
                selection_mode,
            ) = best_region
            waveform = self._extract_segment_waveform(
                chunk,
                chunk_start_time,
                seg_start,
                seg_end,
            )
            if waveform.shape[1] <= 0:
                continue

            waveforms.append(waveform)
            pending_meta.append(
                (
                    int(local_idx),
                    float(seg_start),
                    float(seg_end),
                    float(0.5 * (seg_start + seg_end)),
                    float(local_scores[target_frame_idx]),
                    float(mean_activity),
                    float(speech_ratio),
                    bool(allow_centroid_update),
                    str(selection_mode),
                )
            )

        if not waveforms:
            return []

        # 第四步：批量提取 speaker embedding，再把时间信息和分数信息拼回 observation。
        for start in range(0, len(waveforms), max(1, self.config.segment_batch_size)):
            batch_waveforms = waveforms[start : start + self.config.segment_batch_size]
            batch_embeddings = self.embedder.embed_segments(batch_waveforms)
            batch_meta = pending_meta[start : start + self.config.segment_batch_size]
            for embedding, meta in zip(batch_embeddings, batch_meta):
                (
                    local_idx,
                    seg_start,
                    seg_end,
                    seg_center,
                    score_at_target,
                    mean_activity,
                    speech_ratio,
                    allow_centroid_update,
                    selection_mode,
                ) = meta
                observations.append(
                    SegmentObservation(
                        window_id=window_id,
                        local_idx=local_idx,
                        start=seg_start,
                        end=seg_end,
                        center=seg_center,
                        embedding=embedding,
                        score_at_target=score_at_target,
                        mean_activity=mean_activity,
                        speech_ratio=speech_ratio,
                        duration=float(seg_end - seg_start),
                        allow_centroid_update=allow_centroid_update,
                        selection_mode=selection_mode,
                    )
                )

        return observations
