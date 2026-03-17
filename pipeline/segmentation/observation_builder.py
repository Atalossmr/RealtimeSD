"""把 segmentation 结果转成可用于全局分配的 observation。

本模块负责把 segmentation 的逐帧活跃度软分数矩阵转成后续 clustering 所需的 observation：

- 围绕目标时刻 `target_time` 统计一个小窗，挑选“真正活跃”的 local slot
- 在上下文窗口内为每个活跃 local slot 找一段连续活跃区间，裁剪到合适长度
- 从上下文音频中裁出该片段并提取 speaker embedding

注意：

- 本模块不处理全局 speaker 合并（merge）；merge 发生在 clustering/streaming 层。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..models import NativeERes2NetV2SegmentEmbedder
from .selector import (
    select_target_local_indices as select_target_local_indices_impl,
    summarize_target_local_activity as summarize_target_local_activity_impl,
    window_frame_mask as window_frame_mask_impl,
)
from ..schema import PipelineConfig, SegmentCandidate, SegmentObservation


class SegmentBuilder:
    """从 segmentation 帧结果中构造最小可用的 observation。

    主要流程为：
    - 先在目标帧上找活跃的 local slot；
    - 再在整个上下文里为每个活跃 slot 找连续活跃区间；
    - 对每个 slot 只取“离目标帧最近、且长度合法”的那一段；
    - 然后直接提 embedding。

    后续可替换为更有效地挑选逻辑。
    """

    def __init__(
        self,
        config: PipelineConfig,
        embedder: NativeERes2NetV2SegmentEmbedder,
    ):
        """功能：初始化 observation 构造器。

        参数：
            config: 管线配置。
            embedder: 段级 embedding 提取器。
        """
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
            return float(self.config.advance_step)
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

        local_active = segmentation[:, local_idx] > 0.0
        if segmentation.shape[1] <= 1:
            return local_active
        all_active = segmentation > 0.0
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

        这里的排序规则为：

        - 离目标时间更近；
        - 平均激活更高；
        - 长度是否更长。
        目前仅采用简单排序，后续可改为加权运算

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
                float(np.mean(local_scores > 0.0)) if local_scores.size else 0.0
            )
            duration = float(seg_end - seg_start)
            # 这里用“与片段边界的最小距离”而不是“片段中心距离”，
            # 可以让覆盖 target_time 的片段优先级更高，减少长片段中心偏移导致的误判。
            distance = min(
                abs(seg_start - reference_center), abs(seg_end - reference_center)
            )

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
        而是看一个可配置的小时间窗（`target_activity_window_duration`）。
        """

        return window_frame_mask_impl(
            absolute_centers=absolute_centers,
            target_time=float(target_time),
            activity_window_duration=float(self.config.target_activity_window_duration),
            frame_step_fn=self._frame_step,
        )

    def select_target_local_indices(
        self,
        segmentation: np.ndarray,
        absolute_centers: np.ndarray,
        target_time: float,
    ) -> list[int]:
        """围绕 target_time 在多帧范围内挑选真正要跟踪的 local speaker。

        背景是：
        - segmentation 的一帧只有十几毫秒；
        - 实时系统一次真正输出的决策粒度却可能是 0.5 秒甚至更大；
        - 因此只看 target_time 对应的单帧很容易因为瞬时波动而漏掉说话人。

        当前策略是：
        1. 用 `target_activity_window_duration` 定义一个围绕 target_time 的统计窗口（前后各半窗）；
        2. 统计每个 local slot 在该窗口内的累计活跃时长；
        3. 仅保留累计活跃时长不小于 `target_min_duration` 的 local slot。
        """

        return select_target_local_indices_impl(
            config=self.config,
            segmentation=segmentation,
            absolute_centers=absolute_centers,
            target_time=float(target_time),
            frame_step_fn=self._frame_step,
        )

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

        return summarize_target_local_activity_impl(
            config=self.config,
            segmentation=segmentation,
            absolute_centers=absolute_centers,
            target_time=float(target_time),
            frame_step_fn=self._frame_step,
        )

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

        fallback_mask = segmentation[:, local_idx] > 0.0
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

    def build_candidates(
        self,
        *,
        window_id: int,
        segmentation: np.ndarray,
        absolute_centers: np.ndarray,
        target_local_indices: Optional[list[int]] = None,
        reference_center: Optional[float] = None,
    ) -> list[SegmentCandidate]:
        """围绕目标帧活跃 speaker 构造候选片段列表。

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
            else self.config.advance_step
        )

        if reference_center is None:
            reference_center = (
                0.5 * (float(absolute_centers[0]) + float(absolute_centers[-1]))
                if absolute_centers.size
                else 0.5 * self.config.chunk_duration
            )

        target_frame_idx = int(np.argmin(np.abs(absolute_centers - reference_center)))
        local_indices = (
            sorted({int(local_idx) for local_idx in target_local_indices})
            if target_local_indices is not None
            else list(range(num_locals))
        )

        candidates: list[SegmentCandidate] = []

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
            candidates.append(
                SegmentCandidate(
                    window_id=window_id,
                    local_idx=int(local_idx),
                    start=float(seg_start),
                    end=float(seg_end),
                    center=float(0.5 * (seg_start + seg_end)),
                    score_at_target=float(local_scores[target_frame_idx]),
                    mean_activity=float(mean_activity),
                    speech_ratio=float(speech_ratio),
                    duration=float(seg_end - seg_start),
                    allow_centroid_update=bool(allow_centroid_update),
                    selection_mode=str(selection_mode),
                )
            )

        return candidates

    def embed_candidates(
        self,
        *,
        chunk: torch.Tensor,
        chunk_start_time: float,
        candidates: list[SegmentCandidate],
    ) -> list[SegmentObservation]:
        """功能：仅对候选片段批量提 embedding 并组装 observation。

        参数：
            chunk: 当前上下文音频。
            chunk_start_time: chunk 起始绝对时刻（秒）。
            candidates: 待提 embedding 的候选片段列表。

        返回：
            完整的 observation 列表。
        """
        observations: list[SegmentObservation] = []
        waveforms: list[torch.Tensor] = []
        pending_candidates: list[SegmentCandidate] = []

        for candidate in candidates:
            waveform = self._extract_segment_waveform(
                chunk,
                chunk_start_time,
                candidate.start,
                candidate.end,
            )
            if waveform.shape[1] <= 0:
                continue
            waveforms.append(waveform)
            pending_candidates.append(candidate)

        if not pending_candidates:
            return []

        # 第四步：批量提取 speaker embedding，再把时间信息和分数信息拼回 observation。
        for start in range(
            0,
            len(pending_candidates),
            max(1, self.config.segment_batch_size),
        ):
            batch_waveforms = waveforms[start : start + self.config.segment_batch_size]
            batch_embeddings = self.embedder.embed_segments(batch_waveforms)
            batch_candidates = pending_candidates[
                start : start + self.config.segment_batch_size
            ]
            for embedding, candidate in zip(batch_embeddings, batch_candidates):
                observations.append(
                    SegmentObservation(
                        window_id=candidate.window_id,
                        local_idx=candidate.local_idx,
                        start=candidate.start,
                        end=candidate.end,
                        center=candidate.center,
                        embedding=embedding,
                        score_at_target=candidate.score_at_target,
                        mean_activity=candidate.mean_activity,
                        speech_ratio=candidate.speech_ratio,
                        duration=candidate.duration,
                        allow_centroid_update=candidate.allow_centroid_update,
                        selection_mode=candidate.selection_mode,
                    )
                )

        return observations
