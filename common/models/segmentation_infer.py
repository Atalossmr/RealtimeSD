"""pyannote segmentation 模型的流式推理封装。"""

from __future__ import annotations

from typing import Optional, Protocol, cast

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pyannote.audio.utils.powerset import Powerset

from .hf_resolver import resolve_hf_checkpoint_file, resolve_hf_snapshot_path


class _PredictionWithData(Protocol):
    """描述 pyannote 这类带 `.data` 属性的预测包装对象。"""

    data: object


class PyannoteStreamingSegmentation:
    """对固定长度上下文运行 pyannote segmentation-3.0 推理。

    以 step == duration 做不重叠滑窗推理，skip_conversion=True 保留原始
    分数矩阵（不做 Annotation 转换）；powerset 模型输出会先转回 multilabel
    分数再返回。
    """

    def __init__(
        self,
        model_name: str,
        duration: float,
        batch_size: int,
        device: torch.device,
        cache_dir: str,
        use_auth_token: Optional[str] = None,
    ):
        """加载模型并构建滑窗推理器。

        Args:
            model_name: Hugging Face 仓库 id。
            duration: 滑窗上下文长度（秒），同时作为步长。
            batch_size: 推理批大小。
            device: 推理设备。
            cache_dir: Hugging Face 缓存根目录。
            use_auth_token: Hugging Face 访问 token，可为 None。

        Raises:
            RuntimeError: 模型加载失败。
            ValueError: 模型启用 powerset 但缺少 classes 或 powerset_max_classes。
        """
        model_dir = resolve_hf_snapshot_path(model_name, cache_dir, use_auth_token)
        model_path = resolve_hf_checkpoint_file(model_dir)
        model = Model.from_pretrained(
            model_path, use_auth_token=use_auth_token, strict=False
        )
        if model is None:
            raise RuntimeError(f"Failed to load pyannote model from {model_path}")
        specification = next(iter(model.specifications))
        self.powerset = None
        if getattr(specification, "powerset", False):
            classes = specification.classes
            max_classes = specification.powerset_max_classes
            if classes is None or max_classes is None:
                raise ValueError(
                    "Model specification enables powerset but misses classes or powerset_max_classes"
                )
            self.powerset = Powerset(len(classes), int(max_classes))
        self.inference = Inference(
            model,
            duration=duration,
            step=duration,
            batch_size=batch_size,
            skip_aggregation=False,
            skip_conversion=True,
            device=device,
        )
        self.duration = duration

    def _raw_prediction_to_scores(self, prediction: object) -> np.ndarray:
        """把推理输出规整为 (帧数, 类别数) 的 float32 分数矩阵。

        Args:
            prediction: Inference 的原始输出（可能是带 `.data` 的包装对象）。

        Returns:
            shape 为 (帧数, 类别数) 的 float32 数组；powerset 模型输出已转
            为 multilabel。

        Raises:
            ValueError: 输出无法规整为 2D。
        """
        if hasattr(prediction, "data"):
            scores = np.asarray(cast(_PredictionWithData, prediction).data)
        else:
            scores = np.asarray(prediction)

        if scores.ndim == 3 and scores.shape[0] == 1:
            scores = scores[0]
        if scores.ndim != 2:
            raise ValueError(
                f"Expected 2D segmentation output, got shape {scores.shape}"
            )

        if self.powerset is None:
            return scores.astype(np.float32, copy=False)

        raw_tensor = torch.from_numpy(scores).unsqueeze(0)
        soft_scores = self.powerset.to_multilabel(raw_tensor, soft=False)[0]
        return soft_scores.cpu().numpy().astype(np.float32, copy=False)

    def _prediction_to_centers(
        self, prediction: object, frame_count: int
    ) -> np.ndarray:
        """根据帧数生成各帧中心相对窗口起点的时间（秒）。

        Args:
            prediction: 推理输出（当前未使用，保留以匹配调用签名）。
            frame_count: 分数矩阵的帧数。

        Returns:
            shape 为 (frame_count,) 的 float32 帧中心时间数组；frame_count
            为 0 时返回空数组。
        """
        if frame_count == 0:
            return np.zeros((0,), dtype=np.float32)

        frame_step = self.duration / frame_count
        return np.linspace(
            frame_step / 2,
            self.duration - frame_step / 2,
            frame_count,
            dtype=np.float32,
        )

    def __call__(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """对一段波形运行 segmentation 推理。

        Args:
            waveform: 输入波形张量。
            sample_rate: 波形采样率。

        Returns:
            (scores, centers)：scores 为 (帧数, 类别数) 的 float32 分数矩阵，
            centers 为各帧中心相对波形起点的时间（秒）。
        """
        prediction = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        scores = self._raw_prediction_to_scores(prediction)
        centers = self._prediction_to_centers(prediction, scores.shape[0])
        return scores, centers


__all__ = ["PyannoteStreamingSegmentation"]
