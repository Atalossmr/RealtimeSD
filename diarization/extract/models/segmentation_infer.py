"""segmentation 推理封装。"""

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
    """对固定长度上下文运行 segmentation-3.0。"""

    def __init__(
        self,
        model_name: str,
        duration: float,
        batch_size: int,
        device: torch.device,
        cache_dir: str,
        use_auth_token: Optional[str] = None,
    ):
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
        prediction = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        scores = self._raw_prediction_to_scores(prediction)
        centers = self._prediction_to_centers(prediction, scores.shape[0])
        return scores, centers


__all__ = ["PyannoteStreamingSegmentation"]
