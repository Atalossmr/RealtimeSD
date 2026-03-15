"""模型加载与推理相关模块。"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from pyannote.audio import Inference, Model
from pyannote.audio.utils.powerset import Powerset

from .constants import BASE_DIR
from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
from speakerlab.process.processor import FBank


logger = logging.getLogger(__name__)


MODELSCOPE_DEFAULT_CACHE_DIR = BASE_DIR / "pretrained" / "modelscope"
MODELSCOPE_EMBEDDING_MODELS = {
    "eres2netv2": {
        "model_id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "revision": "v1.0.1",
        "model_pt": "pretrained_eres2netv2.ckpt",
    }
}


class _PredictionWithData(Protocol):
    """描述 pyannote 这类带 `.data` 属性的预测包装对象。"""

    data: object


def load_embedding_model(
    model_path: Optional[str],
    device: torch.device,
    model_type: str = "eres2netv2",
    feat_dim: int = 80,
    embedding_size: int = 192,
    m_channels: int = 64,
) -> torch.nn.Module:
    """加载 ERes2NetV2 说话人嵌入模型。

    当前实现保持与原脚本一致，只支持仓库内使用的 `eres2netv2`。
    如果以后要扩展其他 speaker encoder，优先在这里做分支扩展。
    """

    model_type = model_type.lower()
    if model_type == "eres2netv2":
        model = ERes2NetV2(
            feat_dim=feat_dim, embedding_size=embedding_size, m_channels=m_channels
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. This pipeline currently supports only eres2netv2."
        )

    resolved_model_path = resolve_embedding_model_path(model_path, model_type)
    checkpoint = torch.load(
        resolved_model_path, map_location=device, weights_only=False
    )
    if isinstance(checkpoint, dict) and "embedding_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["embedding_model_state_dict"])
    elif isinstance(checkpoint, dict):
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model


def _is_valid_modelscope_model_id(model_id: str) -> bool:
    """做一个轻量的 ModelScope model id 校验，避免引入 pipelines 侧重依赖。"""

    parts = [part for part in model_id.split("/") if part]
    return len(parts) == 2 and all(parts)


def _load_modelscope_snapshot_download() -> Callable[..., str]:
    """延迟导入 ModelScope 下载接口，避免未使用时强依赖其运行环境。"""

    try:
        snapshot_module = importlib.import_module("modelscope.hub.snapshot_download")
    except ImportError as exc:
        raise RuntimeError(
            "No local ERes2NetV2 checkpoint was provided, but ModelScope fallback is unavailable. "
            "Please install `modelscope` and its dependencies, or pass `--model_path`."
        ) from exc

    snapshot_fn = getattr(snapshot_module, "snapshot_download", None)
    if not callable(snapshot_fn):
        raise RuntimeError(
            "ModelScope is installed but its snapshot download helper is unavailable. "
            "Please check the local `modelscope` installation, or pass `--model_path`."
        )

    return cast(Callable[..., str], snapshot_fn)


def _modelscope_spec_for_model_type(model_type: str) -> dict[str, str]:
    """返回当前 speaker encoder 对应的默认 ModelScope 仓库信息。"""

    spec = MODELSCOPE_EMBEDDING_MODELS.get(model_type.lower())
    if spec is None:
        raise ValueError(
            f"Unsupported model_type for ModelScope fallback: {model_type}"
        )
    return spec


def resolve_embedding_model_path(model_path: Optional[str], model_type: str) -> str:
    """解析 speaker encoder checkpoint 路径。

    优先使用用户显式提供的本地路径；如果缺失，则回退到 ModelScope 默认仓库，
    并把下载结果缓存在仓库内 `pretrained/modelscope` 下。
    """

    if model_path:
        resolved = os.path.expanduser(model_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Embedding model checkpoint not found: {resolved}")
        logger.info("Loading local speaker embedding model from %s", resolved)
        return resolved

    spec = _modelscope_spec_for_model_type(model_type)
    model_id = spec["model_id"]
    revision = spec["revision"]
    model_pt = spec["model_pt"]
    cache_root = Path(MODELSCOPE_DEFAULT_CACHE_DIR)
    expected_checkpoint = cache_root / model_id / model_pt

    if expected_checkpoint.is_file():
        logger.info(
            "Using cached ModelScope speaker embedding model from %s",
            expected_checkpoint,
        )
        return str(expected_checkpoint)

    snapshot_download_fn = _load_modelscope_snapshot_download()
    if not _is_valid_modelscope_model_id(model_id):
        raise ValueError(f"Invalid default ModelScope model id: {model_id}")

    os.makedirs(cache_root, exist_ok=True)
    logger.info(
        "No local ERes2NetV2 checkpoint provided; downloading %s@%s to %s",
        model_id,
        revision,
        cache_root,
    )
    downloaded_dir = Path(
        snapshot_download_fn(model_id, revision=revision, cache_dir=str(cache_root))
    )
    checkpoint_path = downloaded_dir / model_pt
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Downloaded ModelScope repo {model_id} but checkpoint file was not found: {checkpoint_path}"
        )

    logger.info("Downloaded speaker embedding checkpoint to %s", checkpoint_path)
    return str(checkpoint_path)


def _sanitize_repo_id(repo_id: str) -> str:
    """把 Hugging Face repo id 转成适合本地目录名的形式。"""

    return repo_id.replace("/", "--")


def resolve_hf_snapshot_path(
    repo_id: str,
    cache_root: str,
    token: Optional[str] = None,
) -> str:
    """确保 pyannote 模型已经缓存到本地，并返回缓存目录。"""

    os.makedirs(cache_root, exist_ok=True)
    local_dir = os.path.join(cache_root, _sanitize_repo_id(repo_id))
    weight_markers = [
        os.path.join(local_dir, "pytorch_model.bin"),
        os.path.join(local_dir, "model.safetensors"),
        os.path.join(local_dir, "weights.ckpt"),
    ]

    if any(os.path.exists(path) for path in weight_markers):
        logger.info("Using cached Hugging Face model from %s", local_dir)
        return local_dir

    logger.info("Downloading Hugging Face model %s to %s", repo_id, local_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_dir=local_dir,
    )
    return local_dir


def resolve_hf_checkpoint_file(local_dir: str) -> str:
    """在 pyannote snapshot 中寻找实际的权重文件。"""

    candidates = [
        os.path.join(local_dir, "pytorch_model.bin"),
        os.path.join(local_dir, "model.safetensors"),
        os.path.join(local_dir, "weights.ckpt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No supported checkpoint file found in cached Hugging Face model directory: {local_dir}"
    )


class PyannoteStreamingSegmentation:
    """对固定长度上下文运行 segmentation-3.0。

    输出仍然是“局部 slot 的帧级活动分数”，不是最终全局 speaker id。
    这一点在维护时非常重要：如果把这里误理解成 diarization 终结果，
    后续聚类与映射逻辑就会看起来像重复工作。
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
            # 上游类型桩把这些字段标成了可空值；
            # 但 powerset 模型一旦声明开启 powerset，这两个字段按约定都必须可用。
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
        """将模型原生输出转换为标准的逐帧多说话人活跃度软分数矩阵。

        返回 shape: [frame_count, local_speaker_count] 的 2D numpy 数组。

        处理逻辑：
        1. 提取底层 numpy 数组（剥离 SlidingWindowFeature 等封装）；
        2. 去除 Batch 维度（假设 batch_size=1）；
        3. 对于使用了 Powerset 编码的模型（如 pyannote/segmentation-3.0，该编码原生支持 Overlap，
           将多说话人组合如 {A}, {B}, {A,B} 视为互斥类别进行输出），将其解码/展开 (to_multilabel)
           为各个局部说话人相互独立的连续软概率分数 (soft scores)。
        """
        if hasattr(prediction, "data"):
            # `hasattr` 只能帮助运行时判断，静态检查器仍会把 `prediction` 当成 `object`。
            # 这里用 Protocol + cast 明确告诉类型系统：它是一个带 `.data` 的包装对象。
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
        soft_scores = self.powerset.to_multilabel(raw_tensor, soft=True)[0]
        return soft_scores.cpu().numpy().astype(np.float32, copy=False)

    def _prediction_to_centers(
        self, prediction: object, frame_count: int
    ) -> np.ndarray:
        """根据输入上下文窗口总时长和输出的帧数，计算每一帧中心对应的相对时间（单位：秒）。

        例如，如果 duration=10s，输出 500 帧，则每帧跨度(frame_step)为 0.02s。
        第一帧中心在 0.01s，最后一帧在 9.99s。
        这个相对时间戳数组在后续流程中会加上 chunk_start_time，从而转换为全局绝对时间，
        以用于对其 RTTM 输出和目标判断时间窗。
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
        """运行分割模型并返回帧分数与帧中心时刻。"""

        prediction = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        scores = self._raw_prediction_to_scores(prediction)
        centers = self._prediction_to_centers(prediction, scores.shape[0])
        return scores, centers


class NativeERes2NetV2SegmentEmbedder:
    """原生 ERes2NetV2 embedding 提取器。"""

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        feature_extractor: FBank,
        sample_rate: int,
        normalize_embeddings: bool = True,
    ):
        self.model = embedding_model
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.normalize_embeddings = normalize_embeddings
        self.device = next(self.model.parameters()).device

    def embed_segment(self, waveform: torch.Tensor) -> np.ndarray:
        """为单段语音提 embedding。"""

        with torch.inference_mode():
            feats = self.feature_extractor(waveform.cpu()).unsqueeze(0).to(self.device)
            embedding = self.model(feats)
            if self.normalize_embeddings:
                embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def embed_segments(self, waveforms: list[torch.Tensor]) -> list[np.ndarray]:
        """批量为多个候选段提 embedding。"""

        if not waveforms:
            return []
        feat_list = [self.feature_extractor(waveform.cpu()) for waveform in waveforms]
        max_frames = max(int(feat.shape[0]) for feat in feat_list)
        feat_dim = int(feat_list[0].shape[1])
        batch = torch.zeros(
            len(feat_list), max_frames, feat_dim, dtype=feat_list[0].dtype
        )
        for idx, feat in enumerate(feat_list):
            batch[idx, : feat.shape[0]] = feat
        with torch.inference_mode():
            embeddings = self.model(batch.to(self.device))
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return [
            emb.detach().cpu().numpy().astype(np.float32, copy=False)
            for emb in embeddings
        ]
