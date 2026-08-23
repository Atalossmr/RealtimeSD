"""模型加载与推理相关模块。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from common.modelscope import resolve_modelscope_snapshot

from ...constants import BASE_DIR
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
    try:
        # 优先安全反序列化；旧 checkpoint 含非张量对象时才回退
        # weights_only=False（可执行任意代码，仅对可信来源安全）。
        checkpoint = torch.load(
            resolved_model_path, map_location=device, weights_only=True
        )
    except Exception:
        logger.warning(
            "[extract] weights_only=True 加载失败，回退 weights_only=False"
            "（请确认 checkpoint 来源可信）: %s",
            resolved_model_path,
        )
        checkpoint = torch.load(
            resolved_model_path, map_location=device, weights_only=False
        )

    if isinstance(checkpoint, dict) and "embedding_model_state_dict" in checkpoint:
        state_dict = checkpoint["embedding_model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # 容忍 checkpoint 中的多余键，但不容忍缺失键：缺失意味着对应层停在
        # 随机初始化权重上，embedding 完全不可用，必须报错而不是静默跑。
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            raise RuntimeError(
                f"checkpoint 与模型结构不匹配，{len(result.missing_keys)} 个参数缺失"
                f"（如前 5 个: {result.missing_keys[:5]}）: {resolved_model_path}"
            )
        logger.warning(
            "[extract] checkpoint 含 %d 个未使用的多余键（已忽略）: %s",
            len(result.unexpected_keys),
            resolved_model_path,
        )

    model.eval()
    model.to(device)
    return model


def _is_valid_modelscope_model_id(model_id: str) -> bool:
    """做一个轻量的 ModelScope model id 校验，避免引入 pipelines 侧重依赖。"""

    parts = [part for part in model_id.split("/") if part]
    return len(parts) == 2 and all(parts)


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
    model_pt = spec["model_pt"]
    if not _is_valid_modelscope_model_id(model_id):
        raise ValueError(f"Invalid default ModelScope model id: {model_id}")

    snapshot_dir = resolve_modelscope_snapshot(
        model_id,
        Path(MODELSCOPE_DEFAULT_CACHE_DIR),
        lambda path: (path / model_pt).is_file(),
        revision=spec["revision"],
    )
    return str(snapshot_dir / model_pt)


class NativeERes2NetV2SegmentEmbedder:
    """原生 ERes2NetV2 embedding 提取器。"""

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        feature_extractor: FBank,
        sample_rate: int,
        normalize_embeddings: bool = True,
    ):
        """功能：初始化分段 embedding 提取器。

        参数：
            embedding_model: 已加载的说话人嵌入模型。
            feature_extractor: 特征提取器（FBank）。
            sample_rate: 输入音频采样率。
            normalize_embeddings: 是否对输出 embedding 做 L2 归一化。
        """
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
