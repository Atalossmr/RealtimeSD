"""说话人嵌入（speaker embedding）模型的加载与分段推理。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .modelscope import resolve_modelscope_snapshot

from ..constants import BASE_DIR
from ..utils.device import resolve_device
from diarization.speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
from diarization.speakerlab.process.processor import FBank

if TYPE_CHECKING:
    from ..config import ChunkPipelineConfig

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
    """加载 ERes2NetV2 说话人嵌入模型并载入 checkpoint。

    当前只支持仓库内使用的 `eres2netv2`。优先以 weights_only=True 安全
    反序列化，失败时回退 weights_only=False（仅对可信来源安全）；checkpoint
    允许多余键（忽略并告警），但存在缺失键时报错而不是静默跑。

    Args:
        model_path: 本地 checkpoint 路径；为 None 时回退到 ModelScope 默认
            仓库下载。
        device: 模型加载与推理设备。
        model_type: 模型类型，目前仅支持 "eres2netv2"。
        feat_dim: 输入 FBank 特征维度。
        embedding_size: 输出 embedding 维度。
        m_channels: ERes2NetV2 的通道数配置。

    Returns:
        已切到 eval 模式并放到 device 上的嵌入模型。

    Raises:
        ValueError: model_type 不受支持。
        FileNotFoundError: 显式指定的本地 checkpoint 不存在。
        RuntimeError: checkpoint 与模型结构不匹配（存在缺失键）。
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
    """轻量校验 ModelScope model id 格式（`<namespace>/<name>`），避免引入重依赖。"""

    parts = [part for part in model_id.split("/") if part]
    return len(parts) == 2 and all(parts)


def _modelscope_spec_for_model_type(model_type: str) -> dict[str, str]:
    """返回当前 speaker encoder 对应的默认 ModelScope 仓库信息。

    Args:
        model_type: 模型类型。

    Returns:
        包含 model_id / revision / model_pt 的字典。

    Raises:
        ValueError: model_type 没有对应的 ModelScope 默认仓库。
    """

    spec = MODELSCOPE_EMBEDDING_MODELS.get(model_type.lower())
    if spec is None:
        raise ValueError(
            f"Unsupported model_type for ModelScope fallback: {model_type}"
        )
    return spec


def resolve_embedding_model_path(model_path: Optional[str], model_type: str) -> str:
    """解析 speaker encoder checkpoint 路径。

    优先使用用户显式提供的本地路径；未提供时回退到 ModelScope 默认仓库，
    并把下载结果缓存在仓库内 `pretrained/modelscope` 下。

    Args:
        model_path: 本地 checkpoint 路径，可为 None。
        model_type: 模型类型，用于确定 ModelScope 默认仓库。

    Returns:
        解析后的 checkpoint 文件路径。

    Raises:
        FileNotFoundError: 显式指定的本地 checkpoint 不存在。
        ValueError: model_type 不受支持或默认 ModelScope model id 非法。
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
    """原生 ERes2NetV2 分段 embedding 提取器。

    Attributes:
        model: 已加载的说话人嵌入模型。
        feature_extractor: FBank 特征提取器。
        sample_rate: 输入音频采样率。
        normalize_embeddings: 是否对输出 embedding 做 L2 归一化。
        device: 模型所在设备。
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        feature_extractor: FBank,
        sample_rate: int,
        normalize_embeddings: bool = True,
    ):
        """初始化分段 embedding 提取器。

        Args:
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
        """为单段语音提 embedding。

        Args:
            waveform: 单通道语音波形张量。

        Returns:
            shape 为 (embedding_size,) 的 float32 embedding。
        """

        with torch.inference_mode():
            feats = self.feature_extractor(waveform.cpu()).unsqueeze(0).to(self.device)
            embedding = self.model(feats)
            if self.normalize_embeddings:
                embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def embed_segments(self, waveforms: list[torch.Tensor]) -> list[np.ndarray]:
        """批量为多个候选段提 embedding。

        已知精度风险：短段 FBank 被零填充到批内最长帧后直接过模型，而
        ERes2NetV2 的 TSTP 池化（speakerlab/models/eres2net/pooling_layers.py）
        无长度掩码，填充帧经 conv+BN 后会污染 mean/std 统计。因此同一段
        语音走批量路径与单段路径（embed_segment）得到的 embedding 不完全
        一致，短段偏移最严重。
        未来修复方向：按长度分桶组批（bucket batching）减小填充量，或给
        TSTP 池化加长度掩码 / 长度加权统计。

        Args:
            waveforms: 候选段波形张量列表。

        Returns:
            与输入一一对应的 float32 embedding 列表；输入为空时返回空列表。
        """

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


class RegionEmbedder:
    """整段波形 + 绝对秒区间 → 各区间组的 embedding。

    在 NativeERes2NetV2SegmentEmbedder 之上补齐三件事：按绝对秒从整段
    波形裁剪并拼接区间、按 batch_size 分批推理、退化 embedding（NaN/inf
    或零范数）守卫。diarization 的实时前端与 oracle 前端
    （diarization/frontend/oracle.py）共用这一层。
    """

    def __init__(
        self,
        embedder: NativeERes2NetV2SegmentEmbedder,
        sample_rate: int,
        batch_size: int = 4,
    ):
        """初始化区域 embedding 提取器。

        Args:
            embedder: 已构建的分段 embedding 提取器。
            sample_rate: 波形采样率（区间秒数按它换算采样点下标）。
            batch_size: 分批推理的批大小。
        """

        self.embedder = embedder
        self.sample_rate = sample_rate
        self.batch_size = max(1, int(batch_size))

    def slice_regions(
        self,
        waveform: torch.Tensor,
        regions: list[tuple[float, float]],
    ) -> Optional[torch.Tensor]:
        """按绝对时间从整段波形裁出各 region 并拼接。

        Args:
            waveform: 整段波形，形状 (channels, samples)。
            regions: 绝对秒 (start, end) 区间列表，按时间升序。

        Returns:
            各 region 切片沿时间维首尾拼接的波形；全部为空（含越界截断后
            为空）时返回 None。
        """

        total_samples = waveform.shape[1]
        pieces: list[torch.Tensor] = []
        for start, end in regions:
            # region 为绝对秒数，换算成采样点下标并钳制到波形范围内；
            # 越界（区间由帧网格推出，可能略超出实际音频）时截断。
            start_sample = max(
                0, min(int(round(start * self.sample_rate)), total_samples)
            )
            end_sample = max(
                start_sample, min(int(round(end * self.sample_rate)), total_samples)
            )
            if end_sample > start_sample:
                pieces.append(waveform[:, start_sample:end_sample])
        if not pieces:
            return None
        # 同一区间组的多个段直接首尾拼接，作为一次 embedding 的输入。
        return torch.cat(pieces, dim=1)

    def embed_waveforms(
        self, waveforms: list[torch.Tensor]
    ) -> list[Optional[np.ndarray]]:
        """分批为各段波形提 embedding，含退化 embedding 守卫。

        NaN/inf 的 embedding 会让下游 Hungarian 指派直接抛错，零范数会造出
        永远无法匹配/合并的幽灵质心；来源通常是分割误检出的数字静音段。
        此处统一记为 None，交由调用方按"无 embedding"处理。

        Args:
            waveforms: 待提 embedding 的波形段列表。

        Returns:
            与输入一一对应的 embedding 列表；退化项为 None。
        """

        embeddings: list[Optional[np.ndarray]] = []
        for start in range(0, len(waveforms), self.batch_size):
            batch = self.embedder.embed_segments(waveforms[start : start + self.batch_size])
            for embedding in batch:
                if embedding is not None and (
                    not np.isfinite(embedding).all()
                    or float(np.linalg.norm(embedding)) <= 1e-12
                ):
                    logger.warning("degenerate embedding skipped")
                    embedding = None
                embeddings.append(embedding)
        return embeddings

    def embed_regions(
        self,
        waveform: torch.Tensor,
        regions_per_group: list[list[tuple[float, float]]],
    ) -> list[Optional[np.ndarray]]:
        """整段波形上按区间组提 embedding（裁剪拼接 + 分批 + 守卫）。

        Args:
            waveform: 整段波形，形状 (channels, samples)。
            regions_per_group: 每组一份绝对秒 (start, end) 区间列表，一组
                产出一条 embedding。

        Returns:
            与 regions_per_group 一一对应的 embedding 列表；裁不出波形或
            退化的组为 None。
        """

        segments: list[torch.Tensor] = []
        index_map: list[int] = []
        for group_idx, regions in enumerate(regions_per_group):
            segment = self.slice_regions(waveform, regions)
            if segment is None or segment.shape[1] <= 0:
                continue
            segments.append(segment)
            index_map.append(group_idx)

        embeddings: list[Optional[np.ndarray]] = [None] * len(regions_per_group)
        for group_idx, embedding in zip(index_map, self.embed_waveforms(segments)):
            embeddings[group_idx] = embedding
        return embeddings


def build_region_embedder(
    config: "ChunkPipelineConfig",
    model_path: Optional[str] = None,
) -> RegionEmbedder:
    """按管线配置构建 RegionEmbedder。

    封装模型加载、FBank 与两层 embedder 的组装，供 diarization extract
    与 pipeline 的 oracle 模式共用。

    Args:
        config: 管线配置（取 device/model_type/feat_dim/embedding_size/
            m_channels/sample_rate/normalize_embeddings/segment_batch_size）。
        model_path: 显式 checkpoint 路径覆盖；为 None 时按 config.model_type
            解析默认路径。

    Returns:
        构建好的 RegionEmbedder。
    """

    device = resolve_device(config.device)
    embedding_model = load_embedding_model(
        model_path=model_path,
        device=device,
        model_type=config.model_type,
        feat_dim=config.feat_dim,
        embedding_size=config.embedding_size,
        m_channels=config.m_channels,
    )
    feature_extractor = FBank(
        n_mels=config.feat_dim,
        sample_rate=config.sample_rate,
        mean_nor=True,
    )
    embedder = NativeERes2NetV2SegmentEmbedder(
        embedding_model=embedding_model,
        feature_extractor=feature_extractor,
        sample_rate=config.sample_rate,
        normalize_embeddings=config.normalize_embeddings,
    )
    return RegionEmbedder(
        embedder=embedder,
        sample_rate=config.sample_rate,
        batch_size=config.segment_batch_size,
    )
