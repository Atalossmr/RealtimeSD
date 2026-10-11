"""模型加载与推理封装：speaker embedding / segmentation。"""

from .embedding_infer import (
    NativeERes2NetV2SegmentEmbedder,
    RegionEmbedder,
    build_region_embedder,
    load_embedding_model,
)
from .hf_resolver import resolve_hf_checkpoint_file, resolve_hf_snapshot_path
from .modelscope import resolve_modelscope_snapshot
from .segmentation_infer import PyannoteStreamingSegmentation

__all__ = [
    "load_embedding_model",
    "PyannoteStreamingSegmentation",
    "NativeERes2NetV2SegmentEmbedder",
    "RegionEmbedder",
    "build_region_embedder",
    "resolve_hf_checkpoint_file",
    "resolve_hf_snapshot_path",
    "resolve_modelscope_snapshot",
]
