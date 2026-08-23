"""HuggingFace 模型缓存与权重路径解析。"""

from __future__ import annotations

import logging
import os
from typing import Optional

from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)


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


__all__ = ["resolve_hf_snapshot_path", "resolve_hf_checkpoint_file"]
