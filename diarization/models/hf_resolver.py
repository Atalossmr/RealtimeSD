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
    """确保 Hugging Face 模型已缓存到本地，并返回缓存目录。

    缓存目录下存在任一权重标记文件（pytorch_model.bin / model.safetensors /
    weights.ckpt）时直接复用，否则调用 snapshot_download 下载。

    Args:
        repo_id: Hugging Face 仓库 id。
        cache_root: 本地缓存根目录。
        token: Hugging Face 访问 token，可为 None。

    Returns:
        本地缓存目录路径。
    """

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
    """在缓存的 snapshot 目录中寻找实际的权重文件。

    按 pytorch_model.bin → model.safetensors → weights.ckpt 的顺序查找。

    Args:
        local_dir: 本地缓存目录。

    Returns:
        权重文件路径。

    Raises:
        FileNotFoundError: 目录下没有任何受支持的权重文件。
    """

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
