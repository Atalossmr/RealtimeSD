"""ModelScope 模型缓存解析与下载（asr / diarization 共用）。"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable, Optional, cast

logger = logging.getLogger(__name__)


def _load_snapshot_download() -> Callable[..., str]:
    """延迟导入 ModelScope 下载接口，避免未使用时强依赖其运行环境。"""

    try:
        snapshot_module = importlib.import_module("modelscope.hub.snapshot_download")
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope download fallback is unavailable. "
            "Please install `modelscope` and its dependencies, or use a local model path."
        ) from exc

    snapshot_fn = getattr(snapshot_module, "snapshot_download", None)
    if not callable(snapshot_fn):
        raise RuntimeError(
            "ModelScope is installed but its snapshot download helper is unavailable. "
            "Please check the local `modelscope` installation."
        )

    return cast(Callable[..., str], snapshot_fn)


def resolve_modelscope_snapshot(
    model_id: str,
    cache_dir: Path,
    is_complete: Callable[[Path], bool],
    revision: Optional[str] = None,
) -> Path:
    """把 ModelScope 模型 id 解析为本地快照目录：完整缓存 > 下载。

    is_complete 由调用方按模型所需文件判定（防止半成品缓存被当作可用）；
    缓存目录存在但不完整时判 miss 重新下载，下载后同样校验，不完整则抛错。
    """

    cache_dir = Path(cache_dir)
    snapshot_dir = cache_dir / model_id
    if snapshot_dir.is_dir():
        if is_complete(snapshot_dir):
            logger.info("Using cached ModelScope model from %s", snapshot_dir)
            return snapshot_dir
        logger.warning(
            "cached model dir %s is incomplete, falling back to download",
            snapshot_dir,
        )

    snapshot_download = _load_snapshot_download()
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ModelScope model %s to %s", model_id, cache_dir)
    if revision is not None:
        downloaded_dir = Path(
            snapshot_download(model_id, revision=revision, cache_dir=str(cache_dir))
        )
    else:
        downloaded_dir = Path(snapshot_download(model_id, cache_dir=str(cache_dir)))
    if not is_complete(downloaded_dir):
        raise FileNotFoundError(
            f"Downloaded ModelScope repo {model_id} but required files are missing "
            f"under {downloaded_dir}"
        )

    logger.info("Downloaded ModelScope model to %s", downloaded_dir)
    return downloaded_dir


__all__ = ["resolve_modelscope_snapshot"]
