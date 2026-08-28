"""ASR 转写模块的相关常量。"""

from __future__ import annotations

from common.constants import BASE_DIR  # noqa: F401  (re-export，统一路径口径)

# Fun-ASR-Nano 等 ModelScope 模型的下载缓存目录。
MODELSCOPE_CACHE_DIR = BASE_DIR / "pretrained" / "modelscope"
