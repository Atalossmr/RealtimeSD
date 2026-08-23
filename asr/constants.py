"""ASR 转写模块的相关常量。"""

from __future__ import annotations

from pathlib import Path


# 仓库根目录（asr 包位于根目录下一层），与 diarization.constants 同一约定。
BASE_DIR = Path(__file__).resolve().parent.parent

# Fun-ASR-Nano 等 ModelScope 模型的下载缓存目录。
MODELSCOPE_CACHE_DIR = BASE_DIR / "pretrained" / "modelscope"
