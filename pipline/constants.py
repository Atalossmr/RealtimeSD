"""在线说话人分离模块的公共路径常量。"""

from __future__ import annotations

import sys
from pathlib import Path


# 这里统一把仓库根目录算出来，避免每个模块都重复拼路径。
BASE_DIR = Path(__file__).resolve().parent.parent

# 兼容原脚本的导入方式：
# - 仓库根目录加入 `sys.path` 后，可以直接导入 `speakerlab`；
# - `speakerlab` 目录本身也保留在 `sys.path` 中，降低旧代码迁移风险。
base_dir_str = str(BASE_DIR)
speakerlab_dir_str = str(BASE_DIR / "speakerlab")
if base_dir_str not in sys.path:
    sys.path.insert(0, base_dir_str)
if speakerlab_dir_str not in sys.path:
    sys.path.insert(0, speakerlab_dir_str)
