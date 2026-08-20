"""实时说话人识别模块的相关常量。"""

from pathlib import Path


# 这里统一把仓库根目录算出来，避免每个模块都重复拼路径。
BASE_DIR = Path(__file__).resolve().parent.parent

# 入口约定：一律从仓库根目录以 `python3 -m ...` 方式运行（此时仓库根本身
# 就在 sys.path 中），speakerlab/、look2hear/ 等 vendored 包直接按绝对导入
# 使用；本模块只提供常量，不在 import 时修改 sys.path。
