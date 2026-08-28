"""组合层：把 diarization / separation / asr 装配成完整管线。

定位：common/ 是无业务语义的共享底座，diarization/、separation/、asr/
是互不依赖的业务包；本包是唯一同时知道它们的层——

- `api`：一次完整运行的公开入口（`process(audio=...)` 统一支持单文件
  与批量，stage1-3），run.py 与 test_der.py 均为其薄 CLI 封装；
- `runner`：stage1+stage2 的组合执行（DiarizationRunner / run_diarization）；
- `follow`：stage3 ASR 跟随进程的契约校验与命令组装；
- `eval`：管线产物的 DER 评估。

归属判断标准：需要知道 ≥2 个模块协作语义的代码归这里；
纯通用机制或被业务包直接 import 的归 common/。
"""

from .api import (
    ProcessResult,
    RunDirs,
    process,
    resolve_run_dirs,
)
from .eval import run_der_evaluation
from .follow import build_asr_follow_cmd, validate_follow_requirements
from .runner import DiarizationRunner, run_diarization

__all__ = [
    "DiarizationRunner",
    "ProcessResult",
    "RunDirs",
    "build_asr_follow_cmd",
    "process",
    "resolve_run_dirs",
    "run_der_evaluation",
    "run_diarization",
    "validate_follow_requirements",
]
