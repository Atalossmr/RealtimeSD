"""组合层：把 diarization 装配成完整管线（音频 → RTTM）。

定位：common/ 是无业务语义的共享底座，diarization/、separation/ 是互不
依赖的业务包（separation 为独立离线工具，不在管线内）；本包是
唯一同时知道它们的层——

- `api`：一次完整运行的公开入口（`process(audio=...)` 统一支持单文件
  与批量），run.py 与 test_der.py 均为其薄 CLI 封装；
- `runner`：diarization 的组合执行（DiarizationRunner / run_diarization）；
- `eval`：管线产物的 DER 评估；
- `cli`：入口脚本（run.py / test_der.py）共用的 CLI 辅助。

归属判断标准：需要知道 ≥2 个模块协作语义的代码归这里；
纯通用机制或被业务包直接 import 的归 common/。

例外：`diarization/app.py`（独立 CLI）复用 `run_diarization`，构成
diarization → pipeline 的入口脚本级反向依赖——app 是 CLI 入口而非库代码，
diarization 的库代码（extract/cluster/utils）不依赖本包。
"""

from .api import (
    ProcessResult,
    RunDirs,
    process,
    resolve_run_dirs,
)
from .eval import run_der_evaluation
from .runner import DiarizationRunner, run_diarization

__all__ = [
    "DiarizationRunner",
    "ProcessResult",
    "RunDirs",
    "process",
    "resolve_run_dirs",
    "run_der_evaluation",
    "run_diarization",
]
