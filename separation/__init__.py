"""离线 RTTM 驱动的分离切分工具：整段音频 + RTTM -> 各 speaker 的纯净音频段。

入口为 `python -m separation.app --wav <音频> --rttm <RTTM> --output_dir <目录>`；
核心逻辑在 offline.py（separate_by_rttm），RTTM 解析与帧级活跃在 rttm.py。

包级 re-export 走惰性加载（PEP 562）：`import separation.rttm` 只取标准库
实现的 RTTM 解析，不连带 offline/sink 的 torch 重依赖，tools/ 下的轻量
分析脚本因此可以直接 `from separation.rttm import load_rttm`。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .offline import separate_by_rttm
    from .rttm import load_rttm
    from .sink import SegmentCallback, WavSegmentSink

__all__ = [
    "separate_by_rttm",
    "load_rttm",
    "SegmentCallback",
    "WavSegmentSink",
]


def __getattr__(name: str):
    """按需加载包级导出（见模块 docstring）。"""
    if name == "separate_by_rttm":
        from .offline import separate_by_rttm

        return separate_by_rttm
    if name == "load_rttm":
        from .rttm import load_rttm

        return load_rttm
    if name in ("SegmentCallback", "WavSegmentSink"):
        from . import sink

        return getattr(sink, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
