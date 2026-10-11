"""按 ref RTTM 驱动的音频预处理工具：整段音频 + ref RTTM -> 各 speaker 的纯净音频段。

入口为 `python -m tools.separation.app --wav <音频> --rttm <ref RTTM> --output_dir <目录>`；
核心逻辑在 offline.py（separate_by_rttm）。RTTM 解析在 common/utils/rttm.py
（标准库实现，tools/ 下的轻量分析脚本可直接 import，不触发 torch 重依赖）。

包级 re-export 走惰性加载（PEP 562），避免 import 包名即拉入 torch。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .offline import separate_by_rttm
    from .sink import SegmentCallback, WavSegmentSink

__all__ = [
    "separate_by_rttm",
    "SegmentCallback",
    "WavSegmentSink",
]


def __getattr__(name: str):
    """按需加载包级导出（见模块 docstring）。"""
    if name == "separate_by_rttm":
        from .offline import separate_by_rttm

        return separate_by_rttm
    if name in ("SegmentCallback", "WavSegmentSink"):
        from . import sink

        return getattr(sink, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
