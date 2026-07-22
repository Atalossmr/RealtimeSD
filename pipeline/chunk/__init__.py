"""chunk 级局部识别 + 增量聚类全局对齐管线。

与 `pipeline/` 下的滑窗实现并行存在（新旧并行，用于 DER 对照）。
本子包自包含：独立的配置、聚类器、track 构造与 RTTM 写出；
仅复用 `pipeline.models` 的推理封装与 `pipeline.utils` 的通用工具。
"""
