"""asr / diarization / separation 共用的公共实现。

各子模块保持独立的公共 API（如 `asr.config.merge_args_with_config`），
跨模块重复的实现统一收敛到本包，原位置保留薄封装或 re-export。

内容：配置合并（config）、管线配置（pipeline_config）、日志（log）、
chunk 契约（schema / frames）、模型封装（models/）、ModelScope 缓存解析
（modelscope）、编排辅助（run_helpers）、路径常量（constants）。
"""
