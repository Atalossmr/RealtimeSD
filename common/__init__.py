"""asr / diarization 共用的公共实现（配置合并、日志、ModelScope 缓存解析）。

各子模块保持独立的公共 API（如 `asr.config.merge_args_with_config`），
跨模块重复的实现统一收敛到本包，原位置保留薄封装或 re-export。
"""
