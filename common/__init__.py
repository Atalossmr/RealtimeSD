"""asr / diarization / separation 共用的公共实现。

不做包级 re-export（避免隐式依赖），按子包取用：

- `common.config/`：配置体系（base 通用 YAML/CLI 合并机制；pipeline 管线配置）
- `common.contracts/`：跨包数据契约（schema：ChunkArtifacts 等；frames：commit 区帧裁剪）
- `common.utils/`：通用工具（log / run_helpers）
- `common.models/`：模型封装（embedding / segmentation / tiger，hf_resolver /
  modelscope 为模型缓存解析）
- `common.constants`：路径常量（BASE_DIR）

边界：需要知道多个业务模块（diarization/separation/asr）协作语义的代码
不归这里，见顶层 `pipeline/` 组合层。
"""
