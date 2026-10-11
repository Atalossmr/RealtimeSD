"""diarization 与 tools 共用的公共实现。

不做包级 re-export（避免隐式依赖），按子包取用：

- `common.config/`：配置体系（base 通用 YAML/CLI 合并机制；pipeline 管线配置）
- `common.contracts/`：chunk 管线共享数据契约（schema：ChunkArtifacts 等；
  frames：commit 区帧裁剪；当前消费者为 diarization 与 pipeline）
- `common.utils/`：通用工具（log / paths / numeric / device / audio / rttm）
- `common.models/`：模型封装（embedding / segmentation，hf_resolver /
  modelscope 为模型缓存解析）
- `common.speakerlab/`：vendored 模型依赖（3D-Speaker，含 md-eval.pl）
- `common.constants`：路径常量（BASE_DIR）

边界：需要知道多个业务模块协作语义的代码不归这里，见顶层 `pipeline/`
组合层。
"""