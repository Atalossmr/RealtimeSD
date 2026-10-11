# 配置参数说明

本目录唯一配置文件 **`config.yaml`** — diarization 管线与 separation 离线
工具共用的全部调参项。

---

## config.yaml

参数按模块分组，与 `common/config/pipeline.py` 的 `ChunkPipelineConfig` 字段一一对应。
下表"默认"列为仓库自带 `config.yaml` 的生效值；YAML 缺省时落回dataclass中定义的默认值。

### 运行环境

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `device` | `auto` | 推理设备：`auto` / `cpu` / `cuda` / `cuda:0` |
| `sample_rate` | `16000` | 音频采样率（Hz），全部模型与输出 wav 统一用该采样率 |
| `hf_token` | 空 | Hugging Face token（也可用环境变量 `HF_TOKEN` 或 CLI `--hf_token`） |
| `hf_cache_dir` | `./pretrained/huggingface` | Hugging Face 缓存目录，相对路径按仓库根目录解析 |

### extract 阶段：模型

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `model_type` | `eres2netv2` | 说话人 embedding 模型类型（当前仅支持 eres2netv2） |
| `model_path` | 空 | 本地 embedding checkpoint 路径；为空时自动下载 ModelScope 默认模型到 `pretrained/modelscope`（CLI 参数，也可写在 YAML） |
| `embedding_size` | `192` | ERes2NetV2 输出 embedding 维度（需与 checkpoint 一致） |
| `feat_dim` | `80` | FBank 特征维度 |
| `m_channels` | `64` | ERes2NetV2 的通道数配置（需与 checkpoint 一致） |
| `normalize_embeddings` | `true` | 是否对 segment embedding 做 L2 归一化 |
| `segmentation_model` | `pyannote/segmentation-3.0` | pyannote 局部分割模型名，需 HF 授权（`HF_TOKEN`） |
| `segmentation_batch_size` | `1` | segmentation 推理的 batch size |

### extract 阶段：chunk 调度

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `chunk_duration` | `10.0` | 窗口时长（秒），segmentation-3.0 原生窗口即 10s |
| `hop_duration` | `5.0` | 推进步长（秒）。`hop < chunk` 为重叠滑窗：每窗只提交中段 hop 秒，两侧各留 `(chunk-hop)/2` 边界缓冲；`hop == chunk` 退化为非重叠 |

### extract 阶段：local track 构造与 embedding 提取

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `min_local_activity_duration` | `0.30` | local slot 在 chunk 内的最小累计活跃时长（秒），低于则不建 track、帧也不输出 |
| `min_segment_duration_for_embedding` | `0.30` | 允许提取 embedding 的最短拼接时长（秒）。**不要低于 0.105s@16k**（1680 samples），否则可能产生 NaN embedding 并触发 Hungarian 的 `matrix contains invalid numeric entries` |
| `max_segment_duration_for_embedding` | `4.0` | 单 track 用于提 embedding 的最大拼接时长（秒）。 |
| `segment_batch_size` | `4` | 批量提取 segment embedding 的 batch size |

### cluster 阶段：后端选择与通用

| 参数 | 默认 | 说明 |
|---|---|---|
| `clustering_backend` | `streaming` | `streaming`（增量聚类，纯流式）/ `ahc`（离线层次聚类：缓冲全部 embedding，音频结束一次聚类再重放输出）/ `fastdiar`（fastdiar 风格在线聚类：中位数质心 + 连通分量并簇）。新后端实现 `cluster/base.py` 接口并在 `build_clusterer` 注册即可 |

### cluster 阶段：streaming 后端

以下参数仅 `clustering_backend: streaming` 时生效。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `max_speakers` | `50` | 最多维护的全局 speaker 数 |
| `global_match_threshold` | `0.50` | observation 命中已有 speaker 的阈值（相似度 ≥ 此值判 `matched`）。 |
| `new_speaker_threshold` | `0.50` | 新建 speaker 的相似度阈值（相似度低于此值则新建说话人）。 |
| `min_segment_duration_for_new_speaker` | `1.00` | track 时长达到该值才允许新建 speaker（秒） |
| `min_segment_duration_for_centroid_update` | `1.50` | track 时长达到该值才允许更新 centroid（秒） |
| `merge_threshold` | `0.90` | 每次加入新片段后，若最相似的一对 centroid 相似度 ≥ 此值则合并 |

### cluster 阶段：ahc 后端

以下参数仅 `clustering_backend: ahc` 时生效。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `ahc_similarity_threshold` | `0.40` | AHC 余弦相似度阈值。 |
| `ahc_linkage` | `average` | AHC 连接方式：`average` / `complete` / `single` |

### cluster 阶段：fastdiar 后端

以下参数仅 `clustering_backend: fastdiar` 时生效（括号内为对应 fastdiar OnlineClustering 的参数名）。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `fastdiar_assign_threshold` | `0.45` | 指派阈值（clust_th）：observation 与最近质心相似度 ≥ 此值判 `matched`，否则尝试开新簇 |
| `fastdiar_confidence_threshold` | `0.65` | anchor 置信阈值（confidence）：`matched` 且相似度 ≥ 此值才作为 anchor 追加并更新质心（中位数） |
| `fastdiar_merge_threshold` | `0.80` | 连通分量并簇阈值（merge_th）：新建 speaker 后，质心相似度 > 此值的连通分量整组并簇；过低会过度并簇（false glue）抬高 SER |

### cluster 阶段：后处理（小样本簇强制合并）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `post_merge_min_count` | `8` | 小样本簇的样本计数阈值，低于则作为被吸收候选；`0` = 关闭后处理。该值同时是 streaming 前端 uncertain |
| `post_merge_min_similarity` | `0.0` | 强制合并的相似度下限 |

### oracle 前端（--oracle 模式）

以下参数仅 oracle 模式（`run.py / test_der.py --oracle --ref <参考 RTTM>`，
ref RTTM 驱动的 oracle 前端替代 segmentation 前端）使用。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `oracle_frame_step` | `0.05` | oracle 帧网格步长（秒）：ref RTTM 解析后按该步长生成全时间线逐帧活跃记录；同时是 oracle seg_scores 的帧网格基准 |

### RTTM 输出

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `min_segment_duration` | `0.30` | RTTM 写出最短片段时长（秒），闭合时更短的 turn 丢弃 |
| `segment_merge_gap` | `0.25` | 同 speaker 相邻片段自动拼接允许的最大间隔（秒，跨 chunk 生效）；同时是沉默闭合判据：open turn 终点距 commit_end 超过该值即闭合写出 |
| `output_dir` | 空 | 运行时输出目录（由 CLI `--output_dir` 或 `pipeline.process` 填充），一般不写进 YAML |
| `show_rttm` | `false` | 同步把新增 RTTM 行打印到控制台 |

### 调试

| 参数 | 默认 | 说明 |
|---|---|---|
| `debug` | `false` | 输出 chunk 级调试信息（`window_summary` / `new_speakers` / `updated_speakers` 等） |
