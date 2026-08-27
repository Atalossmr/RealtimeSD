# 配置参数说明

本目录两个配置文件：

- **`config.yaml`** — diarization 管线的全部调参项；
- **`asr.yaml`** — ASR 转写的全部调参项。

---

## 一、config.yaml（diarization 管线）

参数按模块分组，与 `common/pipeline_config.py` 的 `ChunkPipelineConfig` 字段一一对应。
下表"默认"列为仓库自带 `config.yaml` 的生效值；YAML 缺省时落回dataclass中定义的默认值。

### 运行环境

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `device` | `auto` | 推理设备：`auto` / `cpu` / `cuda` / `cuda:0` |
| `hf_cache_dir` | `./pretrained/huggingface` | Hugging Face 缓存目录，相对路径按仓库根目录解析 |

### extract 阶段：模型

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `model_type` | `eres2netv2` | 说话人 embedding 模型类型（当前仅支持 eres2netv2） |
| `model_path` | 空 | 本地 embedding checkpoint 路径；为空时自动下载 ModelScope 默认模型到 `pretrained/modelscope`（CLI 参数，也可写在 YAML） |
| `segmentation_model` | `pyannote/segmentation-3.0` | pyannote 局部分割模型名，需 HF 授权（`HF_TOKEN`） |

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
| `clustering_backend` | `streaming` | `streaming`（增量聚类，纯流式）/ `ahc`（离线层次聚类：缓冲全部 embedding，音频结束一次聚类再重放输出）。新后端实现 `cluster/base.py` 接口并在 `build_clusterer` 注册即可 |

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

### cluster 阶段：后处理（小样本簇强制合并）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `post_merge_min_count` | `8` | 小样本簇的样本计数阈值，低于则作为被吸收候选；`0` = 关闭后处理。该值同时是 streaming 前端 uncertain |
| `post_merge_min_similarity` | `0.0` | 强制合并的相似度下限 |

### 分段音频导出

以下参数仅 `separation_enabled: true` 且 `clustering_backend: streaming` 时生效。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `separation_enabled` | `true` | 是否启用语音分离及分段音频导出 |
| `separation_model` | `JusperLee/TIGER-speech` | TIGER 分离模型（HF，固定 2 路输出、16kHz），缓存到 `hf_cache_dir` |
| `separation_energy_ratio` | `0.05` | 能量门控：分离音轨在重叠帧区间的 RMS / 混合音频同区间 RMS，低于判为伪源（OSD 疑似误报）。仅一路过门控时该路匹配归属、落选 speaker 的重叠帧不输出；两路都不过门控则整窗回退原始音频 |
| `separation_min_match_similarity` | `0.10` | 2x2 匹配结果的最小余弦相似度，低于判分离质量不可靠，回退原始音频。 |

### RTTM 输出

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `min_segment_duration` | `0.30` | RTTM 写出最短片段时长（秒），闭合时更短的 turn 丢弃 |
| `segment_merge_gap` | `0.25` | 同 speaker 相邻片段自动拼接允许的最大间隔（秒，跨 chunk 生效）；同时是沉默闭合判据：open turn 终点距 commit_end 超过该值即闭合写出 |
| `show_rttm` | `false` | 同步把新增 RTTM 行打印到控制台 |

### 调试

| 参数 | 默认 | 说明 |
|---|---|---|
| `debug` | `false` | 输出 chunk 级调试信息（`window_summary` / `new_speakers` / `updated_speakers` 等） |

---

## 二、asr.yaml（ASR 转写，Fun-ASR-Nano）

ASR 是独立于 pipeline 的模块

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `sample_rate` | `16000` | 输入音频段的采样率，须与 pipeline 导出一致 |
| `device` | `auto` | ASR 推理设备，独立于 diarization 的 `device`：跟随模式下两进程同时占卡，显存紧张时可将 ASR 单独放到其他卡或 CPU |
| `model` | `FunAudioLLM/Fun-ASR-Nano-2512` | ASR 模型：本地目录 / ModelScope id（下载缓存到 `pretrained/modelscope`；缓存目录需含 `model.pt` + 配置才算命中，否则回退下载） |
| `prev_text_max_tokens` | `0` | 长段窗切续写时 prev_text 的 token 预算（取本段已累计文本尾部）。`0` = 自动（≈ `window_overlap` × 4 token/s，刚好覆盖重叠区）。 |
| `max_segment_duration` | `30.0` | 长段滑窗的窗口总长（秒）。超过该时长的段按窗口滑窗推理，每次推进（窗口 − 重叠）秒，单次推理成本有界 |
| `window_overlap` | `10.0` | 长段滑窗的重叠时长（秒）：窗口头部带这段已转写音频供 prev 尾部对齐，模型跳过重叠区只续写新内容；必须小于 `max_segment_duration` |
