# 配置参数说明

`config.yaml` 是全部调参项的唯一来源；CLI 仅保留运行时输入（`--wav` /
`--output_dir` / `--config`）、模型/环境参数与 `--debug` / `--verbose` /
`--show_rttm` 开关。YAML 键名会校验：合法键 = `ChunkPipelineConfig` 字段名
∪ CLI 参数名，写错键名会直接报错。

参数按模块分组，与 `diarization/config.py` 的 `ChunkPipelineConfig` 字段一一对应。

## 运行环境

| 参数 | 默认 | 说明 |
|---|---|---|
| `device` | `cuda`（YAML 生效值；dataclass 默认 `cpu`，CLI 默认 `auto`） | 推理设备：`auto` / `cpu` / `cuda` / `cuda:0` |
| `hf_cache_dir` | `./pretrained/huggingface` | Hugging Face 缓存目录，相对路径按仓库根目录解析 |

## extract 阶段：模型

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_type` | `eres2netv2` | 说话人 embedding 模型类型（当前仅支持 eres2netv2） |
| `model_path` | 空 | 本地 embedding checkpoint 路径；为空时自动下载 ModelScope 默认模型到 `pretrained/modelscope`（CLI 参数，也可写在 YAML） |
| `segmentation_model` | `pyannote/segmentation-3.0` | pyannote 局部分割模型名，需 HF 授权（`HF_TOKEN`） |

## extract 阶段：chunk 调度

| 参数 | 默认 | 说明 |
|---|---|---|
| `chunk_duration` | `10.0` | 窗口时长（秒），segmentation-3.0 原生窗口即 10s |
| `hop_duration` | `5.0` | 推进步长（秒）。`hop < chunk` 为重叠滑窗：每窗只提交中段 hop 秒，两侧各留 `(chunk-hop)/2` 边界缓冲；`hop == chunk` 退化为非重叠 |

## extract 阶段：local track 构造与 embedding 提取

| 参数 | 默认 | 说明 |
|---|---|---|
| `min_local_activity_duration` | `0.30` | local slot 在 chunk 内的最小累计活跃时长（秒），低于则不建 track、帧也不输出 |
| `min_segment_duration_for_embedding` | `0.30` | 允许提取 embedding 的最短拼接时长（秒）。**不要低于 0.105s@16k**（1680 samples），否则可能产生 NaN embedding 并触发 Hungarian 的 `matrix contains invalid numeric entries` |
| `max_segment_duration_for_embedding` | `4.0` | 单 track 用于提 embedding 的最大拼接时长（秒）。实验结论：0→4s 是收益区，超过 4s 饱和（不限上限 DER 无变化），不建议调大 |
| `region_priority` | `commit` | 纯净区选取优先级。`commit`：提交区内片段优先、不足再从两侧 margin 补齐（与"embedding 给哪段帧定身份"对齐，低阈值区更鲁棒）；`latest`：最新优先（旧默认，最优阈值下 DER 略优 0.1pp 量级）。压线段均保留头部截断 |
| `segment_batch_size` | `8` | 批量提取 segment embedding 的 batch size |

track 构造逻辑（`diarization/extract/track_builder.py`）：纯净帧 = 本 slot 活跃
且无其他 slot 活跃的帧（≥2 人同时活跃即 overlap 帧）；纯净连通区按
`region_priority` 选取、首尾拼接（间隔直接切掉不补静音）。纯净区总长不足
`min_segment_duration_for_embedding` 时回退全活跃区拼接（`overlap_fallback`，
可参与分配但不更新 centroid）；回退后仍不足则跳过该 slot。

## cluster 阶段：后端选择与通用

| 参数 | 默认 | 说明 |
|---|---|---|
| `clustering_backend` | `streaming` | `streaming`（增量聚类，纯流式）/ `ahc`（离线层次聚类：缓冲全部 embedding，音频结束一次聚类再重放输出）。新后端实现 `cluster/base.py` 接口并在 `build_assigner` 注册即可 |
| `save_embeddings` | `false` | 把每文件全部 embedding 落盘为 `<stem>.embeddings.npz`，便于离线实验复用 |

## cluster 阶段：streaming 后端（全局 speaker 匹配与 centroid 维护）

以下参数仅 `clustering_backend: streaming` 时生效。

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_speakers` | `50` | 最多维护的全局 speaker 数 |
| `global_match_threshold` | `0.55` | observation 命中已有 speaker 的主阈值（相似度 ≥ 此值判 `matched`） |
| `new_speaker_threshold` | `0.50` | 新建 speaker 的相似度阈值。**注意遮蔽语义**：`matched` 分支优先，new 只在 `similarity < new_speaker_threshold` 时触发，因此该阈值只有 ≤ match 阈值时才实际生效；两者之差构成 `fallback` 保守带（沿用但不更新）。当前值 0.55/0.50 是 aishell4-test 阈值扫描的最优点 |
| `min_segment_duration_for_new_speaker` | `0.50` | track 时长达到该值才允许新建 speaker（秒） |
| `min_segment_duration_for_centroid_update` | `1.50` | track 时长达到该值才允许更新 centroid（秒） |

纯流式架构下身份一次定案：false split 与 false glue 均不可修复，阈值策略应
"宁可 glue 不可 split"。centroid 全程 SMA 增量更新（`alpha = 1/(count+1)`）。

## cluster 阶段：ahc 后端（离线层次聚类）

以下参数仅 `clustering_backend: ahc` 时生效。

| 参数 | 默认 | 说明 |
|---|---|---|
| `ahc_similarity_threshold` | `0.50` | AHC 余弦相似度阈值 |
| `ahc_linkage` | `average` | AHC 连接方式：`average` / `complete` / `single` |

## 分段音频导出（接流式 ASR）

以下参数仅 `separation_enabled: true` 且 `clustering_backend: streaming` 时生效。
每个 commit 区检测重叠帧：无重叠直接按 speaker 切片输出；有重叠则用 TIGER
分离整个 commit 区，能量门控 + embedding 匹配归属后按 speaker 输出音频段。

| 参数 | 默认 | 说明 |
|---|---|---|
| `separation_enabled` | `false` | 是否启用分段音频导出 |
| `separation_model` | `JusperLee/TIGER-speech` | TIGER 分离模型（HF，固定 2 路输出、16kHz），缓存到 `hf_cache_dir` |
| `separation_energy_ratio` | `0.10` | 能量门控：分离音轨在重叠帧区间的 RMS / 混合音频同区间 RMS，低于判为伪源（OSD 疑似误报），整窗回退原始音频 |
| `separation_min_match_similarity` | `0.10` | 2x2 匹配结果的最小余弦相似度，低于判分离质量不可靠，回退原始音频。aishell4 全量标定表明真/假重叠窗的 min_sim 分布高度重合（中位数 0.406 vs 0.388），该阈值无法区分 OSD 误报，仅防极端分离崩溃，不宜调高（0.30 时误伤 27.7% 真重叠窗） |
| `separation_match_reference` | `observation` | 分离音轨归属匹配的参照 embedding：`observation`（默认，本 chunk 观测，与分离音轨同时间窗、域最接近；候选 speaker 必有观测）/ `centroid`（全局质心，不受本 chunk 观测质量影响，更稳但分数偏低） |

## RTTM 输出

| 参数 | 默认 | 说明 |
|---|---|---|
| `min_segment_duration` | `0.30` | RTTM 写出最短片段时长（秒），闭合时更短的 turn 丢弃 |
| `streaming_merge_gap` | `0.25` | 同 speaker 相邻片段自动拼接允许的最大间隔（秒，跨 chunk 生效）；同时是沉默闭合判据：open turn 终点距 commit_end 超过该值即闭合写出 |
| `show_rttm` | `false` | 同步把新增 RTTM 行打印到控制台 |

## 调试

| 参数 | 默认 | 说明 |
|---|---|---|
| `debug` | `false` | 输出 chunk 级调试信息（`window_summary` / `new_speakers` / `updated_speakers` 等） |

## 附：实验存档

阈值与提取策略的结论均来自 `exp/threshold_sweep/*.csv`（aishell4-test 全量
20 文件，collar=0）。复验或换数据集调参：

```bash
# 1) 提取（产出 chunks.npz）
python3 extract_chunks.py --wav <音频> --output_dir <dir> --config config/config.yaml
# 2) 阈值扫描（不加载模型，秒级重跑）
python3 tools/sweep_thresholds.py --input <dir> --ref <ref_rttm_dir> \
    --config config/config.yaml --output <结果.csv>
```
