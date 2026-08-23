# 配置参数说明

`config.yaml` 是 diarization 管线全部调参项的唯一来源；CLI 仅保留运行时输入
（`--wav` / `--output_dir` / `--config`）、模型/环境参数与 `--debug` /
`--verbose` / `--show_rttm` 开关。YAML 键名会校验：合法键 =
`ChunkPipelineConfig` 字段名 ∪ CLI 参数名，写错键名会直接报错。

ASR 转写的调参项已拆到独立的 `asr.yaml`（`python3 -m asr.app` 读取，
字段定义见 `asr/config.py` 的 `AsrConfig`），说明见文末"ASR 转写"一节。

参数按模块分组，与 `diarization/config.py` 的 `ChunkPipelineConfig` 字段一一对应。
下表"默认"列为仓库自带 `config.yaml` 的生效值，与 dataclass 默认值保持一致。

## 运行环境

| 参数 | 默认 | 说明 |
|---|---|---|
| `device` | `auto` | 推理设备：`auto` / `cpu` / `cuda` / `cuda:0` |
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
| `segment_batch_size` | `4` | 批量提取 segment embedding 的 batch size |

track 构造逻辑（`diarization/extract/track_builder.py`）：纯净帧 = 本 slot 活跃
且无其他 slot 活跃的帧（≥2 人同时活跃即 overlap 帧）；纯净连通区固定按
commit 优先级选取（提交区内片段优先、不足再从两侧 margin 补齐，与
"embedding 给哪段帧定身份"对齐）、首尾拼接（间隔直接切掉不补静音）。纯净区总长不足
`min_segment_duration_for_embedding` 时回退全活跃区拼接（`overlap_fallback`，
可参与分配但不更新 centroid）；回退后仍不足则跳过该 slot。

## cluster 阶段：后端选择与通用

| 参数 | 默认 | 说明 |
|---|---|---|
| `clustering_backend` | `streaming` | `streaming`（增量聚类，纯流式）/ `ahc`（离线层次聚类：缓冲全部 embedding，音频结束一次聚类再重放输出）。新后端实现 `cluster/base.py` 接口并在 `build_clusterer` 注册即可 |

## cluster 阶段：streaming 后端（全局 speaker 匹配与 centroid 维护）

以下参数仅 `clustering_backend: streaming` 时生效。

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_speakers` | `50` | 最多维护的全局 speaker 数 |
| `global_match_threshold` | `0.55` | observation 命中已有 speaker 的主阈值（相似度 ≥ 此值判 `matched`） |
| `new_speaker_threshold` | `0.50` | 新建 speaker 的相似度阈值。**注意遮蔽语义**：`matched` 分支优先，new 只在 `similarity < new_speaker_threshold` 时触发，因此该阈值只有 ≤ match 阈值时才实际生效；两者之差构成 `fallback` 保守带（沿用但不更新）。当前值 0.55/0.50 是 aishell4-test 阈值扫描的最优点 |
| `min_segment_duration_for_new_speaker` | `1.00` | track 时长达到该值才允许新建 speaker（秒） |
| `min_segment_duration_for_centroid_update` | `1.50` | track 时长达到该值才允许更新 centroid（秒） |
| `merge_threshold` | `0.70` | 每次加入新片段后，若最相似的一对 centroid 相似度 ≥ 此值则合并（count 小者并入大者，centroid 按 count 加权平均）。只影响后续分配：raw RTTM 已写出行不改（历史行由 refined 级修正），被合并者退出后续聚类；本 chunk 未写出的分配改挂幸存 id |

身份一次定案 + 事后 merge：false split 可由 merge 修复，false glue 仍不可修复，
阈值策略应"宁可 glue 不可 split"。centroid 全程 SMA 增量更新（`alpha = 1/(count+1)`）。

## cluster 阶段：ahc 后端（离线层次聚类）

以下参数仅 `clustering_backend: ahc` 时生效。

| 参数 | 默认 | 说明 |
|---|---|---|
| `ahc_similarity_threshold` | `0.50` | AHC 余弦相似度阈值 |
| `ahc_linkage` | `average` | AHC 连接方式：`average` / `complete` / `single` |

## cluster 阶段：后处理（小样本簇强制合并）

两后端共用。样本计数低于 `post_merge_min_count` 的簇/speaker
整体并入质心余弦相似度最高的达标簇；最高相似度仍低于
`post_merge_min_similarity` 时保留原身份不并（防止把说得少的真实独立
speaker 错并）。计数口径：streaming 为有效 centroid 更新次数（`counts`，
new 建簇 + 纯净段且 ≥1.5s 的 matched 更新），ahc 为簇内 observation 数。
两后端输出统一分两级：raw 级（`*.raw.rttm` / `*.ahc.rttm`，append-only
零重写）+ refined 级（`*.refined.rttm`，streaming 随 merge 事件动态重生成
修正历史行，ahc 在 EOF 重放后生成；两者 EOF 最终刷新时叠加小样本合并）。

| 参数 | 默认 | 说明 |
|---|---|---|
| `post_merge_min_count` | `8` | 小样本簇的样本计数阈值，低于则作为被吸收候选；`0` = 关闭后处理。**推荐值 8**（aishell4-test 全量 20 文件标定：raw 级施加合并映射 DER 12.95%→12.45%，平台区 6–10、12 起误并真实少发言 speaker；refined 级端到端复跑 streaming DER 12.07%、ahc 13.22%→12.70%）。候选口径对比：有效更新计数（counts）显著优于 matched 判定次数与总分配次数（对比记录见 `exp/match_count_sweep/result.csv`，复扫用 `tools/sweep_params.py --set post_merge_min_count=...`）。该值同时是 streaming 前端 uncertain 标记的解除阈值（speaker 计数未达标期间在 viewer 显示为 uncertain） |
| `post_merge_min_similarity` | `0.0` | 强制合并的相似度下限：与目标簇质心相似度低于该值时保留原身份。aishell4 上 `0.0`（不拦截）最优——小簇几乎都是 false split；换数据集建议 `0.3` 兜底防错并（代价约 0.01pp） |

调参复验（复用 chunks.npz 秒级重跑聚类，详见本文末"实验存档"节）：

```bash
# 方式一：修改 YAML 中 post_merge_* 值后直接重放聚类阶段，再用 compute_der 评估
python3 -m diarization.cluster.app --input <dir>/embeddings --output_dir <dir> \
    --config config/config.yaml
# 方式二：统一参数扫描工具（任意字段网格，每个组合完整重放一次聚类）
python3 tools/sweep_params.py --input <dir>/embeddings --ref <参考 rttm 目录> \
    --config config/config.yaml --set post_merge_min_count=0,4,6,8,12 \
    --output <结果.csv>
```

## 分段音频导出（接流式 ASR）

以下参数仅 `separation_enabled: true` 且 `clustering_backend: streaming` 时生效。
每个 commit 区检测重叠帧：无重叠直接按 speaker 切片输出；有重叠则用 TIGER
分离整个 commit 区，能量门控 + embedding 匹配归属后按 speaker 输出音频段。

| 参数 | 默认 | 说明 |
|---|---|---|
| `separation_enabled` | `true` | 是否启用分段音频导出 |
| `separation_model` | `JusperLee/TIGER-speech` | TIGER 分离模型（HF，固定 2 路输出、16kHz），缓存到 `hf_cache_dir` |
| `separation_energy_ratio` | `0.05` | 能量门控：分离音轨在重叠帧区间的 RMS / 混合音频同区间 RMS，低于判为伪源（OSD 疑似误报）。仅一路过门控时该路匹配归属、落选 speaker 的重叠帧不输出；两路都不过门控才整窗回退原始音频 |
| `separation_min_match_similarity` | `0.10` | 2x2 匹配结果的最小余弦相似度，低于判分离质量不可靠，回退原始音频。aishell4 全量标定表明真/假重叠窗的 min_sim 分布高度重合（中位数 0.406 vs 0.388），该阈值无法区分 OSD 误报，仅防极端分离崩溃，不宜调高（0.30 时误伤 27.7% 真重叠窗）。归属匹配的参照 embedding 固定为本 chunk 观测（与分离音轨同时间窗、域最接近；候选 speaker 必有观测） |

## ASR 转写（Fun-ASR-Nano）

ASR 是独立于 pipeline 的离线阶段，分两步：

1. pipeline 侧（`config.yaml`）：`separation_enabled: true`（且
   `clustering_backend: streaming`）时构造分段导出器（无重叠纯切片、有重叠
   TIGER 分离，惰性加载），闭合的逐 speaker 音频段落盘为
   `segments/<uri>/*.wav` + `segments/<uri>.segments.jsonl`，pipeline 内不做转写。
2. 转写侧（`asr.yaml`）：对 segments 目录跑转写入口，逐段转写并按 start 排序写
   `<output_dir>/<uri>.transcript.jsonl`：

   ```bash
   # 一次性：管线结束后整体转写
   python3 -m asr.app --segments_dir <exp_dir>/segments --config config/asr.yaml
   # 跟随：与管线同时启动，新段即出即转，done 哨兵出现后收尾
   python3 -m asr.app --segments_dir <exp_dir>/segments --config config/asr.yaml \
       --follow --done_file <exp_dir>/.diarization_done
   ```

   转写调参项全部在 `config/asr.yaml`（不在 CLI 上，字段见 `asr/config.py`
   的 `AsrConfig`）。`python3 run.py ...` 默认即跟随模式：run.py 负责拉起
   ASR 进程、等待模型就绪后再启动管线，并在管线退出后落 done 哨兵。

转写侧（`asr.yaml`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `sample_rate` | `16000` | 输入音频段的采样率，须与 pipeline 的 `sample_rate` 一致 |
| `device` | `auto` | ASR 推理设备，独立于 diarization 的 `device`：跟随模式下两进程同时占卡，显存紧张时可将 ASR 单独放到其他卡或 CPU |
| `model` | `FunAudioLLM/Fun-ASR-Nano-2512` | ASR 模型：本地目录 / ModelScope id（下载缓存到 `pretrained/modelscope`；缓存目录需含 `model.pt` + 配置才算命中，否则回退下载） |
| `prev_text_max_tokens` | `0` | 长段窗切续写时 prev_text 的 token 预算（取本段已累计文本尾部）。`0` = 自动（≈ `window_overlap` × 4 token/s，刚好覆盖重叠区）。prev_text 语义是"本段音频前缀的转写"，仅长段窗切内部使用——跨段上下文不成立（文本与音频对不上时模型判转写完成而提前 EOS）；预算也不要手动调大：prev 估计覆盖 ≥ 窗口音频总长时同样提前 EOS（均实测确认） |
| `max_segment_duration` | `30.0` | 长段滑窗的窗口总长（秒）。超过该时长的段按窗口滑窗推理，每次推进（窗口 − 重叠）秒，单次推理成本有界 |
| `window_overlap` | `10.0` | 长段滑窗的重叠时长（秒）：窗口头部带这段已转写音频供 prev 尾部对齐，模型跳过重叠区只续写新内容；必须小于 `max_segment_duration` |

## RTTM 输出

| 参数 | 默认 | 说明 |
|---|---|---|
| `min_segment_duration` | `0.30` | RTTM 写出最短片段时长（秒），闭合时更短的 turn 丢弃 |
| `segment_merge_gap` | `0.25` | 同 speaker 相邻片段自动拼接允许的最大间隔（秒，跨 chunk 生效）；同时是沉默闭合判据：open turn 终点距 commit_end 超过该值即闭合写出 |
| `show_rttm` | `false` | 同步把新增 RTTM 行打印到控制台 |

## 调试

| 参数 | 默认 | 说明 |
|---|---|---|
| `debug` | `false` | 输出 chunk 级调试信息（`window_summary` / `new_speakers` / `updated_speakers` 等） |

## 附：实验存档

阈值与提取策略的结论均来自 `exp/threshold_sweep/*.csv`（aishell4-test 全量
20 文件，collar=0）。复验或换数据集调参：

```bash
# 1) 提取（产出 embeddings/<stem>.chunks.npz）
python3 -m diarization.extract.app --wav <音频> --output_dir <dir> --config config/config.yaml
# 2) 阈值扫描（不加载模型，秒级重跑；--set 任意字段网格）
python3 tools/sweep_params.py --input <dir>/embeddings --ref <ref_rttm_dir> \
    --config config/config.yaml \
    --set global_match_threshold=0.40,0.45,0.50,0.55,0.60 \
    --set new_speaker_threshold=0.40,0.45,0.50,0.55,0.60 \
    --output <结果.csv>
```
