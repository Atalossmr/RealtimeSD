# 实时说话人识别管线

本文档按当前代码实现说明 pipeline 的结构、chunk 级流程与关键参数。

## 总体流程

处理一段音频时，核心链路如下：

1. 读取音频并重采样到 `config.sample_rate`（默认 16k）
2. 按 `hop_duration` 沿时间轴推进，每次切出 `chunk_duration`（默认 10s）窗口
3. 对窗口运行 `pyannote/segmentation-3.0`，得到帧级多标签分数（局部 ≤3 人、帧级 ≤2 人，含重叠）
4. 每个 local slot 聚合纯净（非重叠）语音区提 ERes2NetV2 embedding；纯净区不足时回退 `overlap_fallback`
5. assigner 做 local->global 分配（后端可插拔，见 `diarization/cluster/assigners.py`）：默认 streaming 后端用 Hungarian 做联合分配，按阈值判定 `matched/new/fallback`，SMA 更新 centroid；**身份一次定案，新建 speaker 立即成为永久身份**
6. 只提交窗口中段 `hop_duration` 秒的帧级结果：streaming 后端即时进入 open-turn 写出管线，无缓冲、无延迟确认；deferred（离线）后端逐 chunk 暂存帧参数，音频结束统一聚类后用同一 writer 逻辑重放
7. 音频结束：闭合全部 open turn，writer 纯追加收尾——全程零重写

链路分层：

- `track_builder`：chunk 内 local track 聚合与 embedding 门控
- `assigners`：聚类后端接口与工厂（streaming / ahc），`clusterer` 为默认 streaming 后端实现
- `rttm_writer`：零重写 RTTM 写出（open-turn 管线）

设计要点：

- **纯流式：无 speaker merge、无 probationary/确认机制、无 RTTM 重写（含终局）、无 stable/延迟输出机制**
- speaker 身份一旦建立，流式期间永不改变；false split 与 false glue 均不可修复
- 因此阈值策略应为"宁可 glue 不可 split"：`new_speaker_threshold` 应适当高于旧版（probationary 架构可靠 absorb 修复 false split，阈值可以贴得很近）
- 收益：新 speaker 的首次发言在所属 chunk 提交区写出时即刻出现在输出中，输出延迟 ≈ 一个 hop + 沉默闭合确认

## 关键策略

### 1) local track 聚合与 overlap 回退

每个 local slot 在 chunk 内：

- 活跃总时长 < `min_local_activity_duration` → 跳过（不建 track、帧也不输出）
- 优先拼接非重叠纯净区（按平均活跃度降序，封顶 `max_segment_duration_for_embedding`）→ `non_overlap`，允许更新 centroid
- 纯净区总长 < `min_segment_duration_for_embedding` → 回退全活跃区拼接 → `overlap_fallback`，embedding 可提取但不允许常规更新
- 回退后仍不足 → 跳过

相关代码：`diarization/extract/track_builder.py`

### 2) 同窗联合分配（Hungarian）

同一 chunk 内所有 local track 与全部 global centroid 做 cost = 1 - cosine 的 Hungarian 分配，隐式实现 cannot-link：同一 chunk 的不同 local slot 不会分配给同一 global speaker。

判定规则：

- `matched`：相似度 ≥ `global_match_threshold`
- `new`：相似度 < `new_speaker_threshold` 且 track 时长 ≥ `min_segment_duration_for_new_speaker`（立即成为永久身份）
- `fallback`：介于两者之间，沿用最近 centroid，不更新

注意：纯流式架构下 false split 与 false glue 均不可修复，因此 `new_speaker_threshold` 应适当调高，偏保守建簇。

相关代码：`diarization/cluster/clusterer.py`

### 3) centroid 更新

- 全程 SMA 增量更新：`alpha = 1 / (count + 1)`
- 常规更新门控：track 时长 ≥ `min_segment_duration_for_centroid_update`
- `overlap_fallback` 片段可提 embedding 参与分配，但不更新 centroid

### 4) 提交区与零重写输出

- 重叠滑窗（`hop_duration < chunk_duration`）时，每窗口只提交中段 `hop_duration` 秒，两侧各留 `(chunk-hop)/2` 边界缓冲；首窗从 0 开始
- 每个 speaker 维护一个未闭合的 open turn（驻留内存）：后续帧/段间隔 ≤ `streaming_merge_gap` 即在写出前扩展拼接，跨 chunk 生效
- 沉默闭合：每 chunk 提交后检查，open turn 终点距 commit_end 已超过 `streaming_merge_gap` 即判定说话结束、提前闭合写出（与"下次开口回头闭合"产出等价，但尾段输出延迟从等到 EOF 降到约一个 hop）；EOF 时 `finalize` 兜底闭合全部残余 turn
- 短于 `min_segment_duration` 的 turn 在闭合时丢弃
- `finalize`：闭合全部 open turn，纯追加，不重写文件；最后在文件末尾以 `#` 注释写出内部 global id → RTTM speaker 的映射表（RTTM 编号按首次写出顺序分配）

相关代码：`diarization/cluster/rttm_writer.py`

## 模块分工

包按两个阶段拆成两个子模块，顶层为共享层与端到端组合：

```
diarization/
  app.py               # 端到端 CLI 主入口
  pipeline.py          # 端到端组合：ChunkDiarizationPipeline = extract + cluster
  config.py            # ChunkPipelineConfig + YAML 加载/键名校验 + CLI 定义 + 合并
  constants.py         # 路径常量
  schema.py            # 共享数据结构：ChunkObservation / ChunkArtifacts / SpeakerTurn / 调试 TypedDict
  utils/               # 通用工具子包：log / device / numeric / audio / paths / chunk_io（两阶段中间文件 <stem>.chunks.npz 存取）
  extract/             # 子模块 1：嵌入提取
    extractor.py       #   ChunkExtractor：模型加载、波形预处理、chunk 生成器（chunk 生产唯一来源）
    track_builder.py   #   chunk 内 local track 构造：纯净区优先、overlap_fallback 回退、时长门控
    models/            #   embedding_infer（ERes2NetV2）/ segmentation_infer（pyannote）/ hf_resolver
    app.py             #   提取阶段 CLI（extract_chunks.py 入口）
  cluster/             # 子模块 2：聚类与输出
    assigners.py       #   后端接口 BaseChunkAssigner + 工厂 build_assigner + 离线后端 AHCChunkAssigner
    clusterer.py       #   ChunkSpeakerClusterer：默认 streaming 后端（Hungarian + SMA，一次定案）
    rttm_writer.py     #   零重写 RTTM 写出（open-turn 管线，finalize 纯追加）
    runner.py          #   run_clustering：聚类消费循环（pipeline.py 与 cluster/app 共用）
    app.py             #   聚类阶段 CLI（cluster_chunks.py 入口）
```

- YAML 是全部调参项的唯一来源；CLI 仅保留 `--wav`、`--output_dir`、`--config`、
  模型/环境参数与 `--debug`、`--verbose`、`--show_rttm`；
- 新增聚类方法：在 `cluster/assigners.py` 实现 `BaseChunkAssigner` 并在
  `build_assigner` 注册即可，提取侧与输出侧都不用动。

## 参数分组与作用

以下调参项全部通过 `config.yaml` 配置（YAML 是唯一来源）。

### 1) 调度

- `chunk_duration`：窗口时长（segmentation-3.0 原生为 10s）
- `hop_duration`：推进步长；等于 chunk 时退化为非重叠

### 2) track 构造与 embedding

- `min_local_activity_duration`
- `min_segment_duration_for_embedding`（16k 下建议 ≥ 0.105s，防 NaN embedding）
- `max_segment_duration_for_embedding`
- `segment_batch_size`

### 3) 全局匹配与更新

- `max_speakers`
- `new_speaker_threshold`
- `global_match_threshold`
- `min_segment_duration_for_new_speaker`
- `min_segment_duration_for_centroid_update`

### 3b) 聚类后端

- `clustering_backend`：`streaming`（默认，增量聚类）/ `ahc`（离线层次聚类）
- `ahc_similarity_threshold` / `ahc_linkage`：仅 ahc 后端生效
- `save_embeddings`：把每文件全部 embedding 落盘为 `*.embeddings.npz`，便于离线实验复用

### 4) RTTM 输出

- `min_segment_duration`
- `streaming_merge_gap`
- `show_rttm`

## 输入输出与调试

### 两阶段独立运行（提取 / 聚类分离）

嵌入提取与聚类可以完全拆开运行，中间产物为 `<stem>.chunks.npz`
（逐 chunk 的 observations 含 embedding + 帧级输出参数，纯 numpy 无 pickle）：

```bash
# 阶段 1：嵌入提取（需要音频与模型，产出 chunks.npz）
python3 extract_chunks.py --wav <音频> --output_dir <dir> --config config.yaml

# 阶段 2：聚类 + RTTM 输出（只需 npz，后端由 YAML 的 clustering_backend 决定）
python3 cluster_chunks.py --input <dir或npz> --output_dir <dir> --config config.yaml
```

- 阶段 2 不加载任何模型，换后端/调阈值只需改 YAML 重跑，秒级完成；
- 两端共用的 chunk 生产逻辑在 `extract/extractor.py` 的 `iter_chunk_artifacts`，保证一致；
- 端到端用法（`python3 pipeline.py`）行为不变。

### 输入形式

`--wav` 支持：

- 单音频文件
- 音频目录
- 文本清单（每行一个路径）

支持后缀：`.wav`、`.mp3`、`.flac`

### 输出文件

每个输入音频默认输出：

- `*.<backend_tag>.rttm`（streaming 后端为 `*.streaming.rttm`，ahc 后端为 `*.ahc.rttm`）
- `run.log`
- `*.embeddings.npz`（仅 `save_embeddings: true` 时）

### 调试建议

常用启动：

```bash
python3 pipeline.py --debug --verbose --show_rttm ...
```

重点日志字段：

- `frame_decision`（chunk 级 local->global）
- `window_summary`（chunk 级汇总，兼容旧分析工具）
- `new_speakers` / `updated_speakers` / `skipped_updates`
- `current_global_speakers`

可用工具：

```bash
python3 tools/analyze_run_log.py --log ./exp/demo/run.log
```
