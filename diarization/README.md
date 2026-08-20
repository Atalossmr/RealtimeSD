# 实时说话人识别管线

本文档按当前代码实现说明 pipeline 的结构、chunk 级流程与关键参数。

## 总体流程

处理一段音频时，核心链路如下：

1. 读取音频并重采样到 `config.sample_rate`（默认 16k）
2. 按 `hop_duration` 沿时间轴推进，每次切出 `chunk_duration`（默认 10s）窗口
3. 对窗口运行 `pyannote/segmentation-3.0`，得到帧级多标签分数（局部 ≤3 人、帧级 ≤2 人，含重叠）
4. 每个 local slot 聚合纯净（非重叠）语音区提 ERes2NetV2 embedding；纯净区不足时回退 `overlap_fallback`
5. assigner 做 local->global 分配（后端可插拔，见 `diarization/cluster/backends/`）：默认 streaming 后端用 Hungarian 做联合分配，按阈值判定 `matched/new/fallback`，SMA 更新 centroid；每次加入新片段后按 `merge_threshold` 尝试合并最相似的一对 speaker（小并入大）
6. 只提交窗口中段 `hop_duration` 秒的帧级结果：streaming 后端即时进入 open-turn 写出管线（raw 级），无缓冲、无延迟确认；ahc 后端（deferred，离线）逐 chunk 暂存帧参数，音频结束统一聚类后用同一 writer 逻辑重放
7. refined 级（仅 streaming）：`post_merge.RefinedRTTMWriter` 监听 merge 事件，每次 merge 后读取 raw RTTM + 当前 `merged_into`/centroid 状态整体重生成 `*.refined.rttm`（修正 merge 前写出的旧身份行）；EOF 时最终刷新并叠加小样本强制合并（`post_merge_min_speech_duration > 0` 时生效）
8. 音频结束：闭合全部 open turn，writer 纯追加收尾——raw 全程零重写，修正只发生在 refined 级

链路分层：

- `track_builder`：chunk 内 local track 聚合与 embedding 门控
- `backends`：聚类后端（streaming / ahc）与工厂，`base` 为后端接口
- `rttm_writer`：零重写 raw RTTM 写出（open-turn 管线）
- `post_merge`：refined 级输出 + 小样本簇强制合并后处理

设计要点：

- **raw 级纯流式：无 probationary/确认机制、无 RTTM 重写（含终局）、无 stable/延迟输出机制**；修正全部发生在 refined 级（独立文件，可整体重生成）
- 分配一次定案，但支持事后 merge：每次加入新片段后，若最相似的一对 centroid 相似度 ≥ `merge_threshold` 则合并（count 小者并入大者）；raw 已写出的行不改，由 refined 级在下一次重生成时修正，被合并者退出后续聚类
- false split 可由 merge 修复，false glue 仍不可修复，因此阈值策略仍为"宁可 glue 不可 split"：`new_speaker_threshold` 应适当调高
- 收益：新 speaker 的首次发言在所属 chunk 提交区写出时即刻出现在输出中，输出延迟 ≈ 一个 hop + 沉默闭合确认

## 关键策略

### 1) local track 聚合与 overlap 回退

每个 local slot 在 chunk 内：

- 活跃总时长 < `min_local_activity_duration` → 跳过（不建 track、帧也不输出）
- 优先拼接非重叠纯净区（连通区选取优先级由 `region_priority` 决定：`commit`（默认）提交区内片段优先、不足再从两侧 margin 补齐；`latest` 最新优先，超封顶丢弃最早部分；总长封顶 `max_segment_duration_for_embedding`，压线段保留头部截断）→ `non_overlap`，允许更新 centroid
- 纯净区总长 < `min_segment_duration_for_embedding` → 回退全活跃区拼接 → `overlap_fallback`，embedding 可提取但不允许常规更新
- 回退后仍不足 → 跳过

相关代码：`diarization/extract/track_builder.py`

### 2) 同窗联合分配（Hungarian）

同一 chunk 内所有 local track 与全部 global centroid 做 cost = 1 - cosine 的 Hungarian 分配，隐式实现 cannot-link：同一 chunk 的不同 local slot 不会分配给同一 global speaker。

判定规则：

- `matched`：相似度 ≥ `global_match_threshold`
- `new`：相似度 < `new_speaker_threshold` 且 track 时长 ≥ `min_segment_duration_for_new_speaker`（立即成为永久身份）
- `fallback`：介于两者之间，沿用最近 centroid，不更新

注意：false split 可由 merge 修复（见下节），false glue 不可修复，因此 `new_speaker_threshold` 应适当调高，偏保守建簇。

相关代码：`diarization/cluster/backends/streaming.py`

### 3) 事后 merge

每次加入新片段（observation 完成分配）后，计算全部 centroid 两两余弦相似度，若最相似的一对 ≥ `merge_threshold` 则合并：count 小者并入大者（count 相同保留 id 较小者），centroid 按 count 加权平均后重归一化。合并只影响后续分配：

- 已写出的 raw RTTM 行不受影响（writer 全程 append-only），历史行的归属修正由 refined 级（`post_merge.RefinedRTTMWriter`）在下一次重生成时完成；
- 被合并 speaker 从 centroid 集中移除，不再参与后续 Hungarian 分配与 merge 判定；
- 本 chunk 尚未写出的分配（含 Hungarian 结果里指向被合并者的 stale 匹配）改挂到幸存 id；
- 被合并 id → 幸存 id 记录在 `merged_into`，合并事件以 `[merge]` 日志输出。

相关代码：`diarization/cluster/backends/streaming.py` 的 `_try_merge_speakers`

### 4) centroid 更新

- 全程 SMA 增量更新：`alpha = 1 / (count + 1)`
- 常规更新门控：track 时长 ≥ `min_segment_duration_for_centroid_update`
- `overlap_fallback` 片段可提 embedding 参与分配，但不更新 centroid

### 5) 提交区与零重写输出

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
    app.py             #   提取阶段 CLI（python3 -m diarization.extract.app）
  cluster/             # 子模块 2：聚类与输出
    base.py            #   后端接口 BaseChunkAssigner
    backends/          #   内置后端与工厂 build_assigner
      streaming.py     #     ChunkSpeakerClusterer：默认 streaming 后端（Hungarian + SMA，一次定案）
      ahc.py           #     AHCChunkAssigner：离线层次聚类后端
    rttm_writer.py     #   零重写 RTTM 写出（open-turn 管线，finalize 纯追加）
    post_merge.py      #   refined 级输出（RefinedRTTMWriter，merge 事件动态重生成）+ 小样本簇强制合并（ahc finalize 内重映射）
    runner.py          #   run_clustering：聚类消费循环（pipeline.py 与 cluster/app 共用）
    app.py             #   聚类阶段 CLI（python3 -m diarization.cluster.app）
```

- YAML 是全部调参项的唯一来源；CLI 仅保留 `--wav`、`--output_dir`、`--config`、
  模型/环境参数与 `--debug`、`--verbose`、`--show_rttm`；
- 新增聚类方法：在 `cluster/backends/` 下新建模块实现 `BaseChunkAssigner`
  （接口见 `cluster/base.py`）并在 `build_assigner` 注册即可，
  提取侧与输出侧都不用动。

## 参数分组与作用

以下调参项全部通过 `config/config.yaml` 配置（YAML 是唯一来源），按模块分组；逐项详细说明见 `config/README.md`

### 1) extract 阶段：调度

- `chunk_duration`：窗口时长（segmentation-3.0 原生为 10s）
- `hop_duration`：推进步长；等于 chunk 时退化为非重叠

### 2) extract 阶段：track 构造与 embedding

- `min_local_activity_duration`
- `min_segment_duration_for_embedding`（16k 下建议 ≥ 0.105s，防 NaN embedding）
- `max_segment_duration_for_embedding`
- `region_priority`（纯净区选取优先级：`commit` 提交区优先（默认）/ `latest` 最新优先）
- `segment_batch_size`

### 3) cluster 阶段：后端选择与通用

- `clustering_backend`：`streaming`（默认，增量聚类）/ `ahc`（离线层次聚类）
- `save_embeddings`：把每文件全部 embedding 落盘为 `*.embeddings.npz`，便于离线实验复用

### 4) cluster 阶段：streaming 后端

- `max_speakers`
- `new_speaker_threshold`
- `global_match_threshold`
- `min_segment_duration_for_new_speaker`
- `min_segment_duration_for_centroid_update`
- `merge_threshold`（每次加入新片段后，最相似的一对 centroid 相似度 ≥ 此值即合并，小并入大）

### 5) cluster 阶段：ahc 后端

- `ahc_similarity_threshold` / `ahc_linkage`：仅 ahc 后端生效

### 5b) cluster 阶段：后处理（小样本簇强制合并，两后端共用）

- `post_merge_min_speech_duration`：总发声时长低于该值的簇并入质心最相似的达标簇；ahc 在 finalize 内重映射，streaming 在 refined 级 EOF 最终刷新时叠加（raw 文件不动）；0 = 关闭
- `post_merge_min_similarity`：强制合并的相似度下限，低于则保留原身份

### 6) RTTM 输出

- `min_segment_duration`
- `streaming_merge_gap`
- `show_rttm`

## 输入输出与调试

### 两阶段独立运行（提取 / 聚类分离）

嵌入提取与聚类可以完全拆开运行，中间产物为 `<stem>.chunks.npz`
（逐 chunk 的 observations 含 embedding + 帧级输出参数，纯 numpy 无 pickle）：

```bash
# 阶段 1：嵌入提取（需要音频与模型，产出 chunks.npz）
python3 -m diarization.extract.app --wav <音频> --output_dir <dir> --config config/config.yaml

# 阶段 2：聚类 + RTTM 输出（只需 npz，后端由 YAML 的 clustering_backend 决定）
python3 -m diarization.cluster.app --input <dir或npz> --output_dir <dir> --config config/config.yaml
```

- 阶段 2 不加载任何模型，换后端/调阈值只需改 YAML 重跑，秒级完成；
- 两端共用的 chunk 生产逻辑在 `extract/extractor.py` 的 `iter_chunk_artifacts`，保证一致；
- 端到端用法（`python3 -m diarization.app`）行为不变。

### 输入形式

`--wav` 支持：

- 单音频文件
- 音频目录
- 文本清单（每行一个路径）

支持后缀：`.wav`、`.mp3`、`.flac`

### 输出文件

每个输入音频默认输出：

- `*.<backend_tag>.rttm`（streaming 后端为 `*.raw.rttm`，ahc 后端为 `*.ahc.rttm`）
- `*.refined.rttm`（仅 streaming 后端；merge 事件动态重生成 + EOF 叠加小样本合并，为最终输出）
- `run.log`
- `*.embeddings.npz`（仅 `save_embeddings: true` 时）

### 文件交互接口（下游消费契约）

diarization 与 ASR / viewer 之间无 IPC，全部经共享输出目录的文件交互。
**核心约定：所有对外文件里的 speaker id 都是 assigner 的 global id**
（RTTM 行内的输出编号除外，可用行内/文件末尾的 id 映射表换算）。
写读时序保证：追加在依赖之后、JSON/RTTM 重写均走临时文件 + 原子替换，
消费者永远读不到半写文件。

#### `{uri}.raw.rttm` / `{uri}.refined.rttm`（streaming 后端）

- raw：append-only 零重写，行写出即最终；refined：整体重生成（原子替换），merge 历史修正 + EOF 小样本合并后的最终输出，**下游应消费 refined**；
- 行格式（标准 RTTM）：`SPEAKER <uri> 0 <start> <dur> <NA> <NA> <输出编号> <NA> <NA>`；
- 文件末尾 `#` 注释映射表：`#   <global_id> -> <输出编号>`（refined 为合并修正后的最终映射）。

#### `{uri}.speakers.json`（streaming 后端，refined 级 sidecar）

随 refined RTTM 同步原子更新；viewer 用它做 uncertain 标记与合并展示：

```json
{
  "uri": "...", "final": false,
  "post_merge_min_speech_duration": 30.0,
  "speakers": [
    {"id": 0, "output_id": 1, "duration": 123.4,
     "uncertain": false, "merged_into": null}
  ],
  "merge_events": [{"absorbed": 9, "survivor": 8, "kind": "merge"}]
}
```

- `speakers[].id` = global id（与 transcript 的 `speaker_id` 同源）；
- `duration`：总发声时长（秒）；被并 speaker 置 `null`（时长已计入幸存者）；
- `uncertain`：`post_merge_min_speech_duration > 0` 且未合并、时长未达标；
- `merged_into`：最终幸存 global id（含流式 merge 链 + post-merge 两级），未并时为 `null`；
- `merge_events[].kind`：`merge`（流式期间）/ `post_merge`（EOF 小样本强制合并）；
- `final`：false = 管线仍在运行，状态还会变化；true = EOF 最终刷新。

#### `{uri}.segments.jsonl` + `segments/{uri}/`（仅 `separation_enabled: true`）

接 ASR 的分段音频导出（详见 `separation/exporter.py`）：

- manifest：append-only JSONL，每行
  `{"uri", "speaker_id"(global id), "start", "end", "path"}`；
  **行在其指向的 wav 落盘之后才追加**，读到行即可读音频；
- 音频段：`segments/{uri}/spk{speaker_id}_{start}_{end}.wav`，
  单声道、采样率 = `config.sample_rate`（默认 16kHz）；
- `speaker_id` 为**合并前**的 global id（段一写定音，后续合并不回溯）。

#### `{uri}.embeddings.npz`（仅 `save_embeddings: true`）

npz 字段：`embeddings`(N×dim, L2 归一化) / `local_idx` / `start` / `end` / `duration`，
供离线实验复用（配合 `python3 -m diarization.cluster.app` 重放聚类调参）。

#### `{uri}.chunks.npz`（extract → cluster 两阶段中间产物）

逐 chunk 的 observations（含 embedding）+ 帧级输出参数，纯 numpy 无 pickle；
字段布局见 `utils/chunk_io.py` 的模块 docstring。`diarization.cluster.app`
消费它重放聚类（换后端/调参秒级完成）。

#### 协调哨兵（由 run.py 管理，diarization 本身不写）

- `.diarization_done`：run.py 在管线退出后 touch；ASR 据此收尾退出，viewer 据此熄灭 LIVE。

### 调试建议

常用启动：

```bash
python3 -m diarization.app --debug --verbose --show_rttm ...
```

重点日志字段：

- `frame_decision`（chunk 级 local->global）
- `window_summary`（chunk 级汇总，兼容旧分析工具）
- `new_speakers` / `updated_speakers` / `skipped_updates`
- `current_global_speakers`

可用工具：

```bash
python3 tools/analyze_log.py ./exp/demo/run.log
```
