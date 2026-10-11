# 实时说话人识别管线

本文档记录当前diarization包的结构、chunk 级流程与关键参数。

## 总体流程

处理一段音频时，核心链路如下：

1. 读取音频并重采样到 `config.sample_rate`（默认 16k）
2. 按 `hop_duration` 沿时间轴推进，每次切出 `chunk_duration`（默认 10s）窗口
3. 对窗口运行 `pyannote/segmentation-3.0`，得到帧级多标签分数（局部 ≤3 人、帧级 ≤2 人，含重叠）
4. 聚合每个local slot的非重叠语音区，整体提取一个embedding；纯净区不足时回退到 `overlap_fallback`，将重叠区域也加入语音区计算。
5. clusterer 负责进行local->global分配（后端可替换，见 `diarization/cluster/backends/`，项目内置了 streaming、ahc、fastdiar 三种）。默认的streaming后端用 Hungarian 做联合分配，按阈值判定 `matched/new/fallback`，SMA 更新 centroid；此外每次加入新片段后按 `merge_threshold` 尝试合并最相似的一对 speaker（小并入大）
6. 只提交窗口中段 `hop_duration` 秒的帧级结果：streaming 后端即时进入 open-turn 写出管线（raw 级），无缓冲、无延迟确认；ahc 后端（deferred，离线）逐 chunk 暂存帧参数，音频结束统一聚类后用同一 writer 逻辑重放
7. refined级：`post_process.RefinedRTTMWriter` 读取 raw RTTM + 当前 `merged_into`/centroid/counts 状态整体重生成 `*.refined.rttm`；streaming / fastdiar 在流式期间发生 merge 事件时即时刷新（修正 merge 前写出的旧身份行），无 merge 时按固定 chunk 间隔兜底刷新（避免逐 chunk 全量重写的 O(n²) I/O），ahc 无流式 merge、EOF 重放完成后刷新；三者 EOF 最终刷新均叠加小样本强制合并（`post_merge_min_count > 0` 时生效）
8. 音频结束：闭合全部 open turn，writer 纯追加收尾。

链路分层：

- `track_builder`：chunk 内 local track 聚合与 embedding 门控
- `backends`：聚类后端（streaming / ahc / fastdiar）与工厂，`base` 为后端接口
- `rttm_writer`：零重写 raw RTTM 写出（open-turn 管线）
- `post_process`：refined 级输出 + 小样本簇强制合并后处理

设计要点：

- **raw 级纯流式：无 probationary/确认机制、无 RTTM 重写（含终局）、无 stable/延迟输出机制**；修正全部发生在 refined 级（独立文件，可整体重生成）
- 说话人合并：每次加入新片段后，若最相似的一对 centroid 相似度 ≥ `merge_threshold` 则合并（count 小者并入大者）；raw 已写出的行不改，由 refined 级在下一次重生成时修正，被合并者退出后续聚类
- 收益：新 speaker 的首次发言在所属 chunk 提交区写出时即刻出现在输出中，输出延迟 ≈ 一个 hop + 沉默闭合确认

## 关键策略

### 1) local track 聚合与 overlap 回退

每个 local slot 在 chunk 内：

- 活跃总时长 < `min_local_activity_duration` → 跳过
- 优先拼接非重叠纯净区（连通区固定按 commit 优先级选取：提交区内片段优先、不足再从两侧 margin 补齐；总长封顶 `max_segment_duration_for_embedding`）→ `non_overlap`，允许更新 centroid
- 若纯净区总长 < `min_segment_duration_for_embedding` → 回退全活跃区拼接 → `overlap_fallback`，embedding 可提取但不允许常规更新
- 回退后仍不足 → 跳过

相关代码见`diarization/frontend/track_builder.py`

### 2) Hungarian联合分配

同一 chunk 内所有 local track 与全部 global centroid 做 cost = 1 - cosine 的 Hungarian 分配，隐式实现 cannot-link：同一 chunk 的不同 local slot 不会分配给同一 global speaker。

判定规则：

- `matched`：相似度 ≥ `global_match_threshold`
- `new`：相似度 < `new_speaker_threshold` 且 track 时长 ≥ `min_segment_duration_for_new_speaker`（立即成为永久身份）
- `fallback`：介于两者之间，沿用最近 centroid，不更新

相关代码见`diarization/cluster/backends/streaming.py`

### 3) 基于相似度的说话人合并

每次新片段分配完成后，计算当前所有说话人centroid之间的余弦相似度，若最相似的一对 ≥ `merge_threshold` 则合并：count 小者并入大者（count 相同保留 id 较小者），centroid 按 count 加权平均后重归一化。合并只影响后续分配：

- 已写出的 raw RTTM 行不受影响，历史行的归属修正由 refined 级（`post_process.RefinedRTTMWriter`）在下一次重生成时完成；
- 被合并 speaker 从 centroid 集中移除，不再参与后续 Hungarian 分配与 merge 判定；
- 尚未写出的分配中被合并的部分改挂到幸存 id；
- 被合并 id → 幸存 id 记录在 `merged_into`，合并事件以 `[merge]` 日志输出。

相关代码见`diarization/cluster/backends/streaming.py` 的 `_try_merge_speakers`

### 4) 提交区与零重写输出

- 重叠滑窗（`hop_duration < chunk_duration`）时，每窗口只提交中段 `hop_duration` 秒，两侧各留 `(chunk-hop)/2` 边界缓冲；首窗从 0 开始
- 对每个speaker各维护一个未闭合的 open turn，若下一次活跃的间隔 ≤ `segment_merge_gap` 即在写出前扩展拼接，跨 chunk 生效
- 短于 `min_segment_duration` 的 turn 在闭合时丢弃
- `finalize`：闭合全部 open turn。

相关代码：`diarization/cluster/rttm_writer.py`

### 5) fastdiar 后端（可选）

仿 fastdiar 的 OnlineClustering（帧级在线聚类）适配到 observation 级：贪心取最近质心（不用 Hungarian），相似度 ≥ `fastdiar_assign_threshold` 判 `matched`（对应 fastdiar 的 clust_th），否则满足建簇条件时开新簇、不满足时无阈值兜底强制指派最近质心（`forced`，不更新质心，保证观测不丢）；只有相似度 ≥ `fastdiar_confidence_threshold`（对应 fastdiar 的 confidence 门控）且允许更新的 matched 观测才作为 anchor 追加，质心 = anchor embedding 逐维中位数（L2 归一化）；每次新建 speaker 后按 `fastdiar_merge_threshold`（对应 fastdiar 的 merge_th）在质心相似度图上取连通分量整组并簇（survivor = anchor 数最多者）。与 fastdiar 的差异：无自我纠正窗口（流式契约要求一次定案）、置信度为段级代理。参数均仅 `clustering_backend: fastdiar` 时生效，见 `config/README.md`。

相关代码见`diarization/cluster/backends/fastdiar.py`

## 模块分工

包按「底座 + 两阶段业务」组织：共享底座（配置/契约/工具/模型/vendored
依赖）与业务实现同包；编排逻辑集中在 `diarization/pipeline.py`
（`process` / `run_diarization` 等），`diarization/app.py`
只是委托它的薄 CLI：

```
diarization/           # 底座 + 前端 + 嵌入提取 + 聚类 + RTTM 输出
  app.py               # 端到端 CLI（python3 -m diarization.app，委托 diarization.pipeline.run_diarization）
  pipeline.py          # 编排层：process（单/批量公开入口）/ DiarizationRunner / run_diarization /
                       #   run_der_evaluation（DER 评估）/ 入口脚本 CLI 辅助（add_common_args 等）
  constants.py         # 路径常量（BASE_DIR）
  config/              # 配置体系：base（通用 YAML/CLI 合并机制）+ pipeline（ChunkPipelineConfig + 管线 CLI helper）
  contracts/           # chunk 管线共享数据契约：schema（ChunkObservation / ChunkArtifacts / 调试 TypedDict）+
                       #   frames（commit 区帧裁剪，RTTM writer 使用；oracle 实验脚本镜像同口径）
  utils/               # 通用工具（log / paths / numeric / device / audio / rttm）+
                       #   chunk_io：两阶段中间文件 <stem>.chunks.npz 存取
  models/              # 模型封装：embedding_infer（ERes2NetV2 + RegionEmbedder）/ segmentation_infer（pyannote）/
                       #   hf_resolver / modelscope（模型缓存解析）
  speakerlab/          # vendored：3D-Speaker 依赖（ERes2NetV2 / FBank / fileio）+ md-eval.pl
  frontend/            # 子模块 1：extract 前端（embedding 之前的一切，与 extract 平级）
    frontend.py        #   ChunkFrontend：滑窗调度 + segmentation + track 构建，产 PreparedChunk
    oracle.py          #   OracleFrontend：ref RTTM 驱动的 oracle 前端（无模型，--oracle 模式）
    scheduling.py      #   iter_chunk_schedule：滑窗/commit 区几何（实时与 oracle 前端共用）
    track_builder.py   #   chunk 内 local track 构造：纯净区优先、overlap_fallback 回退、时长门控
  extract/             # 子模块 2：嵌入提取
    extractor.py       #   ChunkExtractor：前端 + RegionEmbedder 的组合 facade（chunk 生产唯一来源）
    oracle.py          #   oracle_chunk_artifacts：oracle 前端 + RegionEmbedder -> ChunkArtifacts
    app.py             #   提取阶段 CLI（python3 -m diarization.extract.app）
  cluster/             # 子模块 3：聚类与输出
    base.py            #   后端接口 BaseChunkClusterer
    backends/          #   内置后端与工厂 build_clusterer
      streaming.py     #     StreamingChunkClusterer：默认 streaming 后端（Hungarian + SMA，一次定案）
      ahc.py           #     AHCChunkClusterer：离线层次聚类后端
      fastdiar.py      #     FastdiarChunkClusterer：fastdiar 风格在线聚类（中位数质心 + 连通分量并簇）
    rttm_writer.py     #   零重写 RTTM 写出（open-turn 管线，finalize 纯追加；SpeakerTurn 定义于此）
    post_process.py    #   refined 级输出（RefinedRTTMWriter，merge 事件即时重生成 + 周期兜底刷新）+ 小样本簇强制合并（ahc finalize 内重映射）
    runner.py          #   run_clustering：聚类消费循环（pipeline.py 与 cluster/app 共用）
    app.py             #   聚类阶段 CLI（python3 -m diarization.cluster.app）
```

- 共享契约（contracts / config / models / utils）统一从本包底座子包取，
  由 `diarization/pipeline.py` 编排层做接线。
- 若要新增聚类方法，则需在 `cluster/backends/` 下新建模块实现 `BaseChunkClusterer`（具体接口见 `cluster/base.py`）并在 `build_clusterer` 注册。

## 输入输出与调试

### 两阶段独立运行

嵌入提取与聚类可以分别运行，中间产物为 `embeddings/<stem>.chunks.npz`
（逐 chunk 的 observations 含 embedding + 帧级输出参数）

```bash
# diarization 端到端（等价于 run.py）
python3 -m diarization.app --wav <音频> --output_dir <dir> --config config/config.yaml

# 阶段 1：嵌入提取（产出 embeddings/<stem>.chunks.npz）
python3 -m diarization.extract.app --wav <音频> --output_dir <dir> --config config/config.yaml

# 阶段 2：聚类 + RTTM 输出（需要阶段1输出的npz，输出写入 <dir>/rttm/）
python3 -m diarization.cluster.app --input <dir或npz> --output_dir <dir> --config config/config.yaml
```

- 两端共用的 chunk 定义在`utils/chunk_io.py`；

### CLI 参数

两个分阶段入口（`diarization.extract.app` 提取 / `diarization.cluster.app` 聚类）
与端到端入口 `diarization.app` 共用 `diarization/config/pipeline.py` 的同一个
参数解析器（完整编排入口为仓库根目录的 `run.py`）。全部调参项以 `config/config.yaml`
为唯一来源（逐项说明见
`config/README.md`），CLI 仅保留运行时输入、模型/环境覆盖与开关；合并优先级：
argparse 默认值 < YAML 配置 < 显式 CLI 参数。

| 参数 | 说明 |
| --- | --- |
| `--wav` | 输入音频（`extract.app` 与 `diarization.app` 必填）：单个 wav/flac/mp3、目录，或每行一个路径的文本文件 |
| `--input` | 仅 `diarization.cluster.app`（必填）：`.chunks.npz`、含 npz 的目录，或路径清单文本 |
| `--output_dir` | 必填；输出目录（RTTM 在 `rttm/`、npz 在 `embeddings/`、日志在 `logs/`） |
| `--config` | YAML 配置文件路径，默认 `config/config.yaml` |
| `--model_path` | 预训练 ERes2NetV2 checkpoint 路径；缺省自动从 ModelScope 下载默认模型 |
| `--device` | 运行设备（`auto`/`cpu`/`cuda:0`），默认 `auto`；cluster 阶段不加载模型，无需设置 |
| `--model_type` | speaker encoder 类型，当前仅支持 `eres2netv2` |
| `--segmentation_model` | pyannote segmentation 模型名（仅 extract 阶段使用） |
| `--hf_token` | Hugging Face token |
| `--hf_cache_dir` | Hugging Face 模型缓存目录，默认仓库内 `pretrained/huggingface` |
| `--show_rttm` | 把新生成的 RTTM 行同步输出到控制台 |
| `--debug` | 输出 chunk 级 debug 信息 |
| `--verbose` | 启用 DEBUG 级日志 |

模型类参数（`--model_path` / `--device` / `--model_type` /
`--segmentation_model` / `--hf_*`）在 cluster 阶段不生效，挂在共用解析器上
仅为兼容同一份 `config.yaml`。

### 输入形式

`--wav` 支持：

- 单音频文件
- 音频目录
- 文本清单（每行一个路径）

支持后缀：`.wav`、`.mp3`、`.flac`

### 输出文件

输出目录按内容类型分子文件夹。每个输入音频默认输出：

- `rttm/*.raw.rttm`（各后端的 raw 级文件名后缀均为 `raw`）
- `rttm/*.refined.rttm`（经过后处理的最终输出）
- `rttm/*.speakers.json`（各后端通用；refined 级 sidecar）
- `embeddings/*.chunks.npz`（仅提取阶段 `python3 -m diarization.extract.app` 产出）
- `logs/run.log`

### 文件交互接口

diarization 的产物经输出目录的文件被下游消费（如 DER 评估与
tools/ 下的分析/扫描工具消费 refined RTTM 与 chunks.npz）。
**核心约定：所有对外文件里的 speaker id 都是 clusterer 的 global id**
（含 RTTM 行内的 speaker 字段）。
写读时序保证：追加在依赖之后、JSON/RTTM 重写均走临时文件 + 原子替换，保证消费者永远读不到半写文件。

#### `rttm/{uri}.raw.rttm` + `rttm/{uri}.refined.rttm`

- raw 级（`.raw.rttm`，各后端统一）：append-only 零重写，行写出即最终；refined：整体重生成（原子替换），merge 历史修正（streaming / fastdiar 有流式 merge）+ EOF 小样本合并后的最终输出，**下游应消费 refined**；
- 行格式（标准 RTTM）：`SPEAKER <uri> 0 <start> <dur> <NA> <NA> <speaker global id> <NA> <NA>`。

#### `rttm/{uri}.speakers.json`（refined 级 sidecar）

随 refined RTTM 同步原子更新，记录各 speaker 的统计与合并状态：

```json
{
  "uri": "...", "final": false,
  "post_merge_min_count": 8,
  "speakers": [
    {"id": 0, "duration": 123.4, "count": 12,
     "uncertain": false, "merged_into": null}
  ],
  "merge_events": [{"absorbed": 9, "survivor": 8, "kind": "merge"}]
}
```

- `speakers[].id` = global id（与 RTTM 行内的 speaker 字段同源）；
- `duration`：总发声时长（秒，展示用）；被并 speaker 置 `null`（时长已计入幸存者）；
- `count`：样本计数（streaming 为有效 centroid 更新次数，fastdiar 为 anchor 数）；被并 speaker 置 `null`；
- `uncertain`：`post_merge_min_count > 0` 且未合并、计数未达标；
- `merged_into`：最终幸存 global id（含流式 merge 链 + post-merge 两级），未并时为 `null`；
- `merge_events[].kind`：`merge`（流式期间）/ `post_merge`（EOF 小样本强制合并）；
- `final`：false = 管线仍在运行，状态还会变化；true = EOF 最终刷新。

#### `embeddings/{uri}.chunks.npz`（extract → cluster 两阶段中间产物）

逐 chunk 的 observations（含 embedding）+ 帧级输出参数，纯 numpy 无 pickle；
字段布局见 `utils/chunk_io.py` 的模块 docstring。`diarization.cluster.app`
消费它重放聚类（换后端/调参秒级完成）；离线分析需要的 embedding 也从这里取。

### 调试建议

常用启动（端到端入口为 `run.py`）：

```bash
python3 run.py <音频.wav> --debug --verbose --show_rttm
```

重点日志字段：

- `frame_decision`（chunk 级 local->global）
- `window_summary`（chunk 级汇总，兼容旧分析工具）
- `new_speakers` / `updated_speakers` / `skipped_updates`
- `current_global_speakers`

可用工具：

```bash
python3 tools/analyze_log.py ./exp/common/demo/logs/run.log
```
