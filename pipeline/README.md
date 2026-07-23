# 实时说话人识别管线

本文档按当前代码实现说明 pipeline 的结构、chunk 级流程与关键参数。

## 总体流程

处理一段音频时，核心链路如下：

1. 读取音频并重采样到 `config.sample_rate`（默认 16k）
2. 按 `hop_duration` 沿时间轴推进，每次切出 `chunk_duration`（默认 10s）窗口
3. 对窗口运行 `pyannote/segmentation-3.0`，得到帧级多标签分数（局部 ≤3 人、帧级 ≤2 人，含重叠）
4. 每个 local slot 聚合纯净（非重叠）语音区提 ERes2NetV2 embedding；纯净区不足时回退 `overlap_fallback`
5. clustering 用 Hungarian 做 local->global 联合分配，按阈值判定 `matched/new/fallback`，SMA 更新 centroid
6. 新建 speaker 进入 probationary 试用期，达标转正或与 confirmed 过于相似时被吸收
7. 只提交窗口中段 `hop_duration` 秒的帧级结果：confirmed 即时写入 open-turn 管线，probationary 先入内存缓冲，身份定案（转正/吸收）后 flush 追加
8. 音频结束：定案残余 probationary 并 flush 其缓冲，writer 纯追加收尾——全程零重写

链路分层：

- `track_builder`：chunk 内 local track 聚合与 embedding 门控
- `clusterer`：local->global 分配 + centroid 维护 + probationary 状态机
- `rttm_writer`：零重写 RTTM 写出（open-turn 管线 + probationary 缓冲 flush）

设计要点：

- **无 speaker merge、无 RTTM 重写（含终局）、无 stable/延迟输出机制**
- confirmed speaker 身份一旦建立，流式期间永不改变
- 身份修正只有一个出口：probationary 吸收；被吸收者的输出因尚在缓冲而未写出，
  flush 时直接以目标 id 落盘，已写出的行永不回改
- 代价：新 speaker 的首次发言延迟到其 probation 定案时才出现在输出中

## 关键策略

### 1) local track 聚合与 overlap 回退

每个 local slot 在 chunk 内：

- 活跃总时长 < `min_local_activity_duration` → 跳过（不建 track、帧也不输出）
- 优先拼接非重叠纯净区（按平均活跃度降序，封顶 `max_segment_duration_for_embedding`）→ `non_overlap`，允许更新 centroid
- 纯净区总长 < `min_segment_duration_for_embedding` → 回退全活跃区拼接 → `overlap_fallback`，embedding 可提取但不允许常规更新
- 回退后仍不足 → 跳过

相关代码：`pipeline/track_builder.py`

### 2) 同窗联合分配（Hungarian）

同一 chunk 内所有 local track 与全部 global centroid 做 cost = 1 - cosine 的 Hungarian 分配，隐式实现 cannot-link：同一 chunk 的不同 local slot 不会分配给同一 global speaker。

判定规则：

- `matched`：相似度 ≥ `global_match_threshold`
- `new`：相似度 < `new_speaker_threshold` 且 track 时长 ≥ `min_segment_duration_for_new_speaker`（建 probationary）
- `fallback`：介于两者之间，沿用最近 centroid，不更新

注意：probationary 架构下 false split 可被 absorb 修复、false glue 永久存在，因此 `new_speaker_threshold` 应靠近 `global_match_threshold`。

相关代码：`pipeline/clusterer.py`

### 3) probationary 状态机与吸收

- 新建 speaker 一律 probationary；`matched`/`fallback` 都会累计其匹配语音时长
- 累计 ≥ `probation_confirm_duration` → 转正（confirmed）
- 仍是 probationary 且与某 confirmed centroid 相似度 ≥ `absorb_threshold` → 吸收：centroid 按 counts 加权并入目标，记录 `redirect_map`；被吸收者的缓冲输出 flush 时以目标 id 落盘

### 4) centroid 更新

- 全程 SMA 增量更新：`alpha = 1 / (count + 1)`
- 常规更新门控：track 时长 ≥ `min_segment_duration_for_centroid_update`
- `overlap_fallback` 弱更新：仅当相似度 > `global_match_threshold + weak_update_similarity_margin` 时，以 `weak_update_weight_multiplier` 衰减权重更新

### 5) 提交区与零重写输出

- 重叠滑窗（`hop_duration < chunk_duration`）时，每窗口只提交中段 `hop_duration` 秒，两侧各留 `(chunk-hop)/2` 边界缓冲；首窗从 0 开始
- 每个 speaker 维护一个未闭合的 open turn（驻留内存）：后续帧/段间隔 ≤ `streaming_merge_gap` 即在写出前扩展拼接，跨 chunk 生效
- 沉默闭合：每 chunk 提交后检查，open turn 终点距 commit_end 已超过 `streaming_merge_gap` 即判定说话结束、提前闭合写出（与"下次开口回头闭合"产出等价，但尾段输出延迟从等到 EOF 降到约一个 hop）；EOF 时 `finalize` 兜底闭合全部残余 turn
- probationary speaker 的帧先入内存 pending 缓冲（缓冲内同样按 merge_gap 拼接）；身份定案后 `flush_speaker` 把缓冲段按时间序重放进 final_id 的 open-turn 管线，与正在进行的 turn 连续段自动拼接
- 短于 `min_segment_duration` 的 turn 在闭合时丢弃
- `finalize`：闭合全部 open turn、兜底 flush 残余 pending，纯追加，不重写文件
- 残余限制：无法向已写出的行回拼（缓冲段桥接两段旧段时，较早写出的行保持独立）；输出文件行序不再严格按时间排序（DER 评分不受影响）

相关代码：`pipeline/rttm_writer.py`

## 模块分工

### `app.py`

CLI 主入口：解析参数、加载 YAML 并合并、校验、初始化日志和 pipeline、遍历输入音频。

### `config.py`

`ChunkPipelineConfig` + YAML 加载/键名校验 + CLI 定义 + 合并。

- YAML 是全部调参项的唯一来源
- CLI 仅保留 `--wav`、`--output_dir`、`--config`、模型/环境参数（`--model_path`、`--model_type`、`--segmentation_model`、`--hf_token`、`--hf_cache_dir`、`--device`）与 `--debug`、`--verbose`、`--show_rttm`

### `schema.py`

共享数据结构：`ChunkObservation`、`SpeakerTurn`、调试 TypedDict。

### `orchestrator.py`

chunk 主循环：切窗 → segmentation → track 聚合 → embedding → 全局分配 → 提交输出 → finalize。

### `track_builder.py`

chunk 内 local track 构造：非重叠纯净区优先拼接、overlap_fallback 回退、时长门控。

### `clusterer.py`

Hungarian local->global 分配、SMA/弱更新、probationary 转正与吸收、定案事件（resolved_speakers）。

### `rttm_writer.py`

零重写 RTTM 写出：confirmed 帧即时进入 open-turn 管线（写出前拼接），probationary 帧入 pending 缓冲，定案后 flush 追加；`finalize` 纯追加收尾。

### `models/`

- `embedding_infer.py`：ERes2NetV2 加载与 segment embedding 推理
- `segmentation_infer.py`：pyannote segmentation 推理封装
- `hf_resolver.py`：Hugging Face 本地缓存解析

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
- `absorb_threshold`
- `min_segment_duration_for_new_speaker`
- `min_segment_duration_for_centroid_update`
- `weak_update_similarity_margin`
- `weak_update_weight_multiplier`
- `probation_confirm_duration`

### 4) RTTM 输出

- `min_segment_duration`
- `streaming_merge_gap`
- `show_rttm`

## 输入输出与调试

### 输入形式

`--wav` 支持：

- 单音频文件
- 音频目录
- 文本清单（每行一个路径）

支持后缀：`.wav`、`.mp3`、`.flac`

### 输出文件

每个输入音频默认输出：

- `*.streaming.rttm`
- `run.log`

### 调试建议

常用启动：

```bash
python3 pipeline.py --debug --verbose --show_rttm ...
```

重点日志字段：

- `frame_decision`（chunk 级 local->global）
- `window_summary`（chunk 级汇总，兼容旧分析工具）
- `new_speakers` / `updated_speakers` / `skipped_updates`
- `absorb_events` / `resolved_speakers` / `final_resolutions`
- `current_global_speakers`（含 probationary/confirmed 状态）

可用工具：

```bash
python3 tools/analyze_run_log.py --log ./exp/demo/run.log
```
