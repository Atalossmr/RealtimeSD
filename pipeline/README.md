# 实时说话人识别管线

本文档按当前代码实现说明 pipeline 的结构、窗口级流程与关键参数。

## 总体流程

处理一段音频时，核心链路如下：

1. 读取音频并重采样到 `config.sample_rate`（默认 16k）
2. 按 `advance_step` 沿时间轴推进，得到窗口推进时刻
3. 每个窗口围绕 `target_time` 截取固定上下文（`context_left_duration + context_right_duration`）
4. 对整个上下文运行 `pyannote/segmentation-3.0`
5. 在 `target_time ± 0.5 * target_activity_window_duration` 内统计 local slot 累积活跃时长
6. 仅保留活跃时长 `>= target_min_duration` 的 local slot
7. 为每个 local slot 生成候选 observation：优先非重叠区，必要时回退到 `overlap_fallback`
8. 对候选 observation 批量提取 embedding
9. clustering 先执行 speaker merge，再通过 Hungarian 做 local->global 联合分配
10. 根据阈值决定 `matched/new/fallback`，并更新 centroid
11. 每个窗口在完成 clustering 后先按帧级决策写说话人音轨，再进入 streaming 写 RTTM turn
12. 可选：当聚类后帧决策出现多说话人时，将其视作重叠并触发 TIGER 分离覆盖写入

链路分层：

- `segmentation`：目标 local 选择 + observation 构造
- `clustering`：local->global 分配 + centroid 维护 + merge 事件
- `streaming`：RTTM 持续写出 + merge 补写
- `separation`：重叠段补齐/分离/匹配/覆盖写入

## 重叠场景关键策略

### 1) 目标 speaker 选择使用多帧累计活跃时长

不再只看单帧；当前实现用 `target_activity_window_duration` 窗口累计活跃时长：

- 窗口：`target_time ± 0.5 * target_activity_window_duration`
- 规则：累计活跃时长 `>= target_min_duration` 才进入后续

相关代码：`pipeline/segmentation/selector.py`

### 2) 同窗联合分配（Hungarian）

同一个目标窗口内，多个 local observation 与所有 global centroid 一起分配：

- 代价矩阵：`cost = 1 - similarity`
- 约束效果：同窗不同 local 不会贴到同一个 global

相关代码：`pipeline/clustering/clusterer.py`

### 3) observation 先非重叠后回退

为降低重叠污染：

- 优先在当前 local 的非重叠帧中选 observation
- 若找不到，回退到全部活跃帧并标记 `selection_mode=overlap_fallback`
- `overlap_fallback` 默认不做常规强更新，仅在高置信条件下允许弱更新

相关代码：`pipeline/segmentation/observation_builder.py`

### 4) 长度门控

当前 clustering 新增两个时长阈值：

- `min_segment_duration_for_new_speaker`
  - observation 片段长度达到该阈值，才允许新建 speaker
- `min_segment_duration_for_centroid_update`
  - observation 片段长度达到该阈值，才允许更新 centroid（常规更新与弱更新都受限）

这两个门控用于抑制超短片段造成的新建抖动与中心漂移。

相关代码：`pipeline/clustering/clusterer.py`

### 5) centroid 更新策略

centroid 全程使用简单移动平均（SMA）增量更新：

- 常规更新：步长 `alpha = 1 / (count + 1)`
- 弱更新：在常规步长基础上再乘 `weak_update_weight_multiplier` 进行衰减

### 6) stable 判定与 merge 约束

stable 判定基于更新次数：

- 更新次数 `>= stable_update_count_threshold` 判定为 stable
- merge 只允许 unstable speaker 作为 small 侧被合并

若 `delay_short_speaker_output=true`：

- 未稳定 speaker 的 RTTM 先缓存
- 稳定后再补写

相关代码：

- `pipeline/clustering/clusterer.py`
- `pipeline/streaming/writer.py`

### 7) merge 后 RTTM 与音轨一致性

merge 事件由 clusterer 产出，上层同步到 streaming/audio：

- streaming 仅在安全时机补写 RTTM
- 新写 RTTM 会裁掉与既有 RTTM 的重叠区间
- 音轨侧将 small speaker 片段并入 large，避免导出丢语音

相关代码：

- `pipeline/orchestrator/merge_ops.py`
- `pipeline/streaming/writer.py`
- `pipeline/audio/speaker_buffer.py`

### 8) 说话人音轨转录与可选 TIGER 重叠分离

`enable_speech_separation=true` 时：

- 按帧级聚类结果持续写入基础说话人音轨
- 重叠判定基于聚类后的帧级 speaker 决策（同一帧 `>=2` 个 speaker）
- 连续重叠时长达到 `min_overlap_duration_to_process` 才触发
- 输入不足 `separation_required_duration` 时自动补齐
- 长重叠按 `max_overlap_process_interval` 分段处理
- 分离后的 embedding 与当前活跃 global speaker 做 Hungarian 匹配
- 匹配成功后覆盖写入对应 speaker 音轨

`enable_speech_separation=false` 时：

- 不执行说话人音轨转录与重叠分离，仅输出 RTTM

相关代码：`pipeline/separation/overlap_processor.py`

## 模块分工

### `app.py`

CLI 主入口：

- 解析参数
- 加载 YAML 配置并与 CLI 合并
- 参数校验
- 初始化日志和 pipeline
- 遍历输入音频

入口：`pipeline.app:main`

### `cli.py`

参数定义与配置构建：

- `build_arg_parser`：定义 CLI 参数（仅运行时输入、模型/环境参数与 `debug/verbose/show_rttm` 开关）
- `merge_args_with_config`：加载 YAML（调参项的唯一来源）并与 CLI 合并，校验 YAML 键名合法性
- `config_from_args`：构建 `PipelineConfig`

### `schema.py`

定义共享数据结构：

- `PipelineConfig`
- `SegmentCandidate` / `SegmentObservation`
- `BufferedDecisionWindow` / `ResolvedDecisionWindow`
- `StreamingFrameDecision`

### `segmentation/observation_builder.py`

observation 构造器：

- 候选片段选择（先非重叠，后回退）
- 长度/位置约束
- batch embedding 提取

### `segmentation/selector.py`

目标时刻活动统计逻辑：

- 目标窗口掩码
- local 累计活跃时长
- 活动摘要（供 debug）

### `clustering/clusterer.py`

全局 speaker 维护核心：

- merge（含 stable 约束）
- Hungarian 联合分配
- `matched/new/fallback` 决策
- centroid 更新（SMA + 弱更新）
- 窗口级 debug 信息构建

### `streaming/writer.py`

流式 RTTM 输出器：

- 活跃 turn 管理
- 稳定前缀刷盘
- merge 场景补写与重定向
- RTTM 区间去重写入

### `streaming/merge_commit.py`

RTTM 区间去重工具：

- `subtract_overlaps`
- `register_written_interval`

### `orchestrator/diarization.py`

主编排器：

- 串接 segmentation / clustering / streaming / separation
- 管理窗口推进、即时提交、flush 与日志
- 处理 merge 事件与帧级音轨写入

### `orchestrator/window_ops.py`

窗口/目标帧工具：

- 固定长度窗口裁剪
- 目标帧索引
- 目标帧 speaker 聚合

### `orchestrator/merge_ops.py`

merge 事件消费：

- 从 clusterer 取 merge 事件
- 同步到 speaker audio buffer
- 通知 streaming 执行 merge 逻辑

### `orchestrator/rewrite_ops.py`

把帧级 `StreamingFrameDecision` 写入 speaker buffer 的基础音轨。

### `models/`

- `embedding_infer.py`：ERes2NetV2 加载与 segment embedding 推理
- `segmentation_infer.py`：pyannote segmentation 推理封装
- `separator_infer.py`：TIGER 分离推理封装
- `hf_resolver.py`：Hugging Face 本地缓存解析

### `audio/speaker_buffer.py`

说话人音轨缓存与导出：

- 片段 append（基础填充 + 覆盖）
- merge 音频合并
- 按 stable/uncertain 分组导出

## `chunk/` 子包（新架构，与滑窗版并行）

chunk 版管线：segmentation-3.0 在 10s chunk 内做局部识别，ERes2NetV2 + 增量聚类做全局 speaker 对齐。入口 `chunk_pipeline.py`，配置 `config_chunk.yaml`（YAML 唯一调参来源，CLI 保留清单与滑窗版一致）。

与滑窗版的核心差异：

- 无 speaker merge、无 RTTM 重写、无 stable/延迟输出机制
- 新建 speaker 进入 probationary 试用期：累计匹配语音达到 `probation_confirm_duration` 转正；试用期内与某 confirmed speaker 相似度 ≥ `absorb_threshold` 则被吸收（只影响后续 chunk 与终局 remap）
- RTTM append-only；音频结束时按 redirect_map 做一次终局 remap 并整文件重写
- 首版为非重叠 10s chunk（hop == chunk_duration）

模块分工：

- `chunk/config.py`：`ChunkPipelineConfig` + YAML 加载/校验 + CLI
- `chunk/track_builder.py`：chunk 内 local track 聚合（非重叠纯净区优先，overlap_fallback 回退）
- `chunk/clusterer.py`：Hungarian local->global 分配、SMA/弱更新、probationary 转正与吸收
- `chunk/rttm_writer.py`：append-only RTTM 写出 + 终局 remap
- `chunk/orchestrator.py`：chunk 主循环
- `chunk/app.py`：CLI 入口编排

复用（不改动）：`pipeline/models/` 推理封装、`pipeline/utils.py` 工具函数、`tools/compute_der.py` / `tools/analyze_run_log.py`。

## 参数分组与作用

以下调参项全部通过 `config.yaml` 配置（YAML 是唯一来源）；CLI 仅保留 `--wav`、`--output_dir`、`--config`、模型/环境参数（`--model_path`、`--model_type`、`--segmentation_model`、`--separation_model`、`--hf_token`、`--hf_cache_dir`、`--device`）与 `--debug`、`--verbose`、`--show_rttm` 开关。

### 1) 实时调度与目标帧

- `context_left_duration`
- `context_right_duration`
- `advance_step`
- `target_activity_window_duration`
- `target_min_duration`

### 2) observation 构造与 embedding

- `min_segment_duration_for_embedding`
- `max_segment_duration_for_embedding`
- `max_segment_shift_from_center`
- `segment_batch_size`

实现备注（当前仓库）：

- 在 `FBank -> ERes2NetV2(TSTP)` 路径下，过短片段可能导致池化前时间维降到 1，`torch.var` 产生 NaN
- 16k 场景下建议 `min_segment_duration_for_embedding >= 0.105s`（`1680` samples）
- 低于该阈值时，NaN embedding 可能进一步导致 Hungarian 分配报错：`matrix contains invalid numeric entries`

### 3) 全局匹配与更新

- `max_speakers`
- `new_speaker_threshold`
- `global_match_threshold`
- `merge_threshold`
- `min_segment_duration_for_new_speaker`
- `min_segment_duration_for_centroid_update`
- `stable_update_count_threshold`
- `update_segment_overlap_threshold`
- `weak_update_similarity_margin`
- `weak_update_weight_multiplier`

### 4) streaming RTTM 输出

- `min_segment_duration`
- `max_frame_speakers`
- `streaming_flush_interval`
- `streaming_merge_gap`
- `delay_short_speaker_output`
- `show_rttm`

### 5) 音轨转录、重叠分离与导出

- `enable_speech_separation`
- `separation_model`
- `min_overlap_duration_to_process`
- `separation_required_duration`
- `max_overlap_process_interval`
- `export_uncertain_speaker_audio`
- `speaker_audio_sample_rate`
- `speaker_audio_format`

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

- `target_local_activity`
- `observations`
- `local_assignments`
- `updated_speakers`
- `skipped_updates`
- `frame_decision`

说明：

- `assignment_cost_matrix` 当前在 clusterer 内部会写入 `debug_info`，但默认日志输出未展开该字段；如需排查 Hungarian 代价矩阵，建议按需在编排层增加显式打印。

可用工具：

```bash
python3 tools/analyze_run_log.py --log ./exp/demo/run.log
```

## 代码入口速查

- CLI 入口：`../pipeline.py`
- 应用编排：`app.py`
- 参数与配置：`cli.py`
- 主流程：`orchestrator/diarization.py`
- observation 构造：`segmentation/observation_builder.py`
- 全局分配：`clustering/clusterer.py`
- RTTM 写出：`streaming/writer.py`
- 数据结构：`schema.py`
