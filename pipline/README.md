# 实时说话人识别管线

## 总体流程

处理一段音频时，主流程可以概括为：

1. 读取音频、配置和模型
2. 按 `step` 沿时间轴推进
3. 围绕当前 `target_time` 截取固定左右上下文
4. 对整个上下文运行 `pyannote/segmentation-3.0`
5. 在 `target_time` 附近的小窗内统计各个 local slot 的累计活跃时长
6. 为活跃的 local slot 挑选 observation，并提取 speaker embedding
7. 将活跃的 local slot 映射到 global speaker
8. 把决策转成 RTTM 并持续输出

主要分成三层：

- `segmentation`：回答“这一时刻附近有哪些 local speaker 在说话”
- `clustering`：回答“这些 local speaker 分别是谁”
- `streaming`：回答“怎样把这些决策写成更稳定的 RTTM 片段”

## 重叠音频处理相关内容

### 1. 目标 speaker 选择不再只看单帧

在 overlap 场景里，只靠单帧活跃度判断非常脆弱，因为第二说话人往往只在部分帧活跃，且得分容易闪烁。当前实现改成在 `target_time` 周围取一个与 `step` 对齐的小窗口，统计每个 local slot 的累计活跃时长。

核心逻辑是：

- 先用 `target_overlap_min_duration` 过滤太短的候选
- 再要求至少有一个候选达到 `target_primary_min_duration`
- 满足后才认为这一时刻真的有人说话

这样做更适合保留短促重叠语音，也更能抵抗瞬时噪声。

对应代码主要在 `segmentation.py`：

- `select_target_local_indices`
- `summarize_target_local_activity`

### 2. 同窗多个 local speaker 使用 Hungarian 联合分配

重叠场景里，一个常见错误是两个同时活跃的 local speaker 都与同一个 global centroid 相似，逐个贪心匹配时就会被错误贴到同一个人。这里的做法是：

- 对当前窗口所有 observation 与所有 global centroid 计算相似度
- 构造 `cost = 1 - similarity` 的代价矩阵
- 通过 Hungarian algorithm 一次性完成联合 assignment

这样等价于在同一个时间窗里显式加入“多个同时活跃 local 不能映射到同一个 global speaker”的约束。

相关代码在 `clustering.py`：

- `_build_assignment`
- `push_window`

### 3. observation 优先挑选非重叠音频

它在 speaker 身份维护上保持保守。每个 local slot 构造 observation 时，会优先从非重叠帧中找候选区间；如果找不到，才回退到全部活跃帧。回退得到的 observation 会被标记为 `overlap_fallback`，默认不参与正常强更新，以降低对 centroid 的污染风险。

这个设计的出发点是：先保证系统“能输出两个人”，再谨慎决定“是否让重叠片段强力参与身份更新”。

不过当前实现补了一层高置信度弱更新：当 `overlap_fallback` 片段和已匹配 global speaker 的相似度超过 `global_match_threshold + weak_update_similarity_margin` 时，仍允许用 `weak_update_weight_multiplier` 指定的衰减倍率对 centroid 做轻微更新；默认值分别是 `0.15` 和 `0.25`。

相关代码在 `segmentation.py`：

- `_non_overlap_mask`
- `_select_region_for_local`

### 4. 最终输出按时间窗聚合，而不是按单帧瞬时分数截断

原始按单帧分数截断的策略在 overlap 场景下很容易把第二说话人裁掉。当前版本会先把目标时间附近的 local 证据聚合到 global speaker 维度，再综合活跃总时长、平均分数和目标帧得分，最后才按 `max_frame_speakers` 做截断。

这比“谁当前帧最高就保留谁”的策略稳定得多。

相关代码在 `pipeline.py`：

- `_target_frame_speakers`

### 5. streaming RTTM 写出改成按 speaker 维护活跃 turn

按 speaker 分别维护 `active_turns`：

- 同一时刻允许多个 turn 并行延展
- 优先写稳定前缀，而不是每一帧立即落盘尾部
- 在 turn 结束或 `finalize()` 时再补尾段

这能明显减少 overlap RTTM 中的碎片化输出。

相关代码在 `streaming.py`。

### 6. 支持可选的可疑 speaker 延迟写出和 RTTM speaker 重映射

为了方便排查短暂误检 speaker，streaming 层支持一组可选行为：

- 开启 `delay_short_speaker_output` 后，speaker 在累计说话时长达到 `speaker_min_total_duration_to_emit` 前不会写入 RTTM
- 达到阈值后，会把之前缓存的 RTTM 片段一次性补写
- 如果整段音频结束时仍未达标，则不写入 RTTM，但会把缓存片段按 RTTM 风格打印到日志

在这个模式下，还会维护一份内部 speaker id 到 RTTM speaker id 的映射：

- 只有真正写入 RTTM 的 speaker 才会分配 RTTM id
- RTTM speaker id 按首次写出的顺序连续编号
- 运行结束后会在log里统一打印最终映射，方便对照内部聚类结果和最终 RTTM

## 模块分工

### `app.py`

最薄的一层 CLI 编排入口，负责：

- 解析参数
- 合并 YAML 配置
- 校验运行必需项
- 初始化 `output_dir/run.log` 日志文件
- 初始化管线并遍历输入音频

入口函数是 `pipline.app:main`。

### `cli.py`

负责命令行参数与 YAML 配置融合：

- 先读取 YAML
- 再用显式 CLI 参数覆盖
- 最后转换为运行时配置对象 `PipelineConfig`

当前默认配置文件路径就是仓库根目录的 `config.yaml`。

### `schema.py`

定义主流程共享的数据结构，例如：

- `PipelineConfig`
- `SegmentObservation`
- `BufferedDecisionWindow`
- `ResolvedDecisionWindow`
- `StreamingFrameDecision`

### `segmentation.py`

负责在目标时间附近筛选值得跟踪的 local speaker，并为这些 local speaker 构造 observation 与 embedding 证据。

### `clustering.py`

负责维护 global speaker centroid，并处理：

- Hungarian 联合分配
- speaker 新建
- speaker 更新
- speaker 合并
- 调试信息记录

### `streaming.py`

负责把目标时刻的离散 speaker 决策写成更稳定的 RTTM turn。

### `pipeline.py`

负责把整个处理链路真正串起来，是运行时的主逻辑编排层。

### `utils.py`

负责通用工具逻辑，例如音频路径收集、日志初始化等。

## 参数分层理解

参数分为一下几个层面。

### 1. 音频与实时调度

- `context_left_duration`：目标时刻左侧历史上下文长度
- `context_right_duration`：目标时刻右侧未来上下文长度
- `step`：推进步长，也是输出决策的基本时间粒度

这组参数决定系统“看多宽、走多快”。

### 2. segmentation 与目标 speaker 选择

- `tau_active`：local slot 判定为活跃的阈值
- `target_primary_min_duration`：至少一个 local speaker 达到该累计活跃时长，当前时刻才认为真的有人说话
- `target_overlap_min_duration`：次要或重叠 speaker 的最低累计活跃时长

这组参数决定哪些 local speaker 值得进入后续流程。

### 3. observation 构造与 embedding 提取

- `min_segment_duration_for_embedding`：允许提 embedding 的最短片段时长
- `max_segment_duration_for_embedding`：允许提 embedding 的最长片段时长
- `max_segment_shift_from_center`：候选片段中心离 `target_time` 允许偏移的最大范围
- `segment_batch_size`：批量提 embedding 的 batch size

这组参数决定提供给全局匹配器的证据质量和长度范围。

### 4. 全局 speaker 匹配与更新

- `delta_new`：更倾向新建 speaker 的阈值
- `max_speakers`：最多维护多少个 global speaker
- `global_match_threshold`：observation 贴已有 speaker 的主阈值
- `merge_threshold`：全局 speaker 自动合并阈值
- `sma_window`：centroid 前期使用简单平均更新的窗口大小
- `update_segment_overlap_threshold`：连续两次更新片段允许的最大重合度
- `weak_update_similarity_margin`：弱更新相对主匹配阈值还需额外超过的相似度裕量
- `weak_update_weight_multiplier`：overlap fallback 高置信度命中时的弱更新倍率

这组参数决定 speaker 身份如何延续、何时分裂、何时合并。

### 5. streaming RTTM 输出控制

- `min_segment_duration`：RTTM 写出允许的最短稳定片段时长
- `max_frame_speakers`：单个目标时刻最多输出多少个 speaker
- `streaming_flush_interval`：稳定前缀累计多久后真正写盘
- `streaming_merge_gap`：同一 speaker 相邻片段可自动合并的最大间隔

这组参数决定RTTM的输出。

## 输入、输出与调试

### 输入形式

`--wav` 支持三种形式：

- 单个音频文件
- 音频目录
- 文本文件，每行一个音频路径

当前支持的音频后缀包括：

- `.wav`
- `.mp3`
- `.flac`

### 常规输出

每个输入音频默认会生成一个：

- `*.streaming.rttm`
- `run.log`

输出文件 stem 与输入音频 stem 对齐。

其中 `run.log` 由 CLI 在启动时强制创建在 `output_dir` 下，不再依赖 shell 重定向。

默认情况下，常规 `INFO` / `DEBUG` 日志只写入 `run.log`，不会同步打印到控制台。

### 可选调试输出

如果开启 `--save_segmentation_scores`，还会额外生成：

- `*.segmentation_scores.jsonl`

它记录每个实时窗口的 segmentation 概率矩阵和时间信息，适合观察窗口级行为。

### 控制台同步 RTTM

如果希望在运行时一边持续写入 `*.streaming.rttm`，一边把新生成的 RTTM 行同步打印到控制台，可以开启：

```bash
python3 pipline.py --show_rttm ...
```

这个开关只影响控制台是否实时看到 RTTM 行，不影响 `output_dir/run.log` 和最终 RTTM 文件落盘。

### 调试

建议在排查问题时打开：

```bash
python3 pipline.py --debug --verbose ...
```

推荐优先关注这些日志字段：

- `target_local_activity`：目标时刻附近各 local slot 的累计活跃时长
- `observations`：每个 local 是使用 `non_overlap` 还是 `overlap_fallback`
- `assignment_cost_matrix`：local 与 global 的相似度/代价矩阵是否合理
- `local_assignments`：多个 local 是否被成功分到不同 global
- `frame_decision`：最终输出阶段是否真的保住了第二说话人

## 运行方式

仓库根目录当前实际入口是：

```bash
python3 pipline.py \
  --wav ./examples/example.wav \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_run \
  --config ./config.yaml \
  --device auto
```

运行后会固定在 `./tmp/overlap_run/run.log` 写入本次执行日志。

如果你想直接复用仓库里维护的 overlap 实验参数，也可以使用根目录脚本，例如：

```bash
bash run.sh ./examples/example.wav
```

如果希望脚本运行时同步在控制台看到 RTTM 行：

```bash
SHOW_RTTM=1 bash run.sh ./examples/example.wav
```

或者在提供参考 RTTM 时直接联动评估：

```bash
REF_RTTM=./datasets/rttm \
bash test_der.sh ./examples/example.wav
```

同样也支持：

```bash
REF_RTTM=./datasets/rttm \
SHOW_RTTM=1 \
bash test_der.sh ./examples/example.wav
```

## 代码入口速查

- CLI 入口：`../pipline.py`
- 应用编排：`app.py`
- 参数与配置：`cli.py`
- 主流程：`pipeline.py`
- observation 构造：`segmentation.py`
- 全局 speaker 分配：`clustering.py`
- RTTM 写出：`streaming.py`
- 数据结构：`schema.py`

另外，日志层面要注意一件事：

- 运行日志总是写入 `output_dir/run.log`
- `--verbose` 控制全局日志级别到 `DEBUG`
- `--debug` 控制是否输出窗口级结构化调试信息
- `--show_rttm` 控制是否把新写出的 RTTM 行同步输出到控制台
