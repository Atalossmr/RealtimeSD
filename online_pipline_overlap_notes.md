# `online_pipline_overlap/` 设计与实现说明

## 1. 这条流水线解决什么问题

`online_pipline_overlap/` 是当前仓库里专门面向重叠说话场景的在线说话人分离版本。

它不是一套完全重写的新系统，而是在原有在线流水线基础上，优先解决下面几类 overlap 场景里最常见的问题：

- 第二说话人明明在说话，但最后输出阶段被截掉；
- 同一个目标时刻里两个 local speaker 被错误贴到同一个 global speaker；
- streaming RTTM 输出过碎，文件里充满短小抖动片段；
- 短促重叠语音还没进入全局匹配流程，就在 observation 构造阶段被丢掉。

这条链路的核心目标可以概括为三点：

1. 保留原有在线处理框架，避免整体复杂度失控；
2. 让系统能稳定输出同一时刻的多个 speaker；
3. 在工程上尽量保持参数可理解、日志可解释、行为可调试。

## 2. 总体处理流程

处理一段音频时，主流程如下：

1. 读取音频、配置、模型；
2. 按 `step` 沿时间轴推进；
3. 围绕当前 `target_time` 切出固定左右上下文；
4. 对整个上下文跑 `pyannote/segmentation-3.0`；
5. 在 `target_time` 附近的小时间窗里统计每个 local slot 的活跃总时长；
6. 只对这些真正持续活跃的 local slot 构造 observation；
7. 用 speaker embedding 将 observation 映射到 global speaker；
8. 汇总当前目标时刻的 global speaker 证据；
9. 生成逐帧 streaming 决策并持续写出 RTTM。

你可以把它理解成三层：

- `segmentation` 负责回答“这一时刻附近有哪些 local speaker 在说话”；
- `clustering` 负责回答“这些 local speaker 分别是谁”；
- `streaming` 负责回答“最终怎样把这些帧级决策写成稳定的 RTTM 片段”。

## 3. 与原在线版本相比，overlap 版的关键变化

### 3.1 目标 speaker 选择不再只看单帧

单帧判定在 overlap 场景里非常脆弱，因为第二说话人常常只在一部分帧上活跃、且得分不够稳定。

overlap 版本改成：

- 围绕 `target_time` 取一个与 `step` 对齐的小窗口；
- 统计每个 local slot 在窗口内累计活跃了多久；
- 先用 `target_overlap_min_duration` 过滤掉太短的候选；
- 再要求至少有一个候选达到 `target_primary_min_duration`，才认定这一时刻真的有人说话。

这样做的结果是：

- 对短促重叠说话更友好；
- 对瞬时噪声和闪烁激活更稳；
- 比“只看一帧谁高”更接近在线输出真正需要的时间粒度。

相关代码：`online_pipline_overlap/segmentation.py`

- `select_target_local_indices`
- `summarize_target_local_activity`

### 3.2 同窗多个 local speaker 用 Hungarian 联合分配

overlap 场景里，一个经典错误是：

- 同一个目标时刻里有两个 local speaker；
- 两者都和某个已有 centroid 比较相似；
- 如果逐个贪心匹配，就可能都被贴到同一个 global speaker。

当前版本会：

- 为当前窗口内所有 observation 和所有 global centroid 计算相似度；
- 构造 `cost = 1 - similarity` 的 cost matrix；
- 用 Hungarian algorithm 做一次联合 assignment。

这相当于在单个窗口内显式加入“不能把两个同时活跃 local 分给同一个 global speaker”的约束。

相关代码：`online_pipline_overlap/clustering.py`

- `_build_assignment`
- `push_window`

### 3.3 observation 构造仍然保留“非重叠优先”

虽然这是 overlap 版本，但在 speaker 身份维护上仍然相对保守。

每个 local slot 构造 observation 时，当前策略是：

- 先尝试只在非重叠帧里找候选区间；
- 若找不到，再退回到全部活跃帧；
- fallback 得到的片段会标记为 `overlap_fallback`；
- 这类 observation 默认不参与正常强更新，避免污染 centroid。

这个设计背后的考虑是：

- overlap 检测和 speaker 身份建模不是同一个问题；
- 系统首先要保证“能输出两个人”；
- 然后才是更激进地让重叠区域参与身份更新。

相关代码：`online_pipline_overlap/segmentation.py`

- `_non_overlap_mask`
- `_select_region_for_local`

### 3.4 最终输出按时间窗聚合，而不是按单帧瞬时分数截断

原始单帧截断方式在 overlap 场景里很容易把第二说话人裁掉。

当前版本会：

- 对目标时间附近的 local 证据先做聚合；
- 再映射到 global speaker 维度；
- 综合活跃总时长、平均分数和目标帧得分；
- 最后再按 `max_frame_speakers` 做输出截断。

这样比“谁这一帧分数最高就留下谁”更稳。

相关代码：`online_pipline_overlap/pipeline.py`

- `_target_frame_speakers`

### 3.5 streaming RTTM 写出改成按 speaker 维护活跃 turn

overlap 版本不再只维护一个待写段，而是：

- 按 speaker 分别维护 `active_turns`；
- 同一时刻允许多个 turn 并行延展；
- 优先写“稳定前缀”，而不是每一帧都把尾巴立刻落盘；
- 在 turn 结束或最终 `finalize()` 时再补尾段。

这会显著减少 overlap RTTM 里的碎片。

相关代码：`online_pipline_overlap/streaming.py`

### 3.6 可选的“短 speaker 延迟写出”与 RTTM speaker 重映射

为方便排查短暂误检 speaker，streaming 写出层新增了一组可选行为：

- 开启 `delay_short_speaker_output` 后，某个 speaker 在累计说话时长达到
  `speaker_min_total_duration_to_emit` 之前，先不写入 RTTM；
- 达到阈值后，会把该 speaker 之前缓存的 RTTM 片段一次性补写出来；
- 若整段音频结束时该 speaker 仍未达标，则不会写入 RTTM，但会把它的缓存片段按 RTTM 风格打印到日志里。

同时，在这个模式下还会维护一份“内部 speaker id -> RTTM speaker id”的映射：

- 只有真正写入 RTTM 的 speaker 才会分配 RTTM id；
- RTTM speaker id 按首次写出的顺序连续编号；
- 运行结束时会把最终映射表统一打印到日志，方便把内部聚类结果与最终 RTTM 对齐。

## 4. 模块分工

### 4.1 `app.py`

最薄的一层 CLI 入口，负责：

- 解析参数；
- 合并 YAML 配置；
- 校验运行必需项；
- 初始化流水线并遍历输入音频。

入口函数：`online_pipline_overlap.app:main`

### 4.2 `cli.py`

负责命令行参数和 YAML 配置融合：

- 先读 YAML；
- 再用显式 CLI 参数覆盖；
- 再转换成 `PipelineConfig`。

或者直接使用仓库里的：

- `test_single_online_overlap.sh`

### 4.3 `schema.py`

定义配置和主流程里共享的数据结构，例如：

- `PipelineConfig`
- `SegmentObservation`
- `BufferedDecisionWindow`
- `ResolvedDecisionWindow`
- `StreamingFrameDecision`

### 4.4 `segmentation.py`

负责两件事：

- 在目标时间附近选择值得跟踪的 local speaker；
- 为这些 local speaker 构造 observation 并提 embedding。

### 4.5 `clustering.py`

负责：

- 维护 global speaker centroid；
- 执行 Hungarian 联合分配；
- 控制 speaker 新建、更新和合并；
- 记录调试信息。

### 4.6 `streaming.py`

负责把目标时刻的离散 speaker 决策写成较稳定的 RTTM turn。

### 4.7 `pipeline.py`

负责把整个流程串起来，是实际的主逻辑编排层。

## 5. 参数分层理解

为了避免把不同阶段的参数混在一起，推荐按下面几类理解。

### 5.1 音频与在线调度

- `context_left_duration`
  - 目标时刻左侧历史上下文长度。
- `context_right_duration`
  - 目标时刻右侧未来上下文长度。
- `step`
  - 在线推进步长，也是输出决策的基本时间粒度。

这组参数决定了系统“看多宽、走多快”。

### 5.2 segmentation 与目标 speaker 选择

- `tau_active`
  - local slot 判定为活跃的阈值。
- `target_primary_min_duration`
  - 至少有一个 local speaker 达到这个累计活跃时长，当前时刻才认为真的有人说话。
- `target_overlap_min_duration`
  - 重叠/次要 speaker 的最低累计活跃时长。

这组参数决定了“哪些 local speaker 值得进入后续流程”。

### 5.3 observation 构造与 embedding 提取

- `min_segment_duration_for_embedding`
  - 允许提 embedding 的最短片段时长。
- `max_segment_duration_for_embedding`
  - 允许提 embedding 的最长片段时长。
- `max_segment_shift_from_center`
  - 候选片段中心离 `target_time` 最多能偏多远。
- `segment_batch_size`
  - 批量提 embedding 的 batch size。

这组参数决定了“给全局匹配器提供什么样的证据”。

### 5.4 全局 speaker 匹配与更新

- `delta_new`
  - 更倾向新建 speaker 的阈值。
- `max_speakers`
  - 最多维护多少个 global speaker。
- `global_match_threshold`
  - observation 贴已有 speaker 的主阈值。
- `merge_threshold`
  - 全局 speaker 自动合并阈值。
- `sma_window`
  - centroid 前期用简单平均更新的窗口大小。
- `update_segment_overlap_threshold`
  - 连续两次更新片段允许的最大重合度。

这组参数决定了“speaker 身份如何延续、如何分裂、何时合并”。

### 5.5 streaming RTTM 输出控制

- `min_segment_duration`
  - RTTM 写出允许的最短稳定片段时长。
  - 只影响最终输出，不参与 observation 选择。
- `max_frame_speakers`
  - 单个目标时刻最终最多输出多少个 speaker。
- `streaming_flush_interval`
  - 稳定前缀至少累计多久才真正写盘。
- `streaming_merge_gap`
  - 同一 speaker 相邻片段允许自动合并的最大间隔。

这里最容易误解的是 `min_segment_duration`：

- 它不控制 embedding 片段筛选；
- 它控制的是 RTTM 落盘时，过短片段是否直接丢弃；
- 真正控制 observation 最短长度的是 `min_segment_duration_for_embedding`。

## 6. 当前 overlap 配置的工程取向

目前仓库里的 `online_pipline_overlap_config.yaml` 已经同步为 `test_single_online_overlap.sh` 里的实验参数，整体取向是：

- `tau_active=0.50`
  - 降低激活门槛，提升第二说话人召回；
- `target_primary_min_duration=0.15`
  - 主说话人要有最低持续性，避免短噪声误触发；
- `target_overlap_min_duration=0.08`
  - 保留短重叠说话；
- `min_segment_duration_for_embedding=0.30`
  - 让更短的 overlap 片段也能进入全局匹配；
- `global_match_threshold=0.40`
  - 保持匹配足够开放，但仍避免过度误贴；
- `merge_threshold=0.70`
  - overlap 版本里对 speaker merge 更保守；
- `max_frame_speakers=3`
  - 当前实验脚本允许单时刻最多输出 3 个 speaker；
- `streaming_flush_interval=20.0`
  - 明显偏向减少 RTTM 碎片，而不是追求很低写出延迟。

这不是唯一可行方案，但比较贴合当前脚本的实验目的：优先观察 overlap 召回和稳定输出效果。

## 7. 输入、输出与落盘文件

### 7.1 支持的输入形式

`--wav` 支持三种输入：

- 单个音频文件；
- 音频目录；
- 文本文件，文件中每行一个音频路径。

支持的音频后缀由 `collect_audio_paths` 决定，目前包括：

- `.wav`
- `.mp3`
- `.flac`

### 7.2 常规输出

每个输入音频默认会生成一个：

- `*.streaming.rttm`

文件名 stem 与输入音频 stem 对齐。

### 7.3 可选调试输出

如果打开 `--save_segmentation_scores`，还会额外生成：

- `*.segmentation_scores.jsonl`

它记录每个窗口的完整 segmentation 概率矩阵及其时间信息，适合分析窗口级行为。

## 8. 调试时建议重点看什么

开启：

```bash
--debug --verbose
```

建议优先关注这些信息：

- `target_local_activity`
  - 看目标时刻附近每个 local slot 的累计活跃时长；
- `observations`
  - 看每个 local 用的是 `non_overlap` 还是 `overlap_fallback`；
- `assignment_cost_matrix`
  - 看 local 与 global 的相似度矩阵是否合理；
- `local_assignments`
  - 看多个 local 是否被分到不同 global；
- `frame_decision`
  - 看最终输出阶段是否真的保住了第二说话人。

如果你遇到“明明有重叠但没输出第二人”的情况，通常按下面顺序排查最有效：

1. `target_local_activity` 是否已经把第二 local 过滤掉；
2. `build_observations` 是否没有为该 local 构造出合法片段；
3. Hungarian assignment 是否把它映射到了异常的 global id；
4. 最终 `max_frame_speakers` 截断是否又把它裁掉。

## 9. 推荐使用方式

### 9.1 直接跑 overlap 入口

```bash
python3 online_pipline_overlap.py \
  --wav ./datasets/tingshen_6.wav \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_run \
  --config ./online_pipline_overlap_config.yaml \
  --device auto
```

### 9.2 用仓库自带脚本跑单文件实验

```bash
bash test_single_online_overlap.sh ./datasets/tingshen_6.wav
```

这个脚本适合做 overlap 参数实验，因为它会：

- 固定一套 overlap 参数 preset；
- 记录实际运行命令；
- 汇总 RTTM 生成结果；
- 在给出参考 RTTM 时顺手计算 DER。

## 10. 当前版本的边界

这条 overlap 流水线已经比普通在线版本更适合重叠输出，但它仍然不是完整的 overlap 联合建模系统。

当前仍然比较保守的点包括：

- observation 构造依然优先非重叠区域；
- overlap fallback 证据不会像干净片段那样强更新 centroid；
- 目标 speaker 选择仍主要依赖累计活跃时长，没有引入更复杂的置信度模型；
- 输出侧虽然支持多 speaker，但本质上仍是在线近似决策，而不是全局最优离线解。

因此更准确的定位是：

- 它是一个对 overlap 场景做了专门增强的在线 diarization 流水线；
- 而不是一个彻底为 overlap end-to-end 重建的研究型系统。

## 11. 代码入口速查

- CLI 入口：`online_pipline_overlap.py`
- 应用编排：`online_pipline_overlap/app.py`
- 参数与配置：`online_pipline_overlap/cli.py`
- 主流程：`online_pipline_overlap/pipeline.py`
- observation 构造：`online_pipline_overlap/segmentation.py`
- 全局 speaker 分配：`online_pipline_overlap/clustering.py`
- RTTM 写出：`online_pipline_overlap/streaming.py`
- 数据结构：`online_pipline_overlap/schema.py`

如果你后续要继续调 overlap 效果，最值得优先联动观察的通常是这几组参数：

1. `tau_active` + `target_primary_min_duration` + `target_overlap_min_duration`
2. `min_segment_duration_for_embedding` + `max_segment_shift_from_center`
3. `global_match_threshold` + `merge_threshold`
4. `max_frame_speakers` + `streaming_flush_interval` + `streaming_merge_gap`
