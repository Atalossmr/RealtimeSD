# RealtimeSD

基于 `pyannote/segmentation-3.0`(<https://github.com/pyannote/pyannote-audio>)、`3D-Speaker/ERes2NetV2`(<https://github.com/modelscope/3D-Speaker>) 与 `TIGER`(<https://github.com/JusperLee/TIGER>) 的实时说话人识别与重叠分离管线。

目前语音分离功能仍在开发中。

## 项目内容

- 入口脚本：`pipeline.py`
- 配置文件：`config.yaml`
- 管线实现：`pipeline/`
- 运行脚本：`run.sh`、`test_der.sh`
- DER 评估：`tools/compute_der.py`
- 运行日志分析：`tools/analyze_run_log.py`

## 环境要求

- Python `>= 3.13`
- 建议 Linux + CUDA 环境
- 首次运行需要可访问 Hugging Face 和 ModelScope（当使用自动下载模型时）

## 安装

使用 `uv`：

```bash
uv sync
```

或使用 `pip`：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 模型依赖

默认运行通常依赖：

- `ERes2NetV2`（说话人 embedding）
- `pyannote/segmentation-3.0`
- `JusperLee/TIGER-speech`（当 `enable_speech_separation=true`）

可选：

- 关闭说话人音轨转录（`enable_speech_separation=false`）后，可不加载 TIGER

补充说明：

- 当前仓库 `config.yaml` 默认关闭 `enable_speech_separation: false`；如需导出说话人音轨可在配置中开启

默认行为：

- 未提供 `--model_path`（或 `config.yaml` 里的 `model_path`）时，会自动下载并缓存默认 ERes2NetV2 到 `./pretrained/modelscope`
- segmentation 模型首次会下载并缓存到 `hf_cache_dir`（默认 `./pretrained/huggingface`）
- 仅当启用说话人音轨转录（`enable_speech_separation=true`）时，才会下载并缓存 TIGER 模型

`pyannote/segmentation-3.0` 需要 Hugging Face 授权。可通过环境变量提供 token：

```bash
export HF_TOKEN=your_token
```

## 快速开始

单文件：

```bash
python3 pipeline.py \
  --wav ./examples/example.wav \
  --output_dir ./exp/demo \
  --config ./config.yaml
```

目录批量：

```bash
python3 pipeline.py \
  --wav ./examples \
  --output_dir ./exp/batch_demo \
  --config ./config.yaml
```

常见覆盖参数：

```bash
python3 pipeline.py \
  --wav ./examples \
  --output_dir ./exp/batch_demo \
  --config ./config.yaml \
  --model_path ./pretrained/examples/example.ckpt \
  --hf_cache_dir ./pretrained/huggingface \
  --verbose
```

调试与可视化：

- `--debug`：输出窗口级调试信息
- `--show_rttm`：运行时把新写出的 RTTM 行同步打印到控制台

## 输出说明

每个输入音频会在 `output_dir` 下生成：

- `*.streaming.rttm`：流式 RTTM 结果
- `run.log`：运行日志

当 `enable_speech_separation=true` 时，会在 `output_dir/{uri}/` 下导出说话人音轨：

- `stable/{uri}_spk_{rttm_id}_stable.wav`
- `uncertain/{uri}_spk_internal_{speaker_id}_uncertain.wav`（仅在 `export_uncertain_speaker_audio=true`）

音轨导出特性：

- 全长与原音频时长对齐
- 非说话时段自动补静音
- 若重叠分离匹配成功，分离结果会覆盖对应时段

当 `enable_speech_separation=false` 时：

- 不进行说话人音轨转录与重叠分离，仅输出 RTTM 与日志

## 脚本使用

基础运行：

```bash
bash run.sh ./examples
```

运行时同步在控制台打印 RTTM：

```bash
SHOW_RTTM=1 bash run.sh ./examples
```

带 DER 评估：

```bash
REF_RTTM=./datasets/rttm \
RUN_NAME=baseline \
bash test_der.sh ./examples
```

同样可叠加：

```bash
REF_RTTM=./datasets/rttm \
RUN_NAME=baseline \
SHOW_RTTM=1 \
bash test_der.sh ./examples
```

脚本常用环境变量：

- `CONFIG_PATH`
- `MODEL_PATH`
- `HF_TOKEN`
- `HF_CACHE_DIR`
- `OUTPUT_ROOT`
- `RUN_NAME`
- `DEBUG`
- `SHOW_RTTM`
- `REF_RTTM`
- `DER_VERBOSE`

脚本行为说明：

- `run.sh` 每次运行会清理 `${OUTPUT_ROOT:-./exp}/common`
- `test_der.sh` 每次运行会清理 `${OUTPUT_ROOT:-./exp}/der_test`

## pipeline 行为概览

每个窗口主流程：

1. 围绕 `target_time` 裁剪固定长度上下文并运行 segmentation
2. 在 `target_time ± 0.5 * target_activity_window_duration` 内统计各 local slot 累积活跃时长
3. local slot 活跃时长达到 `target_min_duration` 才进入候选
4. observation 优先从非重叠帧选段；失败回退 `overlap_fallback`
5. 命中 `observation reuse` 时复用历史分配；否则提 embedding
6. clustering 层先做可选 speaker merge，再做 Hungarian 联合分配
7. 决策后按规则新建/匹配/回退，并更新 centroid 与 debug 统计
8. 每个窗口解析后先按帧级聚类结果写基础音轨，再进入 streaming 处理 RTTM 与 merge 补写
9. 若启用分离，对聚类后帧决策识别出的连续重叠段触发 TIGER 并覆盖说话人音轨

## 配置重点

核心参数在 `config.yaml`，分组与代码实现一一对应。常用项：

- 调度：`context_left_duration`、`context_right_duration`、`advance_step`
- 目标活动窗口：`target_activity_window_duration`
- observation：`target_min_duration`、`min_segment_duration_for_embedding`、`max_segment_duration_for_embedding`
- 匹配：`new_speaker_threshold`、`global_match_threshold`、`merge_threshold`
- 新增长度门控：
  - `min_segment_duration_for_new_speaker`
  - `min_segment_duration_for_centroid_update`
- 更新策略开关：`disable_ema_update`
- 更新稳定性：`centroid_warmup_window`、`stable_update_count_threshold`
- reuse：`disable_observation_reuse`、`reuse_overlap_threshold`、`reuse_time_horizon`
- 输出：`min_segment_duration`、`max_frame_speakers`、`streaming_flush_interval`
- 音轨/分离：`enable_speech_separation`、`min_overlap_duration_to_process`、`separation_required_duration`

说明：

- `min_segment_duration_for_embedding` 在当前 `FBank -> ERes2NetV2(TSTP)` 实现下存在有效下限；16k 场景建议不低于 `0.105s`（`1680` samples），过低可能产生 NaN embedding，进而触发 `matrix contains invalid numeric entries`
- 只有当 observation 片段时长达到 `min_segment_duration_for_new_speaker` 才允许新建 speaker
- 只有当 observation 片段时长达到 `min_segment_duration_for_centroid_update` 才允许更新簇中心
- 当 `disable_ema_update=true` 时，centroid 全程使用增量均值，不再切换 EMA

## 仓库结构

- `pipeline/`：主实现
- `speakerlab/`：本地依赖与 `md-eval.pl`
- `tools/compute_der.py`：DER 统计与批量评估
- `tools/analyze_run_log.py`：run.log 命中统计分析
- `run.sh`：基础运行脚本
- `test_der.sh`：运行 + DER 评估脚本
- `pipeline/README.md`：按模块组织的详细实现说明

## 备注

- 当前仓库偏实验性质，默认参数更贴近中文 16kHz 流式 diarization 场景
- 更细的模块说明与调参建议见 `pipeline/README.md`
