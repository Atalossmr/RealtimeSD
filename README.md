# RealtimeSD

基于 `pyannote/segmentation-3.0`(<https://github.com/pyannote/pyannote-audio>) 与 `3D-Speaker/ERes2NetV2`(<https://github.com/modelscope/3D-Speaker>) 的实时说话人分离（speaker diarization）管线。

架构：segmentation-3.0 在 10s chunk 内做局部说话人识别，ERes2NetV2 + 增量聚类（Hungarian 分配 + SMA centroid + probationary 机制）实现全局 speaker ID 一致。无 merge、无 RTTM 重写，输出 append-only。

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

默认运行依赖：

- `ERes2NetV2`（说话人 embedding）
- `pyannote/segmentation-3.0`（局部说话人分割）

默认行为：

- 未提供 `--model_path`（或 `config.yaml` 里的 `model_path`）时，会自动下载并缓存默认 ERes2NetV2 到 `./pretrained/modelscope`
- segmentation 模型首次会下载并缓存到 `hf_cache_dir`（默认 `./pretrained/huggingface`）

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

- `--debug`：输出 chunk 级调试信息
- `--show_rttm`：运行时把新写出的 RTTM 行同步打印到控制台

说明：全部调参项（阈值、调度、更新策略、输出控制等）只通过 `config.yaml` 配置；CLI 仅保留运行时输入（`--wav`/`--output_dir`/`--config`）、模型/环境参数（`--model_path`/`--model_type`/`--segmentation_model`/`--hf_token`/`--hf_cache_dir`/`--device`）与 `--debug`/`--verbose`/`--show_rttm` 开关。

## 输出说明

每个输入音频会在 `output_dir` 下生成：

- `*.streaming.rttm`：流式 RTTM 结果（chunk 提交即最终，音频结束时按 probationary 吸收产生的 redirect 做一次终局 remap）
- `run.log`：运行日志

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

每个 chunk（10s 窗口，`hop_duration` 推进）主流程：

1. 切出 10s 窗口并运行 segmentation-3.0，得到帧级多标签分数（局部 ≤3 人）
2. 每个 local slot 聚合纯净（非重叠）语音区提 ERes2NetV2 embedding，不足时回退 `overlap_fallback`
3. clustering 层用 Hungarian 做 local->global 联合分配，按阈值判定 `matched/new/fallback`
4. 新建 speaker 进入 probationary 试用期：累计匹配语音达到 `probation_confirm_duration` 转正；试用期内与 confirmed speaker 相似度 ≥ `absorb_threshold` 则被吸收
5. 只提交窗口中段 `hop_duration` 秒的帧级结果（边界缓冲），append 到 RTTM
6. 音频结束：probationary 收尾 + 按 redirect 终局 remap，整文件重写一次

## 配置重点

核心参数在 `config.yaml`，分组与代码实现一一对应。常用项：

- 调度：`chunk_duration`、`hop_duration`
- track 构造：`min_local_activity_duration`、`min_segment_duration_for_embedding`、`max_segment_duration_for_embedding`
- 匹配：`new_speaker_threshold`、`global_match_threshold`、`absorb_threshold`
- 新增长度门控：
  - `min_segment_duration_for_new_speaker`
  - `min_segment_duration_for_centroid_update`
- 更新策略：centroid 全程使用 SMA 增量更新，overlap_fallback 弱更新按 `weak_update_weight_multiplier` 衰减
- 试用期：`probation_confirm_duration`
- 输出：`min_segment_duration`、`streaming_merge_gap`

说明：

- `min_segment_duration_for_embedding` 在当前 `FBank -> ERes2NetV2(TSTP)` 实现下存在有效下限；16k 场景建议不低于 `0.105s`（`1680` samples），过低可能产生 NaN embedding，进而触发 `matrix contains invalid numeric entries`
- probationary 架构下 false split 可被 absorb 修复、false glue 永久存在，因此 `new_speaker_threshold` 应靠近 `global_match_threshold`，宁可多建簇也不要黏合
- 只有当 track 时长达到 `min_segment_duration_for_new_speaker` 才允许新建 speaker
- 只有当 track 时长达到 `min_segment_duration_for_centroid_update` 才允许更新簇中心

## 仓库结构

- `pipeline/`：主实现（入口 `pipeline.py`）
- `speakerlab/`：本地依赖与 `md-eval.pl`
- `tools/compute_der.py`：DER 统计与批量评估
- `tools/analyze_run_log.py`：run.log 命中统计分析
- `run.sh`：基础运行脚本
- `test_der.sh`：运行 + DER 评估脚本
- `pipeline/README.md`：按模块组织的详细实现说明

## 备注

- 当前仓库偏实验性质，默认参数更贴近中文 16kHz 流式 diarization 场景
- 更细的模块说明与调参建议见 `pipeline/README.md`
