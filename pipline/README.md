# online_pipline_overlap

基于segmentation-3.0和EResNetV2的在线说话人分离流水线实现。

它的目标是处理下面这些 overlap 场景问题：

- 同一时刻输出多个 speaker；
- 避免两个同时活跃的 local speaker 被贴到同一个 global speaker；
- 减少 streaming RTTM 的碎片化输出；
- 提高短重叠语音进入全局匹配流程的概率。

更完整的设计说明见仓库根目录：`online_pipline_overlap_notes.md`

## 目录结构

- `app.py`
  - 应用入口编排。
- `cli.py`
  - CLI 参数与 YAML 配置融合。
- `pipeline.py`
  - 主流程串联。
- `segmentation.py`
  - target local speaker 选择与 observation 构造。
- `clustering.py`
  - global speaker 联合分配、更新与 merge。
- `streaming.py`
  - 流式 RTTM 写出。
- `schema.py`
  - 配置和数据结构定义。
- `utils.py`
  - 通用工具函数。

## 依赖项

`online_pipline_overlap` 不是一个完全独立的小包，它依赖仓库里的本地模块、外部 Python 包，以及两类模型权重。

### Python 包

- `torch`
  - 模型推理与张量计算。
- `torchaudio`
  - 音频加载、重采样，以及底层音频解码。
- `numpy`
  - 分数矩阵、时间轴和 embedding 后处理。
- `scipy`
  - 匈牙利算法分配（`linear_sum_assignment`）。
- `pyyaml`
  - 读取 YAML 配置。
- `huggingface_hub`
  - 下载和缓存 `pyannote/segmentation-3.0`。
- `modelscope`
  - 在未提供本地 ERes2NetV2 checkpoint 时，自动下载默认 speaker encoder。
- `pyannote.audio`
  - segmentation 模型加载、推理与 powerset 解码。

如果你是从仓库根目录新建环境，至少需要先安装：

```bash
pip install -r requirements.txt huggingface_hub pyannote.audio
```

### 仓库内本地依赖

当前 overlap 流水线还直接依赖仓库里的 `speakerlab` 模块，包括：

- `speakerlab.models.eres2net.ERes2NetV2`
- `speakerlab.process.processor.FBank`
- `speakerlab.utils.fileio.load_audio`

这也是为什么当前实现需要在完整的 `3D-Speaker` 仓库里运行，而不是只拷贝 `online_pipline_overlap/` 目录。

### 模型与权重

运行时需要两类模型：

- ERes2NetV2 speaker encoder checkpoint
  - 可通过 `--model_path` 指定本地权重；若不提供，则会自动尝试从 ModelScope 下载默认的 `iic/speech_eres2netv2_sv_zh-cn_16k-common`。
- `pyannote/segmentation-3.0`
  - 用于在线 segmentation；如果本地缓存不存在，会自动从 Hugging Face 下载。

默认情况下，segmentation 模型会缓存到：

- `./pretrained/huggingface`

默认情况下，自动下载的 ERes2NetV2 speaker encoder 会缓存到：

- `./pretrained/modelscope/iic/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.ckpt`

示例 speaker encoder checkpoint 路径：

- `./pretrained/custom_eres2netv2_finetune/final_model.ckpt`
- `./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt`

### Hugging Face 访问

如果 `pyannote/segmentation-3.0` 尚未缓存到本地，你通常需要：

- 能访问 Hugging Face；
- 已获得对应模型的访问权限；
- 必要时通过 `--hf_token` 提供 token。

### 输入与运行环境要求

- 输入音频支持：`.wav`、`.mp3`、`.flac`
- 流水线内部统一按 `16kHz` 工作；必要时会自动重采样
- `--output_dir` 必须可写
- `--device` 可设为 `auto`、`cpu` 或 `cuda:0`

如果你的环境里 `torchaudio` 没有可用的编解码后端，`.mp3` / `.flac` 读取可能会失败；这时优先把音频转成 `.wav` 再运行。

## 运行入口

推荐使用仓库根目录下的兼容入口脚本：

```bash
python3 online_pipline_overlap.py \
  --wav ./datasets/tingshen_6.wav \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_run \
  --config ./online_pipline_overlap_config.yaml \
  --device auto
```

等价入口为：

```bash
python3 -m online_pipline_overlap.app \
  --wav ./datasets/tingshen_6.wav \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_run \
  --config ./online_pipline_overlap_config.yaml \
  --device auto
```

## 输入形式

`--wav` 支持三种形式：

1. 单个音频文件，如 `./datasets/demo.wav`
2. 音频目录，如 `./datasets/eval_audio/`
3. 文本清单文件，每行一个音频路径

当前支持的音频后缀：

- `.wav`
- `.mp3`
- `.flac`

## 必需参数

最少需要提供：

- `--wav`
  - 输入音频、目录或清单文件。
- `--model_path`
  - ERes2NetV2 checkpoint 路径；可选，不传时自动回退到 ModelScope 默认模型。
- `--output_dir`
  - 输出目录。

推荐同时显式提供：

- `--config ./online_pipline_overlap_config.yaml`

原因是当前 `cli.py` 里的默认配置路径仍是原版 `online_pipeline_config.yaml`，如果不显式指定，可能会读到非 overlap 配置。

## 推荐跑法

### 1. 直接跑单文件

```bash
python3 online_pipline_overlap.py \
  --wav ./datasets/tingshen_6.wav \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_run \
  --config ./online_pipline_overlap_config.yaml \
  --device auto
```

### 2. 跑整个目录

```bash
python3 online_pipline_overlap.py \
  --wav ./datasets/eval_audio \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_eval \
  --config ./online_pipline_overlap_config.yaml \
  --device auto
```

### 3. 用文本清单批量跑

```bash
python3 online_pipline_overlap.py \
  --wav ./wav_list.txt \
  --model_path ./pretrained/custom_eres2netv2_finetune/final_model_v2.ckpt \
  --output_dir ./tmp/overlap_eval \
  --config ./online_pipline_overlap_config.yaml \
  --device auto
```

## 常用参数

### 上下文与调度

- `--context_left_duration`
- `--context_right_duration`
- `--step`

### target speaker 选择

- `--tau_active`
- `--target_primary_min_duration`
- `--target_overlap_min_duration`

### observation 与 embedding

- `--min_segment_duration_for_embedding`
- `--max_segment_duration_for_embedding`
- `--max_segment_shift_from_center`
- `--segment_batch_size`

### 全局 speaker 匹配

- `--delta_new`
- `--max_speakers`
- `--global_match_threshold`
- `--merge_threshold`
- `--sma_window`
- `--update_segment_overlap_threshold`

### streaming RTTM 输出

- `--min_segment_duration`
- `--max_frame_speakers`
- `--streaming_flush_interval`
- `--streaming_merge_gap`
- `--delay_short_speaker_output`
- `--speaker_min_total_duration_to_emit`

注意：

- `--min_segment_duration` 控制的是 RTTM 最短写出时长；
- `--min_segment_duration_for_embedding` 控制的是 observation 是否允许提 embedding；
- `--speaker_min_total_duration_to_emit` 只在开启 `--delay_short_speaker_output` 后生效；
- 两者作用阶段不同，不要混用。

如果开启 `--delay_short_speaker_output`，系统会先缓存某个 speaker 的 RTTM 片段；
只有当该 speaker 的累计说话时长达到 `--speaker_min_total_duration_to_emit` 后，
才会把这个 speaker 之前缓存的 RTTM 连同后续片段一起写出。

在该模式下，RTTM 中输出的 speaker 编号会按照“首次真正写入 RTTM 的顺序”重新连续编号；
运行结束时，日志里还会额外打印：

- 已达标 speaker 的 `internal speaker -> RTTM speaker` 最终映射；
- 仍未达标 speaker 的内部编号及其缓存的 RTTM 风格片段。

## 输出内容

每个输入音频默认会在 `--output_dir` 下生成：

- `*.streaming.rttm`

如果打开 `--save_segmentation_scores`，还会额外生成：

- `*.segmentation_scores.jsonl`

这个 JSONL 文件适合做窗口级诊断，里面记录了每个在线窗口的 segmentation 概率矩阵及时间信息。

## 调试模式

建议在排查 overlap 行为时加上：

```bash
--debug --verbose
```

这会输出更详细的窗口级日志，便于观察：

- target 时间附近各 local slot 的活跃时长；
- observation 是 `non_overlap` 还是 `overlap_fallback`；
- Hungarian assignment 的相似度 / cost 矩阵；
- 最终一帧到底输出了哪些 global speaker。

## 使用测试脚本

如果你想直接复用仓库里当前维护的 overlap 实验参数，推荐用根目录脚本：

```bash
bash test_single_online_overlap.sh ./datasets/tingshen_6.wav
```

这个脚本会：

- 默认使用 `online_pipline_overlap_config.yaml`；
- 覆盖成当前实验 preset；
- 记录实际命令和运行日志；
- 在提供参考 RTTM 时自动汇总 DER。

常见环境变量：

- `MODEL_PATH`
- `HF_TOKEN`
- `HF_CACHE_DIR`
- `CONFIG_PATH`
- `OUTPUT_ROOT`
- `DEBUG`
- `SAVE_SEGMENTATION_SCORES`
- `REF_RTTM`
- `REF_RTTM_DIR`

示例：

```bash
HF_TOKEN=your_token DEBUG=1 SAVE_SEGMENTATION_SCORES=1 \
bash test_single_online_overlap.sh ./datasets/tingshen_6.wav
```

## 依赖说明

运行这条流水线通常需要：

- PyTorch / torchaudio
- `pyannote/segmentation-3.0` 相关依赖
- SciPy
- 对应的 ERes2NetV2 checkpoint

如果 `segmentation_model` 需要从 Hugging Face 下载，记得准备：

- `HF_TOKEN`

并可选设置缓存目录：

- `--hf_cache_dir`

如果 `--model_path` 不提供，运行时还需要：

- 本地已安装 `modelscope` 及其依赖；
- 能访问 ModelScope；
- 默认会下载 `iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.1`。

## 当前限制

这条流水线已经能更自然地支持 overlap 输出，但仍有一些有意保留的保守设计：

- observation 优先来自非重叠区域；
- overlap fallback 不做同等级强更新；
- 目标 speaker 选择以累计活跃时长为主；
- 更适合做工程型在线增强，而不是完整研究型 overlap 联合建模。

如果你要调 overlap 效果，建议优先一起看这几组参数：

1. `tau_active` + `target_primary_min_duration` + `target_overlap_min_duration`
2. `min_segment_duration_for_embedding` + `max_segment_shift_from_center`
3. `global_match_threshold` + `merge_threshold`
4. `max_frame_speakers` + `streaming_flush_interval` + `streaming_merge_gap`
