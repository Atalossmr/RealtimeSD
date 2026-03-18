# RealtimeSD

基于 [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) 和 [`3D-Speaker/ERes2NetV2`](https://github.com/modelscope/3D-Speaker) 的实时说话人分离管线，重点面向重叠音频处理场景，是以下流程的简单工程实现。

![流程图](docs/pipline.drawio.png)

## 主要内容

- 管线：`pipline.py`
- 配置：`config.yaml`
- 运行脚本：`run.sh`、`test_der.sh`
- 管线实现说明：`pipline/README.md`

## 环境要求

- Python `>= 3.13`
- 建议使用 Linux + CUDA 环境进行推理
- 首次运行可能需要访问 Hugging Face 和 ModelScope 下载模型

## 安装

使用 `uv`：

```bash
uv sync
```

使用 `pip`：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 模型依赖

运行时通常需要两个模型：

- `ERes2NetV2`

- `pyannote/segmentation-3.0`

默认行为：

- 如果 `--model_path` 未提供，会尝试从 ModelScope 下载预训练 `ERes2NetV2`
- segmentation-3.0 模型会在首次使用时从 Hugging Face 下载并缓存到 `./pretrained/huggingface`

Hugging Face 模型需要授权，需要环境变量或参数提供 hf token：

```bash
export HF_TOKEN=your_token
```

## 快速开始

单文件推理：

```bash
python3 pipline.py \
  --wav ./examples/example.wav \
  --output_dir ./exp/demo \
  --config ./config.yaml
```

批量推理：

```bash
python3 pipline.py \
  --wav ./examples \
  --output_dir ./exp/batch_demo \
  --config ./config.yaml
```

常用参数：

```bash
python3 pipline.py \
  --wav ./examples \
  --output_dir ./exp/batch_demo \
  --config ./config.yaml \
  --model_path ./pretrained/examples/example.ckpt \
  --hf_cache_dir ./pretrained/huggingface \
  --verbose
```

需要调试信息，可添加参数 `--debug`。
需要运行时将新生成的 RTTM 行同步输出到控制台，可添加参数 `--show_rttm`。

## 输出文件

每个音频通常会在输出目录下生成：

- `*.streaming.rttm`：流式识别结果
- `run.log`：运行日志

如果开启 `--save_segmentation_scores`，还会生成：

- `*.segmentation_scores.jsonl`

## 使用脚本

推荐先用仓库自带脚本进行实验。

直接运行：

```bash
bash run.sh ./examples
```

如果希望脚本运行时同步在控制台看到 RTTM 行：

```bash
SHOW_RTTM=1 bash run.sh ./examples
```

带 DER 评估的运行：

```bash
REF_RTTM=./datasets/rttm \
RUN_NAME=baseline \
bash test_der.sh ./examples
```

同样支持：

```bash
REF_RTTM=./datasets/rttm \
RUN_NAME=baseline \
SHOW_RTTM=1 \
bash test_der.sh ./examples
```

环境变量：

- `CONFIG_PATH`
- `MODEL_PATH`
- `HF_TOKEN`
- `HF_CACHE_DIR`
- `OUTPUT_ROOT`
- `RUN_NAME`
- `DEBUG`
- `SAVE_SEGMENTATION_SCORES`
- `SHOW_RTTM`
- `REF_RTTM`
- `DER_VERBOSE`

## 配置说明

默认配置文件是 `config.yaml`，里面主要包含：

- 运行设备和模型路径
- 左右上下文长度与实时步长
- segmentation 激活阈值
- embedding 提取片段长度
- speaker 匹配、合并与 streaming 输出参数

常用参数：
`elta_new: 0.40`

`global_match_threshold: 0.50`

`merge_threshold: 0.7`

`sma_window: 3`

`speaker_min_total_duration_to_emit: 5.00s`

## 仓库结构

- `pipline/`：实时分离主实现
- `speakerlab/`：本地依赖的 speaker encoder 相关模块与 `md-eval.pl`
- `compute_der.py`：DER 统计与批量评估
- `run.sh`：基础运行脚本
- `test_der.sh`：运行并自动评估 DER 的脚本
- `pipline/README.md`：按代码结构整理的实现与调参说明

## 备注

- 仓库当前偏实验性质，默认配置更适合中文、16kHz、流式 speaker diarization 场景
- 更细的实现说明见 `pipline/README.md`
