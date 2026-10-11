# RealtimeSD

![flowchart](docs/pipline.drawio.png)
基于 [pyannote/segmentation-3.0](<https://github.com/pyannote/pyannote-audio>) 与 [3D-Speaker/ERes2NetV2](<https://github.com/modelscope/3D-Speaker>) 的实时说话人识别（speaker diarization）管线。

## 项目内容

- 入口：`python3 run.py <音频文件|目录|清单文本>`（diarization 一键运行：音频 → RTTM；目录/清单为批量模式，模型只加载一次）、`python3 -m diarization.app`（diarization 独立 CLI）、`python3 -m diarization.extract.app`（嵌入提取阶段）、`python3 -m diarization.cluster.app`（聚类阶段）
- oracle 模式：`python3 run.py <音频> --oracle --ref <参考 RTTM 文件|目录>`（用 ref RTTM 驱动的 oracle 前端替代 segmentation 前端，不加载 segmentation 模型，用于上限实验；test_der.py 同样支持）
- 库式调用：`pipeline.process(audio="x.wav", run_name="demo")`
- 配置文件：`config/config.yaml`（diarization 管线调参项的唯一来源）
- 管线实现：`diarization/`（frontend + extract + cluster）+ `pipeline/`（组合层）。模型代码统一在 `common/models/`
- 运行脚本：`run.py`（一键运行）、`test_der.py`（DER 评估）
- DER 评估：`tools/compute_der.py`
- 运行日志分析：`tools/analyze_log.py`

## 环境要求

- Python `>= 3.13`
- 建议 Linux + CUDA 环境
- 首次运行需要可访问 Hugging Face 和 ModelScope（当使用自动下载模型时）
- FFmpeg 4–8

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
- `pyannote/segmentation-3.0`（局部说话人识别；oracle 模式不加载）

默认行为：

- 未提供 `--model_path`（或 `config/config.yaml` 里的 `model_path`）时，会自动下载并缓存默认 ERes2NetV2 到 `./pretrained/modelscope`
- segmentation 模型首次会下载并缓存到 `hf_cache_dir`（默认 `./pretrained/huggingface`）

`pyannote/segmentation-3.0` 需要 Hugging Face 授权。可通过环境变量提供 token：

```bash
export HF_TOKEN=your_token
```

## 快速开始

数据说明：音频数据不在版本库中（`datasets/` 见 `.gitignore`），需自行准备
任意 16kHz wav；模型 checkpoint 首次运行会自动下载到 `./pretrained/`，
无需手动准备。

单文件：

```bash
python3 run.py <音频.wav> \
  --config ./config/config.yaml \
  --run_name demo
```

常见覆盖参数：

```bash
python3 run.py <音频.wav> \
  --config ./config/config.yaml \
  --run_name demo \
  --model_path <本地 ckpt 路径> \
  --hf_cache_dir ./pretrained/huggingface \
  --device cuda:0 \
  --verbose
```

diarization 独立 CLI（与 `run.py` 等价，直接指定输出目录）：

```bash
python3 -m diarization.app --wav <音频.wav> --output_dir <dir> --config config/config.yaml
```

oracle 模式（ref RTTM 替代 segmentation 前端，上限实验）：

```bash
python3 run.py <音频.wav> \
  --oracle \
  --ref <参考 RTTM 文件；批量时给含 {uri}.rttm 的目录> \
  --run_name oracle
```

调试：

- `--debug`：输出调试信息
- `--show_rttm`：运行时把新写出的 RTTM 行同步打印到控制台

## 输出说明

`run.py` / `test_der.py` 的输出目录为 `{output_root}/{test_name}/{run_name}/`
（`run.py` 的 test_name 固定为 `common`，`test_der.py` 固定为 `der_test`），
按内容类型分子文件夹：

```
output_dir/
  rttm/           # {uri}.raw.rttm（raw 级 append-only，行写出即最终）
                  # {uri}.refined.rttm（修正与后处理后的最终输出，下游应消费它）
                  # {uri}.speakers.json（refined 级 sidecar）
  embeddings/     # {uri}.chunks.npz（提取阶段 python3 -m diarization.extract.app 产出）
  logs/           # run.log、command.log 等
```

每次 run 的汇总记录追加在 run 目录上一级的 `{output_root}/{test_name}/results.txt`；
`test_der.py` 的 DER 明细另写在 run 目录的 `logs/der.log` 与 `der_summary.txt`。

## 脚本使用

基础运行：

```bash
python3 run.py <wav 文件>
```

运行时同步在控制台打印 RTTM：

```bash
SHOW_RTTM=1 python3 run.py <wav 文件>
```

带 DER 评估（需提供参考 RTTM 目录）：

```bash
REF_RTTM=<参考 rttm 目录> \
RUN_NAME=baseline \
python3 test_der.py <wav 文件>
```

`test_der.py` 也接受音频目录或每行一个路径的文本清单做批量评估：全部音频的
RTTM 写入同一输出目录（模型只加载一次），最后输出逐文件 DER 与拼接全局 DER：

```bash
python3 test_der.py datasets/task1-6 --ref_rttm datasets/task1-6/rttm --run_name task16
```

脚本常用环境变量（作为对应命令行参数的缺省值）：

- `CONFIG_PATH`
- `MODEL_PATH`
- `HF_TOKEN`
- `HF_CACHE_DIR`
- `OUTPUT_ROOT`
- `RUN_NAME`
- `DEBUG`
- `SHOW_RTTM`

以下环境变量仅 `test_der.py` 使用：

- `DEVICE`（运行设备覆盖，如 cpu/cuda）
- `REF_RTTM`（参考 RTTM 目录）
- `DER_VERBOSE`
- `COLLAR`（DER 评估的宽容边界，秒，默认 0.0）

## 配置

参数在 `config/config.yaml`，全部参数的详细说明见 `config/README.md`。

## 仓库结构

分层规则：`common ← diarization ← pipeline ← run.py / test_der.py`。
common 不 import 业务包；pipeline 是唯一同时依赖多个模块的组合层
（详见 `pipeline/README.md`）。

- `diarization/`：说话人识别（frontend + extract + cluster，chunk 管线；frontend 含 ref RTTM 驱动的 oracle 前端）
- `pipeline/`：组合层（api：一次完整运行的公开入口 process（单/批量统一）；runner：diarization 组合执行；eval：DER 评估；cli：入口脚本公共 CLI 辅助）
- `common/`：公共实现（`config/` 配置体系、`contracts/` 跨包契约、`utils/` 日志与设备/音频/路径/数值/RTTM 工具、`models/` 模型封装、路径常量、`speakerlab/` vendored 模型依赖）
- `common/speakerlab/`：3D-Speaker相关依赖（vendored）与 `md-eval.pl`
- `tools`：常用工具，包括参数扫描，log分析等
- `run.py`：diarization 一键运行脚本
- `test_der.py`：DER 评估脚本

## 备注

- 当前仓库偏实验性质，默认参数更贴近中文 16kHz 流式 diarization 场景
- 更细的模块说明与调参建议见各模块目录下的README
