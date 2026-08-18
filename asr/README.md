# ASR 转写模块

独立于 diarization 管线的离线/跟随转写阶段：读取管线分段导出的音频段
manifest，逐段转写（Fun-ASR-Nano），落盘逐 speaker 转写结果供 viewer 展示。

两种模式（同一个增量消费循环，只是终止条件不同）：

- **一次性**（无 `--done_file`）：管线跑完后对目录单遍消费，转写当前已有
  全部段即落盘返回；
- **跟随**（`--follow --done_file <path>`）：与管线同时启动，轮询 manifest
  增量读取新段、即出即转；done 哨兵出现且积压清空后做最后一次落盘退出。

```bash
# 一次性：管线结束后整体转写
python3 -m asr.app --segments_dir <exp_dir> --config config/asr.yaml
# 跟随：与管线同时启动（run.py 默认行为）
python3 -m asr.app --segments_dir <exp_dir> --config config/asr.yaml \
    --follow --done_file <exp_dir>/.diarization_done --ready_file <path>
```

转写调参项全部在 `config/asr.yaml`（不在 CLI 上，字段见 `asr/config.py`
的 `AsrConfig`，逐项说明见 `config/README.md` 文末）。

## 文件交互接口

ASR 与 diarization / viewer 之间无 IPC，全部经共享输出目录的文件交互。

### 输入

#### `{uri}.segments.jsonl`（diarization 分段导出 manifest）

- append-only JSONL，每行
  `{"uri", "speaker_id", "start", "end", "path"}`；
- **时序契约**：行在其指向的 wav 落盘之后才追加，读到行即可读音频；
- ASR 按字节 offset 增量消费，末尾不完整行（writer 正在写）留到下一轮；
- `speaker_id` = diarization assigner 的 global id，ASR 原样透传、不重编号
  （**合并前的身份**：段一写定音，后续聚类合并不回溯，合并语义由 viewer
  经 `speakers.json` sidecar 在展示层处理）。

#### `path` 指向的音频段 wav

- 单声道，采样率必须 = `asr.yaml` 的 `sample_rate`（默认 16000，须与管线
  `config.yaml` 的 `sample_rate` 一致，不一致直接报错）。

### 输出

#### `{uri}.transcript.jsonl`（viewer 的数据源）

- 每行 `{"uri", "speaker_id", "start", "end", "text"}`，按 `start` 排序；
- 写语义：**整体重写**（文件小、幂等）；跟随模式每转完一段即时落盘（供
  viewer 实时看到逐段生长，而不是等整批积压转完才一次性出现），退出前
  的统一落盘是最后一次重写；
- 空文本的段不写入。

### 协调哨兵（由 run.py / 调用方管理）

- `--done_file`（通常为 `<exp_dir>/.diarization_done`）：跟随模式的结束
  哨兵——出现且本轮无新段（积压清空）后做最终落盘并退出；
- `--ready_file`：可选就绪哨兵，模型加载完成后 touch，供 run.py 等外部
  编排脚本等待（避免管线在 ASR 未就绪时空跑、段积压）。

### 日志

`transcribe.log`（写在输出目录；刻意区别于管线的 `run.log`，两进程常
共用输出目录，避免互相覆盖）。
