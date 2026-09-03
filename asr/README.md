# ASR 转写模块

独立于 diarization 管线的离线/跟随转写阶段：读取管线分段导出的音频段
manifest，逐段转写（Fun-ASR-Nano），落盘逐 speaker 转写结果供 viewer 展示。

两种模式（同一个增量消费循环，只是终止条件不同）：

- **一次性**（无 `--done_file`）：管线跑完后对目录单遍消费，转写当前已有
  全部段即落盘返回；
- **跟随**（`--follow --done_file <path>`）：与管线同时启动，轮询 manifest
  增量读取新段、即出即转；done 哨兵出现且积压清空后做最后一次落盘退出。

```bash
# 一次性：管线结束后整体转写（输出默认与 segments_dir 同目录，可用 --output_dir 指定）
python3 -m asr.app --segments_dir <exp_dir>/segments --config config/asr.yaml
# 跟随：与管线同时启动（run.py 默认行为，run.py 会传
# --segments_dir <exp_dir>/segments --output_dir <exp_dir>/transcripts
# --log_file <exp_dir>/logs/transcribe.log）
python3 -m asr.app --segments_dir <exp_dir>/segments --config config/asr.yaml \
    --follow --done_file <exp_dir>/.diarization_done --ready_file <path>
```

转写调参项全部在 `config/asr.yaml`（逐项说明见 `config/README.md` 的
asr 板块），CLI 仅保留运行时参数：

## CLI 参数

| 参数 | 说明 |
| --- | --- |
| `--segments_dir` | 必填；exporter 输出目录（扫描其中所有 `*.segments.jsonl`） |
| `--config` | YAML 配置文件路径，默认 `config/asr.yaml` |
| `--output_dir` | transcript 输出目录，缺省与 `--segments_dir` 相同 |
| `--follow` | 跟随模式：与管线同时启动，轮询增量转写，需配合 `--done_file` |
| `--done_file` | 跟随模式的结束哨兵：出现且积压清空后做最终落盘并退出 |
| `--ready_file` | 就绪哨兵：模型加载完成后 touch，供 run.py 等外部编排脚本等待 |
| `--log_file` | 转写日志路径，默认 `{output_dir}/transcribe.log` |
| `--verbose` | 启用 DEBUG 级日志 |

## 文件交互接口

ASR 与 diarization / viewer 之间无 IPC，全部经共享输出目录的文件交互。

### 输入

#### `segments/{uri}.segments.jsonl`（diarization 分段导出 manifest）

- append-only JSONL，每行
  `{"uri", "speaker_id", "start", "end", "path"}`；
- **时序契约**：行在其指向的 wav 落盘之后才追加，读到行即可读音频；
- ASR 按字节 offset 增量消费，末尾不完整行（writer 正在写）留到下一轮；
- `speaker_id` = diarization clusterer 的 global id，ASR 原样透传、不重编号
  （**合并前的身份**：段一写定音，后续聚类合并不回溯，合并语义由 viewer
  经 `speakers.json` sidecar 在展示层处理）。

#### `path` 指向的音频段 wav

- 单声道，采样率必须 = `asr.yaml` 的 `sample_rate`（默认 16000，须与管线
  `config.yaml` 的 `sample_rate` 一致）。采样率不匹配属确定性配置错误
  （第一次失败即注定全部失败），首次出现即 fail-fast（非零退出），
  不逐段跳过。

### 输出

#### `{output_dir}/{uri}.transcript.jsonl`（viewer 的数据源；run.py 编排时为 `<exp_dir>/transcripts/`）

- 每行 `{"uri", "speaker_id", "start", "end", "text"}`，按 `start` 排序；
- 写语义：运行期 **append-only 追加**——跟随模式每转完一段即时单行追加
  （供 viewer 实时看到逐段生长；单行单次 write 保证行完整，viewer 轮询
  读到不完整行会跳过、下轮再读）。段按 manifest 顺序处理、start 基本
  递增，append 顺序天然接近排序；进程收尾（退出/中断）的统一落盘做一次
  **全量排序重写**（临时文件 + 原子替换），保证最终文件有序；
- 空文本的段不写入；转写失败（瞬时 I/O 等）的段在收尾对账时重试一次，
  仍失败则 WARNING 列出丢失清单。

### 协调哨兵（由 run.py / 调用方管理）

- `--done_file`（通常为 `<exp_dir>/.diarization_done`）：跟随模式的结束
  哨兵——出现且本轮无新段（积压清空）后做最终落盘并退出；
- `--ready_file`：可选就绪哨兵，模型加载完成后 touch，供 run.py 等外部
  编排脚本等待（避免管线在 ASR 未就绪时空跑、段积压）。

### 日志

`transcribe.log`（默认写在输出目录；run.py 编排时经 `--log_file` 写入
`<exp_dir>/logs/`。刻意区别于管线的 `run.log`，避免两进程日志互相覆盖）。
