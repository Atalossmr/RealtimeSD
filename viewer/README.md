# ASR 结果时间线可视化

展示管线产物 `{uri}.transcript.jsonl`（`asr/app.py` 落盘）与原始整段音频：
波形 + 按说话人着色的时间线 + 同步高亮的转写列表。零第三方依赖。

**实时模式**：与管线同时启动时（`run.py` 默认行为），ASR follow 进程每轮
把新转写结果即时落盘，页面每 2s 轮询增量刷新——时间线随管线运行逐段
生长，顶栏 `● LIVE` 亮起表示仍在运行；管线结束（`.diarization_done` 哨兵
出现）后 LIVE 熄灭，结果为最终完整版。管线跑完后 `run.py` 不立即退出：
打印"音频已处理完成"并挂起等待，viewer 保持可访问；用户 Ctrl+C 后
脚本才退出并关闭 viewer。之后想回看结果，按下方命令手动重启即可。

## 服务器模式（推荐）

```bash
python3 viewer/server.py --exp_root exp --audio_root datasets --port 8000
# 打开 http://127.0.0.1:8000
```

- 自动扫描 `exp/**/*.transcript.jsonl`，在 `datasets/**` 下按 `{uri}.wav` 匹配原始音频；
- `--exp_root` / `--audio_root` 可重复指定多个目录；
- 音频不在扫描目录时：`--audio URI=/path/to/audio.wav` 显式指定；
- 音频支持 HTTP Range，长会议音频 seek 无压力。

## 数据接口（消费的文件）

viewer 只读文件、不写；与管线/ASR 的交互全部经以下产物：

| 文件 | 生产者 | 字段/语义 |
|---|---|---|
| `{uri}.transcript.jsonl` | ASR（`asr.app`） | 每行 `{"uri","speaker_id","start","end","text"}`，按 start 排序、整体重写；`speaker_id` 为合并前的 global id |
| `{uri}.speakers.json` | diarization refined 级 | 见下；不存在（ahc 后端/旧产物/静态模式）时按无标记展示 |
| `{uri}.wav` | 原始音频 | 在 `--audio_root` 下按文件名（不含扩展名）= uri 递归匹配 |
| `.diarization_done` | run.py | 哨兵：不存在 = 管线仍在运行（LIVE，持续轮询） |

`{uri}.speakers.json` 结构（字段定义见 `diarization/README.md` 的文件接口节）：

```json
{
  "final": false, "post_merge_min_speech_duration": 30.0,
  "speakers": [{"id": 0, "output_id": 1, "duration": 123.4,
                "uncertain": false, "merged_into": null}],
  "merge_events": [{"absorbed": 9, "survivor": 8, "kind": "merge"}]
}
```

前端用法：`speakers[].id` 与 transcript 的 `speaker_id` 同源（global id），
沿 `merged_into` 链解析出展示用有效 id——被并说话人的段按幸存者颜色展示、
标签显示 `spkA→spkB`，顶栏保留删除线 chip 提示合并关系，重叠判定同样按
有效 id；`uncertain` + 顶栏 "uncertain 标记" 开关控制不确定标记（虚线 chip、
标签加 `?` 且转写里显示为灰色、时间线段淡化），开关仅在 `post_merge_min_speech_duration > 0`
时出现，说话人时长累积达标后标记自动解除。

HTTP 端点（服务器模式）：

- `GET /api/sessions`：会话列表（`uri` / `transcript_url` / `audio_url` / `live`），每次请求重扫目录；
- `GET /api/transcript/{uri}`：转写段数组（`speaker_id`/`start`/`end`/`text`）；
- `GET /api/speakers/{uri}`：speakers.json 内容，无 sidecar 返回 `null`；
- `GET /api/audio/{uri}`：音频流（支持 Range / 206）；
- `POST /api/shutdown`：停止服务器（响应送达后延时触发 shutdown；run.py
  的等待循环检测到 server 退出后随之收尾）。

前端每 2s 轮询 sessions + transcript + speakers：transcript 按"旧段是新段
前缀"增量合并；speakers.json 用 JSON 串比对，变化即整体重算（合并映射、
uncertain、重叠标记）。

## 静态模式（无服务器）

直接用浏览器打开 `viewer/static/index.html`，手动选择音频文件和
`*.transcript.jsonl` 文件即可（静态模式无实时刷新）。

## 页面操作

- 时间线色块 = 一个 ASR 段，颜色 = 说话人；hover 看文本，点击 seek；
- 滚轮缩放（以鼠标为锚点）、拖动平移、"适配全段"重置；
- 顶栏说话人 chip 点击可隐藏/显示该说话人（时间线与列表同步过滤）；
- 播放时播放头自动跟随，转写列表同步高亮当前段。
