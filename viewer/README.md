# ASR 结果时间线可视化

展示管线产物 `{uri}.transcript.jsonl`（`asr/app.py` 落盘）与原始整段音频：
波形 + 按说话人着色的时间线 + 同步高亮的转写列表。零第三方依赖。

**实时模式**：与管线同时启动时（`run.py` 默认行为），ASR follow 进程每轮
把新转写结果即时落盘，页面每 2s 轮询增量刷新——时间线随管线运行逐段
生长，顶栏 `● LIVE` 亮起表示仍在运行；管线结束（`.diarization_done` 哨兵
出现）后 LIVE 熄灭，结果为最终完整版。

## 服务器模式（推荐）

```bash
python3 viewer/server.py --exp_root exp --audio_root datasets --port 8000
# 打开 http://127.0.0.1:8000
```

- 自动扫描 `exp/**/*.transcript.jsonl`，在 `datasets/**` 下按 `{uri}.wav` 匹配原始音频；
- `--exp_root` / `--audio_root` 可重复指定多个目录；
- 音频不在扫描目录时：`--audio URI=/path/to/audio.wav` 显式指定；
- 音频支持 HTTP Range，长会议音频 seek 无压力。

## 静态模式（无服务器）

直接用浏览器打开 `viewer/static/index.html`，手动选择音频文件和
`*.transcript.jsonl` 文件即可（静态模式无实时刷新）。

## 页面操作

- 时间线色块 = 一个 ASR 段，颜色 = 说话人；hover 看文本，点击 seek；
- 滚轮缩放（以鼠标为锚点）、拖动平移、"适配全段"重置；
- 顶栏说话人 chip 点击可隐藏/显示该说话人（时间线与列表同步过滤）；
- 播放时播放头自动跟随，转写列表同步高亮当前段。

## refined 级 speaker 状态（uncertain / 合并事件）

管线 streaming 后端的 refined 级每次刷新会同步落 `{uri}.speakers.json`
sidecar（与 `{uri}.refined.rttm` 同目录、原子更新），viewer 轮询读取：

- **合并事件**：被并说话人的段按幸存者颜色展示，标签显示 `spkA→spkB`，
  顶栏保留删除线 chip 提示合并关系；重叠判定也按合并后的有效 id；
- **uncertain 标记**：sidecar 开启小样本阈值（`post_merge_min_speech_duration > 0`）
  时顶栏出现 "uncertain 标记" 开关；开启后发声时长未达标的说话人 chip 变
  虚线框、标签加 `?`、时间线段淡化。说话人时长累积达标后标记自动解除，
  被合并后按幸存者展示；
- 无 sidecar（ahc 后端 / 旧产物 / 静态模式）时一切按无标记展示。
