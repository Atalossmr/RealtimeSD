# ASR 结果时间线可视化

展示管线产物 `{uri}.transcript.jsonl`（`asr/app.py` 落盘）与原始整段音频：
波形 + 按说话人着色的时间线 + 同步高亮的转写列表。零第三方依赖。

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
`*.transcript.jsonl` 文件即可。

## 页面操作

- 时间线色块 = 一个 ASR 段，颜色 = 说话人；hover 看文本，点击 seek；
- 滚轮缩放（以鼠标为锚点）、拖动平移、"适配全段"重置；
- 顶栏说话人 chip 点击可隐藏/显示该说话人（时间线与列表同步过滤）；
- 播放时播放头自动跟随，转写列表同步高亮当前段。
