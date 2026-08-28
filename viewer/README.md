# 管线产物时间线可视化

按产物分层支持管线的三个 stage——stage 越多，能看到的信息越多：

| stage | 消费的产物 | 展示内容 |
|---|---|---|
| 1 | `rttm/{uri}.refined.rttm` + `{uri}.speakers.json` | 波形 + 按说话人着色的 diarization 时间线 + speaker chip（uncertain/合并标记）+ 段列表（无文本） |
| 2 | + `segments/{uri}.segments.jsonl` + 分段 wav | + 每段"▶ 分离"按钮，回放分离后的单说话人音轨 |
| 3 | + `transcripts/{uri}.transcript.jsonl` | + ASR 转写文本（同步高亮的转写列表） |

零第三方依赖（纯标准库服务端 + 原生 JS 前端）。

**实时模式**：与管线同时启动时（`run.py` 默认行为），产物逐 chunk/逐段落盘，
页面每 2s 轮询增量刷新——时间线随管线运行逐段生长，顶栏 `● LIVE` 亮起表示
仍在运行；管线结束（`.diarization_done` 哨兵出现，stage 1/2/3 均落）后 LIVE
熄灭，结果为最终完整版。运行中能力位会随产物出现自动升档（rttm → +分离段
→ +转写），无需手动刷新。管线跑完后 `run.py` 不立即退出：打印"音频已处理完成"
并挂起等待，viewer 保持可访问；用户 Ctrl+C 后脚本才退出并关闭 viewer。之后想
回看结果，按下方命令手动重启即可。

## 服务器模式（推荐）

```bash
python3 viewer/server.py --exp_root exp --audio_root datasets --port 8000
# 打开 http://127.0.0.1:8000
```

- 自动扫描 `exp/**` 下的 refined RTTM / segments manifest / transcript（任一
  存在即成会话），在 `datasets/**` 下按 `{uri}.wav` 匹配原始音频；
- `--exp_root` / `--audio_root` 可重复指定多个目录；
- 音频不在扫描目录时：`--audio URI=/path/to/audio.wav` 显式指定；
- 音频支持 HTTP Range，长会议音频 seek 无压力。

CLI 参数（viewer 无 YAML 配置，全部选项都在 CLI 上）：

| 参数 | 说明 |
|---|---|
| `--exp_root` | 产物扫描根目录，可重复，默认 `exp` |
| `--audio_root` | 原始音频扫描根目录，可重复，默认 `datasets` |
| `--audio URI=PATH` | 显式指定某 uri 的音频路径，可重复 |
| `--port` | 监听端口，默认 `8000` |

## 数据接口（消费的文件）

viewer 只读文件、不写；与管线/ASR 的交互全部经以下产物：

| 文件 | 生产者 | 字段/语义 |
|---|---|---|
| `rttm/{uri}.refined.rttm` | diarization refined 级 | 标准 RTTM 行（speaker 列为 global id），merge 修正后整体原子重写 |
| `rttm/{uri}.speakers.json` | diarization refined 级 | 见下；不存在（旧产物/静态模式）时按无标记展示 |
| `segments/{uri}.segments.jsonl` | separation（`separation.sink`） | 每行 `{"uri","speaker_id","start","end","path"}`，append-only；wav 在同名子目录 |
| `transcripts/{uri}.transcript.jsonl` | ASR（`asr.app`） | 每行 `{"uri","speaker_id","start","end","text"}`，按 start 排序、整体重写；`speaker_id` 为合并前的 global id |
| `{uri}.wav` | 原始音频 | 在 `--audio_root` 下按文件名（不含扩展名）= uri 递归匹配 |
| `.diarization_done` | pipeline（`api.process`） | 哨兵：不存在 = 管线仍在运行（LIVE，持续轮询）；stage 1/2/3 结束均落 |

路径前缀为新布局（`rttm/`、`transcripts/`、`segments/` 子目录）；viewer 扫描
与索引均为递归匹配，旧平铺布局的 exp 目录同样兼容。

`{uri}.speakers.json` 结构（字段定义见 `diarization/README.md` 的文件接口节）：

```json
{
  "uri": "...", "final": false, "post_merge_min_count": 8,
  "speakers": [{"id": 0, "duration": 123.4, "count": 12,
                "uncertain": false, "merged_into": null}],
  "merge_events": [{"absorbed": 9, "survivor": 8, "kind": "merge"}]
}
```

前端用法：`speakers[].id` 与各数据源的 `speaker_id` 同源（global id），
沿 `merged_into` 链解析出展示用有效 id——被并说话人的段按幸存者颜色展示、
标签显示 `spkA→spkB`，顶栏保留删除线 chip 提示合并关系，重叠判定同样按
有效 id；`uncertain` + 顶栏 "uncertain 标记" 开关控制不确定标记（虚线 chip、
标签加 `?` 且转写里显示为灰色、时间线段淡化），开关仅在 `post_merge_min_count > 0`
时出现，说话人样本计数累积达标后标记自动解除。

HTTP 端点（服务器模式）：

- `GET /api/sessions`：会话列表（`uri` / `audio_url` / `live` / 能力位
  `has_rttm` `has_segments` `has_transcript` 及各产物 URL），每次请求重扫目录；
- `GET /api/rttm/{uri}`：refined RTTM 段数组（`speaker_id`/`start`/`end`，无文本）；
- `GET /api/segments/{uri}`：manifest 段数组（`speaker_id`/`start`/`end`/`file`，
  file 为 basename）；
- `GET /api/segment_audio/{uri}/{file}`：分段 wav 音频流（Range/206；只接受
  basename，拒绝路径穿越）；
- `GET /api/transcript/{uri}`：转写段数组（`speaker_id`/`start`/`end`/`text`）；
- `GET /api/speakers/{uri}`：speakers.json 内容，无 sidecar 返回 `null`；
- `GET /api/audio/{uri}`：原始音频流（支持 Range / 206）；
- `POST /api/shutdown`：停止服务器（响应送达后延时触发 shutdown；run.py
  的等待循环检测到 server 退出后随之收尾）。

前端每 2s 轮询 sessions + 当前数据源 + speakers：段结构数据源按能力位选择
（segments manifest > refined RTTM，始终跟随 diarization 节奏），transcript
不作为段结构源——它是文本叠加层，按键 `spk{id}_{start}` 贴到段上、独立轮询，
转写出一段补一段（无 manifest 的会话才兜底用 transcript 当段结构源）；
能力位升档（rttm → segments）时保留页面状态增量叠加，不整页重载；
transcript 兜底源按"旧段是新段前缀"增量合并，RTTM/manifest 源（会被 merge
重写历史行）用 JSON 串比对、变化即整体重算；speakers.json 同理整体重算
（合并映射、uncertain、重叠标记）。

## 静态模式（无服务器）

直接用浏览器打开 `viewer/static/index.html`，手动选择音频文件和
`*.transcript.jsonl` 或 `*.refined.rttm` 文件即可（静态模式无实时刷新，
不支持分离段播放）。

## 页面操作

- 时间线色块 = 一个段，颜色 = 说话人；hover 看详情，点击 seek 原音频；
- 段列表行的 "▶ 分离" 按钮回放该段的分离音频（stage>=2，独立播放器，
  不打断主音频位置）；
- 滚轮缩放（以鼠标为锚点）、拖动平移、"适配全段"重置；
- 顶栏说话人 chip 点击可隐藏/显示该说话人（时间线与列表同步过滤）；
- 播放时播放头自动跟随，段列表同步高亮当前段。
