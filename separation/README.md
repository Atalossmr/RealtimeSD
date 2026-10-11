# separation：离线 RTTM 驱动的分离切分工具

独立离线工具：按 RTTM 把含混叠的整条音频拆成各 speaker 的纯净音频段
（重叠区经 TIGER 分离 + embedding 归属匹配）。不在管线内，与 diarization
互不 import。

## 用法

```
python -m separation.app --wav <音频> --rttm <RTTM> --output_dir <目录> \
    [--config config/config.yaml] [--device cuda:0] [--model_path <ckpt>] [--verbose]
```

库式调用见 `separation.separate_by_rttm`（offline.py）。

## 处理流程

1. **RTTM 解析**（rttm.py）：speaker label 按字典序映射为 int speaker id
   （0 起），全时间线按 `separation_frame_step`（默认 50ms）生成逐帧活跃
   记录；
2. **参照 embedding**：每 speaker 取全时间线上仅其一人活跃的独唱帧区间
   （短于 `min_segment_duration_for_embedding` 丢弃、长于
   `max_segment_duration_for_embedding` 从起点截断），拼接后提
   ERes2NetV2 embedding 作为归属匹配参照；无独唱区间的 speaker 记 warning，
   匹配缺参照时自然回退原始音频；
3. **分离窗口**：重叠帧（>=2 人活跃）按相邻间隔 <= `segment_merge_gap`
   归并为窗口，窗口边界向两侧扩展到相邻窗口中点（首窗到 0、末窗到音频
   末尾），TIGER 输入有上下文且窗口间无缝覆盖全时间线；
4. **逐窗口分离**：solo 帧直接切原始
   波形；重叠帧经 TIGER 分离（固定 2 路、16kHz，`common/models/tiger.py`，
   仅在存在重叠窗口时惰性加载）后做身份确认（attribution.py）：
   - 能量门控：重叠帧区间 音轨RMS / 混合RMS ≥ `separation_energy_ratio`；
   - 两路都过：2x2 embedding 互斥匹配（`match_pair`），
     最小相似度 < `separation_min_match_similarity` 判不可靠；
   - 一路过：过门控音轨做 winner 匹配（`best_match`），落选方重叠帧丢弃；
   - 都不过 / 匹配不可靠 / 分离输入过短：整窗回退原始音频
     （宁可串音也不要坏轨/空轨）；
   - 重叠帧活跃 speaker > 2 时，未进候选（前 2）的 speaker 在重叠帧上
     用原始音频。
5. **段输出**：各窗口帧片按时间顺序、按 `hop_duration` 跨度分批喂
   open-segment 管线拼接闭合（assembler.py；分批保证
   `max_segment_duration` 的强制闭合检查在离线下也生效，
   `min_segment_duration` / `segment_merge_gap` 控制段闭合），默认 sink
   落盘 wav + manifest（sink.py）。

## 模块分工

```
separation/
  app.py           # 离线 CLI：python -m separation.app --wav x.wav --rttm x.rttm --output_dir out/
  offline.py       # separate_by_rttm：窗口归并、分离 dispatch、回退策略、段组装编排
  rttm.py          # load_rttm / build_frame_activities：RTTM 解析与帧级活跃生成
  attribution.py   # 身份确认纯函数：overlap_regions / cosine / energy_gate /
                   #   match_pair / best_match
  assembler.py     # SegmentAssembler：open-segment 拼接与闭合推出
  sink.py          # SegmentCallback / WavSegmentSink：wav 落盘 + manifest.jsonl
  utils.py         # 切片/能量纯函数：AudioSource / slice_absolute / rms / rms_over_regions
```

与 `common/models/` 的分工：模型加载与推理封装（TIGER、ERes2NetV2）统一在
common/models；本包只做编排与归属判定。embedder 由 CLI 独立构建
（load_embedding_model + FBank + NativeERes2NetV2SegmentEmbedder，不需要
segmentation）；共享配置取 `common.config.ChunkPipelineConfig`。

## 输出文件契约

`{output_dir}/` 下：

- `{uri}.segments.jsonl`：append-only manifest，每行
  `{"uri", "speaker_id", "start", "end", "path"}`；
  **行在其指向的 wav 落盘之后才追加**，读到行即可读音频；
- `{uri}/spk{speaker_id}_{start}_{end}.wav`：单声道、
  采样率 = `config.sample_rate`（默认 16kHz）；
- `speaker_id` 为 RTTM label 按字典序映射的 int id（0 起）；
- 运行日志在 `{output_dir}/logs/run.log`。
