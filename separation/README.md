# separation：分段音频导出（stage2）

chunk 管线的第二阶段：在 diarization（stage1，extract + cluster）完成
local->global 分配后，把每个 commit 区的音频按 global speaker 拆成分段音频，
供流式 ASR 消费。仅 streaming 聚类后端生效（ahc 离线后端不产出）。

## 处理流程

每个 commit 区（挂在 chunk_hook 上，由 run.py 的 `run_diarization` 接线）：

1. **重叠判断**：用 seg_scores 硬标签逐帧统计活跃 global speaker
   （帧裁剪与 RTTM writer 共用 `common/contracts/frames.py` 的单份实现），
   ≥2 人活跃即重叠帧；
2. **无重叠**：各 speaker 活跃帧直接从原始波形切片输出；
3. **TIGER 分离**：有重叠帧时分离整个 commit 区（固定 2 路、16kHz，
   模型封装在 `common/models/tiger.py`，惰性加载）；
4. **身份确认**（attribution.py）：重叠帧区间能量门控（音轨 RMS / 混合 RMS
   ≥ `separation_energy_ratio`）后做归属匹配（参照为本 chunk 观测 embedding）：
   - 两路都过：2x2 embedding 互斥匹配（`match_pair`）；
   - 一路过：过门控音轨做 winner 匹配（`best_match`），落选方重叠帧丢弃；
   - 都不过 / 最小相似度 < `separation_min_match_similarity`：整窗回退纯切片
     （宁可串音也不要坏轨/空轨进 ASR）。
5. **段输出**：帧片经 open-segment 管线跨 commit 区拼接与闭合
   （assembler.py，与 RTTM writer 的 open-turn 同构），由 SegmentCallback
   推出；默认 sink 落盘 wav + manifest（sink.py）。

## 模块分工

```
separation/
  exporter.py      # StreamingSegmentExporter：编排（重叠检测、dispatch、回退策略、段闭合调度）
  attribution.py   # 身份确认纯函数：overlap_regions / cosine / reference_embeddings /
                   #   energy_gate / match_pair / best_match
  assembler.py     # SegmentAssembler：open-segment 跨 commit 区拼接与闭合推出
  sink.py          # SegmentCallback / WavSegmentSink：wav 落盘 + manifest.jsonl
  utils.py         # 切片/能量纯函数：AudioSource / slice_absolute / rms / rms_over_regions
```

与 `common/models/` 的分工：模型加载与推理封装（TIGER、ERes2NetV2、
pyannote）统一在 common/models；本包只做管线编排与归属判定，模型实例
（separator、embedder）由外部构造后传入。

本包与 diarization 互不 import；共享契约（ChunkArtifacts、
iter_commit_frames、ChunkPipelineConfig）统一从 `common.contracts` /
`common.config` 取。

## 输出文件契约

`{output_dir}/segments/` 下（仅 `separation_enabled: true`）：

- `segments/{uri}.segments.jsonl`：append-only manifest，每行
  `{"uri", "speaker_id"(global id), "start", "end", "path"}`；
  **行在其指向的 wav 落盘之后才追加**，读到行即可读音频；
- `segments/{uri}/spk{speaker_id}_{start}_{end}.wav`：单声道、
  采样率 = `config.sample_rate`（默认 16kHz）；
- `speaker_id` 为**合并前**的 global id（段一写定音，后续合并不回溯）。
