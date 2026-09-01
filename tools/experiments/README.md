# tools/experiments — 架构上限实验

对当前 diarization 架构做 "oracle 分解实验"：逐级用真值（参考 RTTM）替换流水线中的
环节，把总 DER 分解到 分割层 / 嵌入判别力 / 聚类决策 三层，确认不换组件时的上限。

实验设计：总 DER = 数据集固有下限 + 分割层误差 + 嵌入可分性误差 + 聚类/流式决策误差。
各实验对应关系：

| 实验 | 脚本 | 内容 |
|---|---|---|
| E1 | `dataset_stats.py` | 数据集固有下限统计（重叠占比、>=3 人重叠、短 turn） |
| E2 | `oracle_tracks.py` | 真值给 local slot 打标签、跳过聚类 → 分割层上限 |
| E3 | `oracle_segments.py` | 真值分割 + 真实 embedding：extract / analyze / cluster 三个子命令 |
| E4c | `ahc_oracle_k.py` | AHC 固定 n_clusters=真值人数 → 簇数选不准的损失 |
| E4ab/E5 | （复用 `tools/sweep_params.py`） | streaming/ahc 阈值网格扫描 → 调参天花板 |
| E6 | `separated_recluster.py` | 分离段重提嵌入 + AHC 重聚类 → 验证"嵌入噪声源于混合短段串扰" |
| E7 | `padding_diff.py` | 两套提取缓存（不同 segment_batch_size）的 embedding 差分 → padding 泄漏量化 |

## 依赖与产物

- 除 E1 外都依赖提取阶段缓存 `*.chunks.npz`：
  `.venv/bin/python -m diarization.extract.app --wav datasets/task1-6 --output_dir exp/experiments/task1-6_extract --device cuda`
- DER 一律经 `tools/compute_der.py`（md-eval.pl，collar=0、计入重叠）。
- 实验产物约定放 `exp/experiments/`（gitignore 内）。

## 用法

```bash
# E1 数据集统计（无需缓存/GPU）
.venv/bin/python tools/experiments/dataset_stats.py --ref datasets/task1-6/rttm \
    --output exp/experiments/e1_dataset_stats.txt

# E2 分割层上限
.venv/bin/python tools/experiments/oracle_tracks.py \
    --input exp/experiments/task1-6_extract/embeddings \
    --ref datasets/task1-6/rttm \
    --output-dir exp/experiments/e2_oracle_tracks/sys_rttm \
    --report exp/experiments/e2_oracle_tracks/report.txt

# E3 嵌入+聚类上限（extract 需 GPU，analyze/cluster 纯 CPU）
.venv/bin/python tools/experiments/oracle_segments.py extract \
    --audio datasets/task1-6 --ref datasets/task1-6/rttm \
    --work-dir exp/experiments/e3_oracle_segments --device cuda
.venv/bin/python tools/experiments/oracle_segments.py analyze \
    --ref datasets/task1-6/rttm --work-dir exp/experiments/e3_oracle_segments
.venv/bin/python tools/experiments/oracle_segments.py cluster \
    --ref datasets/task1-6/rttm --work-dir exp/experiments/e3_oracle_segments

# E4c AHC + oracle 人数
.venv/bin/python tools/experiments/ahc_oracle_k.py \
    --input exp/experiments/task1-6_extract/embeddings \
    --ref datasets/task1-6/rttm \
    --output-dir exp/experiments/e4c_ahc_oracle_k/sys_rttm \
    --report exp/experiments/e4c_ahc_oracle_k/report.txt

# E4ab / E5 网格扫描（复用现有 sweep 工具）
.venv/bin/python tools/sweep_params.py \
    --input exp/experiments/task1-6_extract/embeddings --ref datasets/task1-6/rttm \
    --set global_match_threshold=... --output exp/experiments/e5_sweep_streaming.csv

# E6 分离段重聚类（需先以 --stage 2 跑出 segments/，提嵌入需 GPU）
.venv/bin/python tools/experiments/separated_recluster.py \
    --segments exp/der_test/<run>/segments --ref datasets/task1-6/rttm \
    --mixture-input exp/experiments/task1-6_extract/embeddings \
    --output exp/experiments/e6_separated_recluster --device cuda

# E7 padding 泄漏差分（先以不同 segment_batch_size 提两套缓存，需 GPU）
.venv/bin/python -m diarization.extract.app --wav datasets/task1-6 \
    --output_dir exp/experiments/e7_padding_leak/bs1 \
    --config exp/experiments/e7_padding_leak/config_bs1.yaml --device cuda
.venv/bin/python tools/experiments/padding_diff.py \
    --base exp/experiments/e7_padding_leak/bs1/embeddings \
    --other bs4=exp/experiments/e7_padding_leak/bs4/embeddings \
    --rttm datasets/task1-6/rttm \
    --output exp/experiments/e7_padding_leak/padding_diff.txt
```

## 实现要点

- `oracle_tracks.py` / `ahc_oracle_k.py` 镜像 writer 的帧判定（`score > 0`、commit 区
  裁剪，见 `common/contracts/frames.py`）重放 `AppendOnlyRTTMWriter`，保证与管线
  输出同口径；缺失 local（无活跃帧/无 embedding）不进 mapping，writer 会丢帧。
- `oracle_segments.py extract` 的 4s 截断只作用于喂 embedding 的音频，npz 中保存
  完整 turn 区间供 cluster 步写 RTTM（截断区间会直接凭空贡献 MS，已踩过坑）。
- sys RTTM 的 raw 文件须放子目录：`compute_der.py` 按 `endswith(".rttm")` 收集，
  `*.raw.rttm` 会被误匹配。
- E6 的段时间线由 stage-1/2 导出决策固定，阈值扫描里只有 SER 变化（MS/FA 为
  常数）；其 DER 绝对值与管线 refined RTTM 的 DER 时间线不同，不可直接对比。
  另注意分离段是整段 embedding（无 4s 截断），与混合短段对照并非严格单变量。
