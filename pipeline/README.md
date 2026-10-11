# pipeline：组合层

把 `diarization/` 装配成完整管线（音频 → RTTM）的代码集中在这一层。
分层规则：

```
common ← diarization ← pipeline ← run.py / test_der.py
```

- `common/` 是无业务语义的共享底座，不 import 任何业务包；
- 共享契约（contracts / config / models / utils）统一从 `common` 取；
- pipeline 是唯一同时依赖多个模块的层。

例外：`diarization/app.py`（独立 CLI）复用 `pipeline.run_diarization`，
构成入口脚本级的 diarization → pipeline 反向依赖——app 是 CLI 不是库代码，
diarization 的库代码（extract/cluster/utils）不依赖本包。

## 归属判断标准

- **归 pipeline/**：需要知道 ≥2 个模块协作语义的代码——组合执行、产物评估、
  入口脚本的公共 CLI 辅助。
- **归 common/**：纯通用机制（日志、设备/音频/路径/数值工具、YAML/CLI 合并），
  或被业务包直接 import 的底座（配置定义、数据契约、模型封装）。

## 模块

- `api.py`：一次完整运行的公开入口——`process(audio=...)` 统一支持单文件
  与批量（`audio` 传单个路径或路径列表；批量时模型只加载一次、单文件失败
  不中断整批）。覆盖输出目录、diarization 执行、结果汇总；
  `resolve_run_dirs` 提供输出路径约定的单一口径。
- `runner.py`：`DiarizationRunner` / `run_diarization`——extract + cluster
  + RTTM 的组合执行；模型只加载一次，可连续处理多条音频。oracle 模式
  （`--oracle --ref <参考 RTTM 文件|目录>`）用
  `diarization.frontend.OracleFrontend` 替代 segmentation 前端（不加载
  segmentation 模型），与 `diarization.extract.oracle.oracle_chunk_artifacts`
  组合后走同一聚类/输出路径，用于上限实验。
- `eval.py`：`run_der_evaluation`——对管线产出的 refined RTTM 目录计算
  逐文件 + 拼接全局 DER（封装 tools/compute_der.py 子进程调用与结果回写，
  路径按仓库根锚定，与 cwd 无关）。
- `cli.py`：入口脚本共用的 CLI 辅助（公共参数、环境变量缺省、启动前输入
  检查）——只被 run.py / test_der.py 使用，只依赖标准库。

## 消费者

- `run.py`：主入口 CLI（薄封装：argparse/env 缺省，管线本体走
  `pipeline.process`）；
- `test_der.py`：DER 评估 CLI（批量走 `pipeline.process` +
  `run_der_evaluation`）；
- `diarization/app.py`：独立 CLI（`run_diarization`）；
- 库式调用：`pipeline.process(audio="x.wav")` 直接嵌入他处代码
  （config 缺省经 `common.config.load_config_and_runtime` 无 argparse 构建）。
