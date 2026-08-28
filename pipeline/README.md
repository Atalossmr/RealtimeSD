# pipeline：组合层

把 `diarization/`（stage1）、`separation/`（stage2）、`asr/`（stage3）装配成完整
管线的代码集中在这一层。它是唯一同时依赖多个业务包的包；`common/` 是
无业务语义的共享底座，业务包之间互不依赖。

## 归属判断标准

- **归 pipeline/**：需要知道 ≥2 个模块协作语义的代码——组合执行、跨 stage
  契约校验、follow 进程编排、产物评估。
- **归 common/**：纯通用机制（日志、进程哨兵等待/收尾、CLI 合并），或被业务包
  直接 import 的底座（配置定义、数据契约、模型封装）。

## 模块

- `api.py`：一次完整运行的公开入口——`process(audio=...)` 统一支持单文件
  与批量（`audio` 传单个路径或路径列表；批量时模型只加载一次、单文件失败
  不中断整批，stage 3 批量共享一个常驻 ASR follow 进程）。覆盖输出目录、
  ASR follow 子进程与哨兵协调、diarization 执行、结果汇总；viewer 由调用方
  经 `on_asr_ready` 钩子自行编排。`resolve_run_dirs` 提供输出路径约定的
  单一口径。
- `runner.py`：`DiarizationRunner` / `run_diarization`——stage1（extract + cluster
  + RTTM）与 stage2（separation exporter 挂 chunk hook）的组合执行；
  模型只加载一次，可连续处理多条音频。
- `follow.py`：stage3 ASR 跟随进程的协作逻辑——`validate_follow_requirements`
  （ASR follow 需要 separation_enabled + streaming 后端）与
  `build_asr_follow_cmd`（follow 子进程命令组装）。
- `eval.py`：`run_der_evaluation`——对管线产出的 refined RTTM 目录计算
  逐文件 + 拼接全局 DER（封装 tools/compute_der.py 子进程调用与结果回写）。
- `run_helpers.py`：编排辅助（公共 CLI 参数、环境变量缺省、results.txt 记录头、
  输入检查、跟随子进程就绪等待/收尾）——只被 run.py / test_der.py 与本包
  api 使用，只依赖标准库。

## 消费者

- `run.py`：主入口 CLI（薄封装：argparse/env 缺省 + viewer 编排，管线本体
  走 `pipeline.process`）；
- `test_der.py`：DER 评估 CLI（批量走 `pipeline.process` +
  `run_der_evaluation`）；
- `diarization/app.py`：stage1 独立 CLI（`run_diarization(with_separation=False)`）；
- 库式调用：`pipeline.process(audio="x.wav", stage=1)` 直接嵌入他处代码
  （config 缺省经 `common.config.load_config` 无 argparse 构建）。
