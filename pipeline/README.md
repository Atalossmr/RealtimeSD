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

- `runner.py`：`DiarizationRunner` / `run_diarization`——stage1（extract + cluster
  + RTTM）与 stage2（separation exporter 挂 chunk hook）的组合执行；
  模型只加载一次，可连续处理多条音频。
- `follow.py`：stage3 ASR 跟随进程的协作逻辑——`validate_follow_requirements`
  （ASR follow 需要 separation_enabled + streaming 后端）与
  `build_asr_follow_cmd`（follow 子进程命令组装）。
- `eval.py`：`run_der_evaluation`——对管线产出的 refined RTTM 目录计算
  逐文件 + 拼接全局 DER（封装 tools/compute_der.py 子进程调用与结果回写）。

## 消费者

- `run.py`：主入口（单文件实时管线 + viewer/ASR 进程编排）；
- `test_der.py`：DER 评估（批量复用 `DiarizationRunner` + `run_der_evaluation`）；
- `diarization/app.py`：stage1 独立 CLI（`run_diarization(with_separation=False)`）。
