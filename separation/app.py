"""stage2 分段音频导出 CLI（骨架：说明契约，暂不可独立运行）。

separation 的输入契约——每个 commit 区需要同时具备：

1. commit 区波形（整段波形按 commit_start/commit_end 切片）；
2. 本 chunk 的 local->global 聚类分配（决定哪些帧归哪个 global speaker）；
3. 本 chunk 观测的参照 embedding（分离音轨的归属匹配基准）。

这三者只有 stage1 的流式循环在分配完成时才齐备，因此 separation 目前只能
以 chunk hook 形式挂接在管线内（由 `python3 run.py <音频> --stage >= 2`
驱动）。独立离线运行（读 refined RTTM 重放驱动分离导出）是未来工作。
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="stage2 分段音频导出（骨架：说明输入契约，暂不可独立运行）"
    )
    parser.add_argument(
        "--wav", default=None, help="（未来）输入音频文件"
    )
    parser.add_argument(
        "--rttm", default=None, help="（未来）驱动重放的 refined RTTM"
    )
    parser.add_argument(
        "--output_dir", default=None, help="（未来）segments 输出目录"
    )
    return parser


def main() -> None:
    """打印契约说明并以非零退出（骨架，不做任何真实分离）。"""

    parser = build_parser()
    parser.parse_args()
    parser.error(
        "separation 暂不可独立运行：其输入契约（commit 区波形 + 本 chunk 聚类分配"
        " + 参照 embedding）只有在 stage1 流式循环内才齐备。请用"
        " `python3 run.py <音频> --stage 2`（或默认 --stage 3）驱动；"
        "独立离线运行（读 refined RTTM 重放）是未来工作。"
    )


if __name__ == "__main__":
    main()
