#!/usr/bin/env python3
"""嵌入提取阶段入口：音频 -> <stem>.chunks.npz（不做聚类与 RTTM 输出）。"""

from diarization.extract.app import main


if __name__ == "__main__":
    main()
