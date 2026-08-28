"""ASR 跟随进程的协作编排：前置契约校验 + 子进程命令组装 + 哨兵协调。

stage3（ASR 转写）以独立 follow 进程消费 stage2 产出的 segments manifest，
这里集中放这对跨进程协作关系本身的逻辑：启动前的配置契约校验、follow
子进程命令组装、就绪哨兵等待与 done 哨兵收尾。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from common.config import ChunkPipelineConfig


def validate_follow_requirements(config: ChunkPipelineConfig) -> None:
    """校验 ASR 跟随（python -m asr.app --follow）对管线配置的前置要求。

    需要配置开启分段导出且使用 streaming 后端；
    ahc 后端不产出 segments manifest，ASR follow 进程会空转到超时。
    """

    if not config.separation_enabled:
        raise SystemExit("ERROR: ASR 跟随需要配置中 separation_enabled 为 true")
    if config.clustering_backend != "streaming":
        raise SystemExit(
            "ERROR: ASR 跟随需要配置中 clustering_backend 为 streaming"
            f"（当前为 {config.clustering_backend!r}，该后端不产出分段 manifest）"
        )


def build_asr_follow_cmd(
    exp_dir: Path,
    asr_config: str,
    done_file: Path,
    ready_file: Path,
    verbose: bool = False,
) -> list[str]:
    """组装 ASR follow 子进程命令（消费 exp_dir/segments，产出 transcripts）。"""

    cmd = [
        sys.executable,
        "-m",
        "asr.app",
        "--segments_dir",
        str(exp_dir / "segments"),
        "--output_dir",
        str(exp_dir / "transcripts"),
        "--log_file",
        str(exp_dir / "logs" / "transcribe.log"),
        "--config",
        str(asr_config),
        "--follow",
        "--done_file",
        str(done_file),
        "--ready_file",
        str(ready_file),
    ]
    if verbose:
        cmd.append("--verbose")
    return cmd


def wait_process_ready(proc: subprocess.Popen, ready_file: Path) -> None:
    """轮询等子进程的就绪哨兵；进程早退则报错退出。

    无超时上限：模型下载/加载耗时不可预估，卡住时由用户 Ctrl+C 中断。
    """

    while not ready_file.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"子进程在就绪前退出（rc={proc.returncode}）")
        time.sleep(1.0)


def finish_follow_process(proc: subprocess.Popen, done_file: Path) -> None:
    """落 done 哨兵放行跟随进程收尾，等其排空积压后自行退出。

    无超时上限：follow 循环的清空判定（哨兵出现且本轮无新段）保证其
    自行退出；批量场景排空耗时可能远超管线运行时长，设上限等于截断转写。
    """

    done_file.touch()
    proc.wait()


__all__ = [
    "validate_follow_requirements",
    "build_asr_follow_cmd",
    "wait_process_ready",
    "finish_follow_process",
]
