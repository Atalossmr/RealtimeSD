"""ASR 跟随进程的协作编排：前置契约校验 + 子进程命令组装 + 哨兵协调。

stage3（ASR 转写）以独立 follow 进程消费 stage2 产出的 segments manifest，
这里集中放这对跨进程协作关系本身的逻辑：启动前的配置契约校验、follow
子进程命令组装、就绪哨兵等待与 done 哨兵收尾。
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from common.config import ChunkPipelineConfig

logger = logging.getLogger(__name__)


def validate_follow_requirements(config: ChunkPipelineConfig) -> None:
    """校验 ASR 跟随（python -m asr.app --follow）对管线配置的前置要求。

    需要配置开启分段导出且使用 streaming 后端；
    ahc 后端不产出 segments manifest，ASR follow 进程会空转到超时。

    Args:
        config: 管线配置。

    Raises:
        SystemExit: separation_enabled 为 false，或 clustering_backend
            不是 streaming。
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
    """组装 ASR follow 子进程命令（消费 exp_dir/segments，产出 transcripts）。

    Args:
        exp_dir: 本次 run 的输出目录。
        asr_config: ASR 转写的 YAML 配置路径。
        done_file: diarization 完成哨兵路径。
        ready_file: ASR 就绪哨兵路径。
        verbose: 是否透传 --verbose。

    Returns:
        子进程命令列表。
    """

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

    Args:
        proc: follow 子进程。
        ready_file: 就绪哨兵路径。

    Raises:
        RuntimeError: 子进程在就绪前退出。
    """

    while not ready_file.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"子进程在就绪前退出（rc={proc.returncode}）")
        time.sleep(1.0)


def finish_follow_process(
    proc: subprocess.Popen, done_file: Path, timeout: float = 300.0
) -> int:
    """落 done 哨兵放行跟随进程收尾，等其排空积压后自行退出。

    正常路径行为不变：follow 循环的清空判定（哨兵出现且本轮无新段）
    保证其自行退出，这里等到它排空为止。timeout 是兜底上限：超时不
    等于截断转写的常规预期，而是防止父进程被无限阻塞——批量积压排空
    过久、或中断路径上等待期间第二次 Ctrl+C 会把 ASR 子进程遗弃为孤儿。
    超时后按 terminate → 短超时 wait → kill 升级（参照 run.py
    stop_viewer），并打 WARNING。

    Args:
        proc: follow 子进程。
        done_file: 完成哨兵路径。
        timeout: 排空等待的兜底上限（秒）。

    Returns:
        子进程退出码：非 0 表示 follow 进程在排空阶段崩溃（OOM 等）或
        被超时强杀，转写产物可能缺失/不全，调用方应据此置非零退出码。
    """

    done_file.touch()
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "ASR follow 进程排空超过 %ss 未退出，terminate 强收（transcript 可能不全）",
            timeout,
        )
    proc.terminate()
    try:
        return proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("ASR follow 进程 terminate 后仍未退出，kill 兜底")
        proc.kill()
        return proc.wait()


__all__ = [
    "validate_follow_requirements",
    "build_asr_follow_cmd",
    "wait_process_ready",
    "finish_follow_process",
]
