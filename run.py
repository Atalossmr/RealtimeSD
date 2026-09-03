#!/usr/bin/env python3
"""实时管线一键运行 CLI（薄封装：编排本体在 `pipeline.process`）。

`--stage N` 为累积式：1=只 diarization（RTTM）；2=+separation（分段音频）；
3=+ASR 转写（默认，ASR 先就绪再开跑）。

本脚本只负责：argparse 与环境变量缺省、CLI/YAML 配置合并、viewer 子进程的
启动（经 process 的 on_asr_ready 钩子，在 ASR 就绪后启动）/挂起等待/收尾。
管线本体（输出目录、ASR follow 进程、哨兵协调、结果汇总）见 pipeline.api。

环境变量作为对应命令行参数的缺省值：

    STAGE           → --stage         CONFIG_PATH  → --config
    MODEL_PATH      → --model_path    HF_TOKEN     → --hf_token
    HF_CACHE_DIR    → --hf_cache_dir  DEBUG=1      → --debug
    SHOW_RTTM=1     → --show_rttm     OUTPUT_ROOT  → --output_root
    RUN_NAME        → --run_name      VIEWER_PORT  → --viewer_port
    ASR_CONFIG_PATH → --asr_config    WAIT_VIEWER=0 → --no-wait
    WITH_VIEWER=0   → --no-viewer

用法：

    python3 run.py <音频文件|目录|清单文本> [--config config/config.yaml] [--run_name default]
    # 目录/清单为批量模式：模型只加载一次，逐文件处理，viewer 展示全部 uri
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pipeline
from common.config import config_from_args, merge_args_with_config
from common.utils import collect_audio_paths
from pipeline.cli import add_common_args, check_input_paths, env_flag

BASE_DIR = Path(__file__).resolve().parent

_TEST_NAME = "common"  # 输出固定在 {output_root}/common/{run_name}


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="实时管线一键运行（ASR 先就绪，再启管线与 viewer）",
        allow_abbrev=False,  # 见 common/config/pipeline.py 的同名单元注释
    )
    parser.add_argument(
        "audio",
        help="输入音频：单个文件（wav/flac/mp3）、目录，或每行一个路径的清单文本",
    )
    add_common_args(parser)
    parser.add_argument("--device", default="auto", help="运行设备，如 auto/cpu/cuda:0")
    parser.add_argument(
        "--model_type",
        default="eres2netv2",
        choices=["eres2netv2"],
        help="speaker encoder 类型，当前仅支持 eres2netv2",
    )
    parser.add_argument(
        "--segmentation_model",
        default="pyannote/segmentation-3.0",
        help="pyannote segmentation 模型名",
    )
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 级日志")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=int(os.environ.get("STAGE", "3")),
        help="累积式运行阶段：1=只 diarization（RTTM）；2=+separation 分段音频；"
        "3=+ASR 转写（默认）",
    )
    parser.add_argument(
        "--asr_config",
        default=os.environ.get("ASR_CONFIG_PATH", "./config/asr.yaml"),
        help="ASR 转写的 YAML 配置（传给 python -m asr.app；仅 --stage 3 生效）",
    )
    parser.add_argument(
        "--debug", action="store_true", default=env_flag("DEBUG", False)
    )
    parser.add_argument(
        "--show_rttm", action="store_true", default=env_flag("SHOW_RTTM", False)
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        default=env_flag("WITH_VIEWER", True),
        help="不启动 viewer 服务器",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        default=env_flag("WAIT_VIEWER", True),
        help="管线结束后不挂起等待 Ctrl+C，直接收尾退出",
    )
    parser.add_argument(
        "--viewer_port", type=int, default=int(os.environ.get("VIEWER_PORT", "8000"))
    )
    return parser


def main() -> int:
    """CLI 入口：解析参数、跑管线并管理 viewer 生命周期。"""
    parser = build_parser()
    args = parser.parse_args()

    audio = Path(args.audio)
    config_path = Path(args.config)
    check_input_paths(audio, config_path)
    audio_paths = collect_audio_paths(str(audio))
    if not audio_paths:
        raise SystemExit(f"ERROR: 未收集到音频文件: {audio}")
    dirs = pipeline.resolve_run_dirs(args.output_root, _TEST_NAME, args.run_name)

    # 合并 CLI 与 YAML（argparse 默认值 < YAML < 显式 CLI），构建管线配置。
    merged_args = merge_args_with_config(parser, args, sys.argv[1:])
    # 沿用原 build_pipeline_cmd 的传参语义：--debug/--show_rttm 为 true
    # （含环境变量缺省）时作为显式 CLI 覆盖 YAML；为 false 时落回 YAML。
    if args.debug:
        merged_args.debug = True
    if args.show_rttm:
        merged_args.show_rttm = True
    config = config_from_args(merged_args)

    print("==========================================")
    print("Running with YAML config")
    print(f"audio_input={audio} ({len(audio_paths)} file(s))")
    print(f"config_path={config_path}")
    print(f"output_dir={dirs.exp_dir}")
    for name in ("model_path", "hf_cache_dir"):
        if getattr(args, name):
            print(f"{name} override enabled")
    if args.hf_token:
        print("hf_token override enabled")
    print(f"stage={args.stage} with_viewer={args.viewer}")
    print("==========================================")

    # viewer 支持多条 --audio uri=path 映射；批量时逐条传入。
    # 脚本路径锚定仓库根：库式/他目录调用时 cwd 不一定是仓库根。
    viewer_cmd = [
        sys.executable,
        str(BASE_DIR / "viewer" / "server.py"),
        "--exp_root",
        str(dirs.basic_dir),
        "--audio_root",
        "datasets",
    ]
    for path in audio_paths:
        viewer_cmd += ["--audio", f"{Path(path).stem}={path}"]
    viewer_cmd += ["--port", str(args.viewer_port)]

    viewer_proc: subprocess.Popen | None = None

    def start_viewer() -> None:
        """on_asr_ready 钩子：ASR 就绪后启动 viewer（此刻 exp_dir/logs 已建好）。"""

        nonlocal viewer_proc
        viewer_log = open(dirs.logs_dir / "viewer.log", "w", encoding="utf-8")
        try:
            viewer_proc = subprocess.Popen(
                viewer_cmd,
                stdout=viewer_log,
                stderr=subprocess.STDOUT,
            )
        finally:
            # Popen 已复制 fd，父进程持有的句柄可以立即关闭。
            viewer_log.close()
        time.sleep(0.5)
        if viewer_proc.poll() is not None:
            print(
                f"WARNING: viewer 启动失败，见 {dirs.logs_dir}/viewer.log",
                file=sys.stderr,
            )
            viewer_proc = None
            return
        print(f"Viewer: http://127.0.0.1:{args.viewer_port}")
        with open(dirs.logs_dir / "command.log", "a", encoding="utf-8") as log:
            log.write(f"Viewer Command: {shlex.join(viewer_cmd)}\n")

    def stop_viewer() -> None:
        """收尾 viewer 子进程：terminate，超时则 kill。"""
        nonlocal viewer_proc
        if viewer_proc is None:
            return
        viewer_proc.terminate()
        try:
            viewer_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            viewer_proc.kill()
            viewer_proc.wait()
        viewer_proc = None

    try:
        result = pipeline.process(
            audio=audio_paths,
            stage=args.stage,
            config=config,
            config_path=config_path,
            output_root=args.output_root,
            run_name=args.run_name,
            test_name=_TEST_NAME,
            model_path=merged_args.model_path,
            asr_config=args.asr_config,
            verbose=bool(merged_args.verbose or args.debug),
            on_asr_ready=start_viewer if args.viewer else None,
            header_fields={
                "audio_input": audio,
                "num_audio_files": len(audio_paths),
                "config_path": config_path,
                "model_path_override": args.model_path,
                "hf_cache_dir_override": args.hf_cache_dir,
                "with_asr": args.stage >= 3,
                "stage": args.stage,
            },
        )
    except BaseException:
        # ASR 就绪失败等异常路径同样关闭 viewer。
        stop_viewer()
        raise

    # 管线跑完后默认不立即退出：viewer 保持可访问，等用户 Ctrl+C
    # （或 server 因其他原因退出，poll 检测到后同样放行）；
    # --no-wait / WAIT_VIEWER=0 跳过等待（无人值守场景）；
    # 管线被中断（rc=130）时不二次等待，直接收尾。
    if viewer_proc is not None and args.wait and result.returncode != 130:
        if result.returncode == 0:
            print("音频已处理完成。")
        else:
            print(f"管线已退出（rc={result.returncode}）。")
        print(
            f"Viewer 仍在运行: http://127.0.0.1:{args.viewer_port}"
            "（按 Ctrl+C 结束并关闭 viewer）"
        )
        try:
            while viewer_proc.poll() is None:
                time.sleep(1.0)
        except KeyboardInterrupt:
            # 管线已正常结束，这次 Ctrl+C 只用于关闭 viewer，不向外套层抛。
            print("\nShutting down viewer ...", file=sys.stderr)
    stop_viewer()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
