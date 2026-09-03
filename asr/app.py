"""ASR 转写 CLI 入口编排模块：读 exporter 输出目录的音频段 manifest，逐段转写并落盘。

输入是 pipeline 分段导出（`separation_enabled`）的产物：

- `{segments_dir}/{uri}.segments.jsonl`：逐行 {"uri", "speaker_id", "start",
  "end", "path"}；
- `path` 指向的 wav 段（采样率 = pipeline 的 sample_rate）。

两种模式（同一个增量消费循环，只是终止条件不同）：

- 一次性（默认，无 `--done_file`）：管线跑完后对目录单遍消费，转写当前
  已有的全部段即落盘返回；
- 跟随（`--follow --done_file <path>`）：与管线同时启动，轮询 manifest 增量
  读取新追加的段、即出即转（manifest 行在 wav 落盘之后才追加，读到行即
  可读音频）；每转完一段即时追加落盘（供 viewer 实时展示），done
  哨兵文件出现且积压清空后做最后一次全量排序落盘并退出。

输出：`{uri}.transcript.jsonl`（按 start 排序）。

用法：

    # 管线结束后一次性转写
    python3 -m asr.app --segments_dir <exp_dir>/segments --config config/asr.yaml
    # 与管线同时启动，跟随转写
    python3 -m asr.app --segments_dir <exp_dir>/segments --config config/asr.yaml \
        --follow --done_file <exp_dir>/.diarization_done
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torchaudio

from .config import (
    AsrConfig,
    build_arg_parser,
    config_from_args,
    merge_args_with_config,
)
from common.utils import setup_logger

from .transcriber import SegmentTranscriber


logger = logging.getLogger(__name__)

# manifest 文件名后缀：{uri}.segments.jsonl。
_MANIFEST_SUFFIX = ".segments.jsonl"

# 跟随模式的轮询间隔（秒）。
_FOLLOW_POLL_INTERVAL = 1.0


class SegmentConfigError(ValueError):
    """确定性配置错误（如段采样率与 asr.yaml 不一致）。

    第一次失败即注定后续全部失败：消费循环遇到直接 fail-fast（非零退出），
    不做逐段跳过（否则会产出空 transcript 且 rc=0）。
    """


def _write_transcript(
    uri: str,
    results: list[tuple[float, float, int, str]],
    output_dir: Path,
) -> None:
    """按 start 排序全量重写 transcript jsonl。

    收尾兜底用；运行期增量落盘走 `_append_transcript`。格式与旧
    ASRWorker.finalize 一致。

    Args:
        uri: 音频标识（输出文件名前缀）。
        results: (start, end, speaker_id, text) 元组列表，函数内部排序。
        output_dir: transcript 输出目录。
    """

    results = sorted(results, key=lambda r: (r[0], r[1]))
    jsonl_path = output_dir / f"{uri}.transcript.jsonl"
    # 临时文件 + 原子替换：viewer 每 2s 轮询，原地重写会让读者看到截半文件。
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as jsonl_file:
        for start, end, speaker_id, text in results:
            jsonl_file.write(
                json.dumps(
                    {
                        "uri": uri,
                        "speaker_id": speaker_id,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(tmp_path, jsonl_path)
    logger.info("[asr] wrote %d segments to %s", len(results), jsonl_path)


def _append_transcript(
    uri: str,
    result: tuple[float, float, int, str],
    output_dir: Path,
) -> None:
    """把一条新结果追加到 transcript jsonl（append-only）。

    运行期增量落盘用。单行单次 write 保证行完整（与 manifest 的
    append-only 模式一致）；viewer 轮询若读到不完整行会跳过、下轮再读
    （见 viewer._read_jsonl）。段按 manifest 顺序处理、start 基本递增，
    append 顺序天然接近排序，收尾时 `_write_transcript` 再做一次全量排序
    重写。

    Args:
        uri: 音频标识（输出文件名前缀）。
        result: (start, end, speaker_id, text) 元组。
        output_dir: transcript 输出目录。
    """

    start, end, speaker_id, text = result
    line = (
        json.dumps(
            {
                "uri": uri,
                "speaker_id": speaker_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    jsonl_path = output_dir / f"{uri}.transcript.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(line)


def _transcribe_entry(
    config,
    transcriber: SegmentTranscriber,
    entry: dict,
    results: list[tuple[float, float, int, str]],
) -> tuple[float, float, int, str] | None:
    """转写一条 manifest 记录，非空结果追加到 results 并返回该条结果。

    Args:
        config: ASR 配置（用于校验段采样率）。
        transcriber: 段级转写器。
        entry: manifest 单行记录（含 speaker_id/start/end/path）。
        results: 累积结果列表，非空结果会追加到其中。

    Returns:
        (start, end, speaker_id, text) 元组；空文本返回 None。

    Raises:
        SegmentConfigError: 段采样率与配置不一致（确定性配置错误，调用方
            fail-fast）。其余异常（坏音频、瞬时 I/O 等）向上抛，由调用方
            按瞬时失败处理。
    """

    speaker_id = int(entry["speaker_id"])
    start = float(entry["start"])
    end = float(entry["end"])
    waveform, sample_rate = torchaudio.load(entry["path"])
    if int(sample_rate) != int(config.sample_rate):
        raise SegmentConfigError(
            f"segment sample rate mismatch: {entry['path']} is {sample_rate}, "
            f"expected {config.sample_rate}"
        )
    text = transcriber.transcribe_segment(waveform)
    if not text:
        logger.info("[asr] empty text: spk%d [%.3f, %.3f]", speaker_id, start, end)
        return None
    result = (start, end, speaker_id, text)
    results.append(result)
    logger.info(
        "[asr] segment_done: spk%d [%.3f, %.3f] (%.3fs) %r",
        speaker_id,
        start,
        end,
        end - start,
        text[:50],
    )
    return result


def _read_new_entries(manifest_path: Path, offset: int) -> tuple[list[dict], int]:
    """从 offset 起增量读取 manifest 的完整行。

    末尾不完整的行（writer 正在写）留到下一轮：offset 回退到最后一处换行
    之后。manifest 是 append-only，不回溯已消费部分。单行 JSON 损坏时跳过
    该行（offset 照常前进，不重试），不杀死整个消费循环。

    Args:
        manifest_path: manifest 文件路径。
        offset: 已消费的字节偏移。

    Returns:
        (新条目列表, 新 offset) 元组。
    """

    with open(manifest_path, "rb") as file_obj:
        file_obj.seek(offset)
        data = file_obj.read()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
    entries = []
    for raw in data.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            # 单行损坏不杀死整个消费循环：跳过该行（offset 照常前进，
            # 不重试），等待 writer 侧修复或人工介入。
            logger.warning(
                "[asr] skipping corrupt manifest line in %s: %.100r",
                manifest_path,
                raw,
            )
    return entries, offset + len(data)


def consume_segments_dir(
    config: AsrConfig,
    segments_dir: Path,
    output_dir: Path,
    transcriber: SegmentTranscriber,
    done_file: Path | None = None,
) -> None:
    """统一的增量消费循环：扫描目录内 manifest，转写尚未消费的段。

    两种模式只是终止条件不同，全量读 = offset 全从 0 开始的第一轮增量读：

    - 一次性（done_file=None）：单遍消费当前已有段即退出；
    - 跟随（done_file 给定）：轮询直到哨兵出现且本轮无新段（积压清空），
      每转完一段即时追加落盘（供 viewer 实时读取）。

    转写失败的段记录后在循环结束、最终落盘前重试一次（仍失败的 WARNING
    列出丢失清单）；退出时（含异常/中断，走 finally）按 uri 做一次全量
    排序重写落盘 transcript（运行期是 append-only，保证最终文件有序）。

    Args:
        config: ASR 配置。
        segments_dir: exporter 输出目录（扫描其中所有 *.segments.jsonl）。
        output_dir: transcript 输出目录。
        transcriber: 段级转写器。
        done_file: 跟随模式的结束哨兵文件路径；None 表示一次性模式。

    Raises:
        SegmentConfigError: 确定性配置错误（如段采样率不一致），fail-fast
            直接向上抛。
    """

    offsets: dict[Path, int] = {}
    results_by_uri: dict[str, list[tuple[float, float, int, str]]] = {}
    failed_entries: list[dict] = []  # 转写失败的 manifest 记录，收尾重试一次
    try:
        while True:
            progressed = False
            for manifest_path in sorted(segments_dir.glob(f"*{_MANIFEST_SUFFIX}")):
                uri = manifest_path.name[: -len(_MANIFEST_SUFFIX)]
                entries, new_offset = _read_new_entries(
                    manifest_path, offsets.get(manifest_path, 0)
                )
                offsets[manifest_path] = new_offset
                results = results_by_uri.setdefault(uri, [])
                for entry in entries:
                    try:
                        new_result = _transcribe_entry(
                            config, transcriber, entry, results
                        )
                    except SegmentConfigError as exc:
                        # 确定性配置错误：第一段失败即注定全部失败，逐段跳过
                        # 只会产出空 transcript 且 rc=0，直接 fail-fast。
                        logger.error("[asr] fatal config error, aborting: %s", exc)
                        raise
                    except Exception:
                        # 单段失败（坏音频、瞬时 I/O 抖动等）不杀死整个消费
                        # 循环：记录后跳过（offset 不回退），收尾统一重试一次。
                        logger.exception(
                            "[asr] skipping segment: %s", entry.get("path", entry)
                        )
                        failed_entries.append(entry)
                        continue
                    progressed = True
                    # 跟随模式逐段追加落盘（append-only）：viewer 能看到转写
                    # 逐段生长，而不是等整批积压转完才一次性出现；且避免每段
                    # 全量重写导致磁盘 I/O 随段数平方增长。
                    if done_file is not None and new_result is not None:
                        _append_transcript(uri, new_result, output_dir)
            if done_file is None:
                break
            # 哨兵出现 = 管线已结束，不会再有新段；本轮无进展说明积压已清空。
            if done_file.exists() and not progressed:
                break
            time.sleep(_FOLLOW_POLL_INTERVAL)
        # 收尾对账：转写失败的段重试一次（瞬时 I/O 抖动兜底），仍失败的
        # WARNING 列出丢失清单；重试成功的结果经下方全量落盘一并写出。
        if failed_entries:
            logger.info("[asr] retrying %d failed segment(s)", len(failed_entries))
            lost_entries = []
            for entry in failed_entries:
                retry_uri = str(entry.get("uri", ""))
                try:
                    _transcribe_entry(
                        config,
                        transcriber,
                        entry,
                        results_by_uri.setdefault(retry_uri, []),
                    )
                except Exception:
                    logger.exception(
                        "[asr] retry failed, segment lost: %s",
                        entry.get("path", entry),
                    )
                    lost_entries.append(entry)
            if lost_entries:
                logger.warning(
                    "[asr] %d segment(s) lost after retry: %s",
                    len(lost_entries),
                    [e.get("path", e) for e in lost_entries],
                )
    finally:
        # 无论正常结束还是异常/中断退出，已转写结果都兜底做一次全量排序
        # 重写（运行期是 append-only 追加，这里保证最终文件按 start 有序）。
        for uri, results in results_by_uri.items():
            _write_transcript(uri, results, output_dir)

    if not results_by_uri:
        logger.warning(
            "[asr] no %s files found in %s", f"*{_MANIFEST_SUFFIX}", segments_dir
        )


def main() -> None:
    """CLI 入口。"""

    parser = build_arg_parser()
    parser.add_argument(
        "--segments_dir",
        required=True,
        help="exporter 输出目录（扫描其中所有 *.segments.jsonl）",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="跟随模式：与管线同时启动，轮询增量转写，需配合 --done_file",
    )
    parser.add_argument(
        "--done_file",
        default=None,
        help="跟随模式的结束哨兵文件路径：出现且积压清空后落盘退出",
    )
    parser.add_argument(
        "--ready_file",
        default=None,
        help="就绪哨兵文件路径：跟随模式下模型加载完成后 touch，供外部编排脚本等待",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="转写日志路径（默认 {output_dir}/transcribe.log）",
    )
    raw_args = parser.parse_args()
    args = merge_args_with_config(parser, raw_args, sys.argv[1:])
    if args.follow and not args.done_file:
        parser.error("--follow requires --done_file")

    config = config_from_args(args)
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir) if args.output_dir else segments_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # 日志文件名区别于 pipeline 的 run.log：output_dir 常与管线输出目录相同，
    # 避免覆盖管线运行日志；--log_file 可显式指定（run.py 传入 logs/ 子目录）。
    log_file = args.log_file or str(output_dir / "transcribe.log")
    setup_logger(bool(getattr(args, "verbose", False)), log_file)

    transcriber = SegmentTranscriber(config)

    if args.follow:
        logger.info(
            "[asr] follow mode: watching %s, done file %s", segments_dir, args.done_file
        )
        # 提前加载模型：加载耗时与管线启动阶段重叠，首个段闭合即可转写。
        transcriber.warmup()
        if args.ready_file:
            Path(args.ready_file).touch()
            logger.info("[asr] model ready, touched %s", args.ready_file)
        consume_segments_dir(
            config, segments_dir, output_dir, transcriber, Path(args.done_file)
        )
        return

    logger.info("[asr] offline mode: scanning %s, output to %s", segments_dir, output_dir)
    consume_segments_dir(config, segments_dir, output_dir, transcriber)


if __name__ == "__main__":
    main()


__all__ = ["consume_segments_dir", "main"]
