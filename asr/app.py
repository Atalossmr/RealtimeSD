"""ASR 转写 CLI 入口编排模块：读 exporter 输出目录的音频段 manifest，逐段转写并落盘。

输入是 pipeline 分段导出（`asr_enabled` / `separation_enabled`）的产物：

- `{segments_dir}/{uri}.segments.jsonl`：逐行 {"uri", "speaker_id", "start",
  "end", "path"}；
- `path` 指向的 wav 段（采样率 = pipeline 的 sample_rate）。

两种模式：

- 一次性（默认）：管线跑完后对目录整体转写；
- 跟随（`--follow --done_file <path>`）：与管线同时启动，轮询 manifest 增量
  读取新追加的段、即出即转（manifest 行在 wav 落盘之后才追加，读到行即
  可读音频）；done 哨兵文件出现且积压清空后，统一落盘并退出。

输出：`{uri}.transcript.jsonl` + `{uri}.transcript.txt`（按 start 排序）。

用法：

    # 管线结束后一次性转写
    python3 transcribe.py --segments_dir exp/common/default --config config/config.yaml
    # 与管线同时启动，跟随转写
    python3 transcribe.py --segments_dir exp/common/default --config config/config.yaml \
        --follow --done_file exp/common/default/.diarization_done
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torchaudio

from diarization.config import (
    build_arg_parser,
    config_from_args,
    merge_args_with_config,
)
from diarization.utils import setup_logger

from .transcriber import SegmentTranscriber


logger = logging.getLogger(__name__)

# manifest 文件名后缀：{uri}.segments.jsonl。
_MANIFEST_SUFFIX = ".segments.jsonl"

# 跟随模式的轮询间隔（秒）。
_FOLLOW_POLL_INTERVAL = 1.0


def _load_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with open(manifest_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_transcript(
    uri: str,
    results: list[tuple[float, float, int, str]],
    output_dir: Path,
) -> None:
    """按 start 排序写 transcript（格式与旧 ASRWorker.finalize 一致）。"""

    results = sorted(results, key=lambda r: (r[0], r[1]))
    jsonl_path = output_dir / f"{uri}.transcript.jsonl"
    txt_path = output_dir / f"{uri}.transcript.txt"
    with (
        open(jsonl_path, "w", encoding="utf-8") as jsonl_file,
        open(txt_path, "w", encoding="utf-8") as txt_file,
    ):
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
            txt_file.write(f"[{start:9.3f} - {end:9.3f}] spk{speaker_id}: {text}\n")
    logger.info(
        "[asr] wrote %d segments to %s / %s", len(results), jsonl_path, txt_path
    )


def _transcribe_entry(
    config,
    transcriber: SegmentTranscriber,
    entry: dict,
    results: list[tuple[float, float, int, str]],
) -> None:
    """转写一条 manifest 记录，非空结果追加到 results。"""

    speaker_id = int(entry["speaker_id"])
    start = float(entry["start"])
    end = float(entry["end"])
    waveform, sample_rate = torchaudio.load(entry["path"])
    if int(sample_rate) != int(config.sample_rate):
        raise ValueError(
            f"segment sample rate mismatch: {entry['path']} is {sample_rate}, "
            f"expected {config.sample_rate}"
        )
    text = transcriber.transcribe_segment(waveform)
    if not text:
        logger.info("[asr] empty text: spk%d [%.3f, %.3f]", speaker_id, start, end)
        return
    results.append((start, end, speaker_id, text))
    logger.info(
        "[asr] segment_done: spk%d [%.3f, %.3f] (%.3fs) %r",
        speaker_id,
        start,
        end,
        end - start,
        text[:50],
    )


def transcribe_manifest(
    config,
    manifest_path: Path,
    output_dir: Path,
    transcriber: SegmentTranscriber,
) -> None:
    """转写一个 manifest 覆盖的全部音频段并落盘 transcript。"""

    entries = _load_manifest(manifest_path)
    uri = manifest_path.name[: -len(_MANIFEST_SUFFIX)]
    results: list[tuple[float, float, int, str]] = []
    for entry in entries:
        _transcribe_entry(config, transcriber, entry, results)
    _write_transcript(uri, results, output_dir)


def _read_new_entries(manifest_path: Path, offset: int) -> tuple[list[dict], int]:
    """从 offset 起增量读取 manifest 的完整行，返回 (新条目, 新 offset)。

    末尾不完整的行（writer 正在写）留到下一轮：offset 回退到最后一处换行之后。
    manifest 是 append-only，不回溯已消费部分。
    """

    with open(manifest_path, "rb") as file_obj:
        file_obj.seek(offset)
        data = file_obj.read()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
    entries = [
        json.loads(raw) for raw in data.decode("utf-8").splitlines() if raw.strip()
    ]
    return entries, offset + len(data)


def follow_segments_dir(
    config,
    segments_dir: Path,
    output_dir: Path,
    transcriber: SegmentTranscriber,
    done_file: Path,
) -> None:
    """跟随模式：轮询目录内 manifest，新段即出即转；done 哨兵出现且积压
    清空后统一落盘 transcript 并返回。"""

    offsets: dict[Path, int] = {}
    results_by_uri: dict[str, list[tuple[float, float, int, str]]] = {}
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
                _transcribe_entry(config, transcriber, entry, results)
                progressed = True
        # 哨兵出现 = 管线已结束，不会再有新段；本轮无进展说明积压已清空。
        if done_file.exists() and not progressed:
            break
        time.sleep(_FOLLOW_POLL_INTERVAL)

    for uri, results in results_by_uri.items():
        _write_transcript(uri, results, output_dir)


def main() -> None:
    """CLI 入口。"""

    parser = build_arg_parser()
    parser.description = "exporter 音频段目录的离线/跟随 ASR 转写"
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
    raw_args = parser.parse_args()
    args = merge_args_with_config(parser, raw_args, sys.argv[1:])
    if args.follow and not args.done_file:
        parser.error("--follow requires --done_file")

    config = config_from_args(args)
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir) if args.output_dir else segments_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # 日志文件名区别于 pipeline 的 run.log：output_dir 常与管线输出目录相同，
    # 避免覆盖管线运行日志。
    setup_logger(
        bool(getattr(args, "verbose", False)), str(output_dir / "transcribe.log")
    )

    transcriber = SegmentTranscriber(config)

    if args.follow:
        logger.info(
            "[asr] follow mode: watching %s, done file %s", segments_dir, args.done_file
        )
        # 提前加载模型：加载耗时与管线启动阶段重叠，首个段闭合即可转写。
        transcriber.warmup()
        follow_segments_dir(
            config,
            segments_dir,
            output_dir,
            transcriber,
            Path(args.done_file),
        )
        return

    manifests = sorted(segments_dir.glob(f"*{_MANIFEST_SUFFIX}"))
    if not manifests:
        logger.warning(
            "[asr] no %s files found in %s", f"*{_MANIFEST_SUFFIX}", segments_dir
        )
        return
    logger.info(
        "[asr] found %d manifest(s) in %s, output to %s",
        len(manifests),
        segments_dir,
        output_dir,
    )

    for manifest_path in manifests:
        logger.info("[asr] transcribing %s", manifest_path)
        transcribe_manifest(config, manifest_path, output_dir, transcriber)


if __name__ == "__main__":
    main()


__all__ = ["transcribe_manifest", "follow_segments_dir", "main"]
