"""管线产物可视化服务器（纯标准库，无第三方依赖）。

按产物分层支持管线的三个 stage（stage 越多，可看到的信息越多）：

- stage 1：`{uri}.refined.rttm` + `{uri}.speakers.json` → diarization 时间线
  + speaker 状态（uncertain/合并标记）；
- stage 2：+ `segments/{uri}.segments.jsonl` 与分段 wav → 每段分离音频回放；
- stage 3：+ `transcripts/{uri}.transcript.jsonl` → ASR 转写文本。

扫描 exp_root 下三类产物（并集）建立会话索引，并在 audio_root 下按
`{uri}.wav`（递归）匹配原始整段音频；为前端页面提供会话列表、RTTM/转写/
分段清单 JSON 和音频流（支持 HTTP Range，长音频可 seek）。

实时推送：`GET /api/events` 为 SSE 事件流——后台监听线程对已知产物路径
做 mtime/size 签名比对（0.5s 一拍，无 rglob），产物落盘/原子重写即推送
{"type":"artifact",uri,kind}，会话列表/能力位/live 变化推送
{"type":"sessions"}；前端收到事件后按需拉取对应 JSON 端点（SSE 只传
"什么变了"，不传数据本身）。

用法：

    python3 viewer/server.py --exp_root exp --audio_root datasets --port 8000
    python3 viewer/server.py --audio tingshen_6=datasets/examples/tingshen_6.wav
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

_TRANSCRIPT_SUFFIX = ".transcript.jsonl"
_RTTM_SUFFIX = ".refined.rttm"
_MANIFEST_SUFFIX = ".segments.jsonl"
_SPEAKERS_SUFFIX = ".speakers.json"
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
# 目录全量递归扫描的节流：/api/sessions 轮询重扫的最小间隔；产物是流式增长的，
# 几秒内的可见性延迟可接受（文件内容另有逐请求读取路径，不经此缓存）
_SCAN_MIN_INTERVAL = 3.0
# get() 未命中触发的重扫最小间隔：防止客户端用不存在的 uri 轮询放大磁盘扫描
_RESCAN_MIN_INTERVAL = 10.0
# 新布局中产物的直接父目录名（其上一级才是 run 根目录）
_LAYOUT_SUBDIRS = {"rttm", "transcripts", "segments"}
# 产物监听线程的节拍：只 stat 索引中已有会话的产物路径，代价可忽略
_WATCH_TICK = 0.5
# SSE 心跳间隔：防中间代理/浏览器把空闲长连接掐断
_SSE_HEARTBEAT = 15.0


def _exp_dir_of(artifact: Path) -> Path:
    """由产物路径推 run 根目录。

    新布局在 rttm/ transcripts/ segments/ 子目录（parent.parent），
    旧平铺布局同目录（parent）。

    Args:
        artifact: 产物文件路径。

    Returns:
        run 根目录路径。
    """

    if artifact.parent.name in _LAYOUT_SUBDIRS:
        return artifact.parent.parent
    return artifact.parent


class SessionIndex:
    """uri -> 各类产物路径的索引（可重扫，发现新产物）。

    三类产物任一存在即成会话：refined RTTM（stage1）/ segments manifest
    （stage2）/ transcript（stage3）；speakers.json sidecar 与原始音频
    独立索引后挂到会话上。scan/读路径经 lock 串行化（ThreadingHTTPServer
    并发 handler 共享本索引）。重扫带最小间隔节流，间隔内复用上次结果。

    Attributes:
        exp_roots: 产物扫描根目录列表。
        audio_roots: 原始音频扫描根目录列表。
        overrides: uri -> 显式指定的音频路径。
        sessions: uri -> 会话产物路径字典。
        lock: scan 与读路径的串行化锁。
    """

    def __init__(self, exp_roots: list[Path], audio_roots: list[Path],
                 overrides: dict[str, Path]):
        """建索引并立即全量扫描一次。

        Args:
            exp_roots: 产物扫描根目录列表。
            audio_roots: 原始音频扫描根目录列表。
            overrides: uri -> 显式指定的音频路径。
        """
        self.exp_roots = exp_roots
        self.audio_roots = audio_roots
        self.overrides = overrides
        self.lock = threading.Lock()
        self.sessions: dict[str, dict] = {}
        self._last_scan = float("-inf")  # 首次扫描不受节流限制
        self.scan()

    def scan(self) -> None:
        """重扫目录；距上次扫描不足最小间隔时复用上次结果（节流）。"""

        with self.lock:
            now = time.monotonic()
            if now - self._last_scan < _SCAN_MIN_INTERVAL:
                return
            self._last_scan = now
            self._scan_locked()

    def _scan_locked(self) -> None:
        """全量重扫三类产物并挂上 speakers/audio sidecar（调用方须持锁）。"""

        for exp_root in self.exp_roots:
            for path in sorted(exp_root.rglob(f"*{_RTTM_SUFFIX}")):
                uri = path.name[: -len(_RTTM_SUFFIX)]
                session = self.sessions.setdefault(uri, self._new_session(path))
                session["rttm"] = path
            for path in sorted(exp_root.rglob(f"*{_MANIFEST_SUFFIX}")):
                uri = path.name[: -len(_MANIFEST_SUFFIX)]
                session = self.sessions.setdefault(uri, self._new_session(path))
                session["manifest"] = path
            for path in sorted(exp_root.rglob(f"*{_TRANSCRIPT_SUFFIX}")):
                uri = path.name[: -len(_TRANSCRIPT_SUFFIX)]
                session = self.sessions.setdefault(uri, self._new_session(path))
                session["transcript"] = path
        # speakers.json sidecar 独立索引：新布局在 rttm/ 子目录、旧布局平铺，
        # rglob 两种都兼容。
        speakers_index: dict[str, Path] = {}
        for exp_root in self.exp_roots:
            for path in exp_root.rglob(f"*{_SPEAKERS_SUFFIX}"):
                speakers_index.setdefault(path.name[: -len(_SPEAKERS_SUFFIX)], path)
        audio_index: dict[str, Path] = {}
        for audio_root in self.audio_roots:
            for path in audio_root.rglob("*.wav"):
                audio_index.setdefault(path.stem, path)
        for uri, session in self.sessions.items():
            session["speakers"] = speakers_index.get(uri)
            if uri in self.overrides:
                session["audio"] = self.overrides[uri]
            elif session["audio"] is None:
                session["audio"] = audio_index.get(uri)

    @staticmethod
    def _new_session(artifact: Path) -> dict:
        """构造空会话记录，exp_dir 由首个发现的产物推得。"""

        return {
            "rttm": None,
            "manifest": None,
            "transcript": None,
            "audio": None,
            "speakers": None,
            "exp_dir": _exp_dir_of(artifact),
        }

    def is_live(self, session: dict) -> bool:
        """done 哨兵未出现 = 管线仍在运行，产物还会增量增长（前端据此轮询）。"""

        return not (session["exp_dir"] / ".diarization_done").exists()

    def list_sessions(self) -> list[dict]:
        """列出全部会话（按 uri 排序，含各类产物路径）。"""

        with self.lock:
            return [
                {"uri": uri, **session} for uri, session in sorted(self.sessions.items())
            ]

    def _get(self, uri: str) -> dict | None:
        """取会话快照（不触发重扫）。"""

        with self.lock:
            session = self.sessions.get(uri)
            return dict(session) if session else None

    def get(self, uri: str) -> dict | None:
        """取会话（未命中时先按更大间隔节流重扫一次）。

        产物可能刚落盘；重扫有独立的更大间隔节流，防止用不存在的 uri
        轮询放大磁盘扫描。

        Args:
            uri: 会话标识。

        Returns:
            会话产物路径字典的副本；不存在返回 None。
        """

        session = self._get(uri)
        if session is None:
            with self.lock:
                now = time.monotonic()
                if now - self._last_scan >= _RESCAN_MIN_INTERVAL:
                    self._last_scan = now
                    self._scan_locked()
            session = self._get(uri)
        return session

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        """逐行解析 jsonl；坏行（如读到写入中途的产物）跳过，下轮轮询再读。

        Args:
            path: jsonl 文件路径。

        Returns:
            解析出的 dict 列表（坏行与空行已剔除）。
        """

        entries = []
        with open(path, encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def transcript_json(self, uri: str) -> list[dict] | None:
        """解析 transcript jsonl → 段数组（speaker_id/start/end/text）。

        Args:
            uri: 会话标识。

        Returns:
            按 (start, end) 排序的段列表；无 transcript 返回 None。
        """

        session = self.get(uri)
        if not session or not session["transcript"]:
            return None
        segments = []
        for entry in self._read_jsonl(session["transcript"]):
            try:
                segments.append({
                    "speaker_id": int(entry["speaker_id"]),
                    "start": float(entry["start"]),
                    "end": float(entry["end"]),
                    "text": entry["text"],
                })
            except (KeyError, TypeError, ValueError):
                continue
        segments.sort(key=lambda s: (s["start"], s["end"]))
        return segments

    def rttm_json(self, uri: str) -> list[dict] | None:
        """解析 refined RTTM → 段数组（speaker_id/start/end，无文本）。

        Args:
            uri: 会话标识。

        Returns:
            按 (start, end) 排序的段列表；无 RTTM 返回 None。
        """

        session = self.get(uri)
        if not session or not session["rttm"]:
            return None
        segments = []
        with open(session["rttm"], encoding="utf-8") as file_obj:
            for raw in file_obj:
                line = raw.strip()
                if not line.startswith("SPEAKER"):
                    continue  # 跳过注释头与空行
                parts = line.split()
                try:
                    start = float(parts[3])
                    segments.append({
                        "speaker_id": int(parts[7]),
                        "start": start,
                        "end": start + float(parts[4]),
                    })
                except (IndexError, ValueError):
                    continue  # 坏行：跳过，下轮轮询再读
        segments.sort(key=lambda s: (s["start"], s["end"]))
        return segments

    def segments_json(self, uri: str) -> list[dict] | None:
        """解析 segments manifest → 段数组（file 只返回 basename，不泄露路径）。

        Args:
            uri: 会话标识。

        Returns:
            按 (start, end) 排序的段列表；无 manifest 返回 None。
        """

        session = self.get(uri)
        if not session or not session["manifest"]:
            return None
        segments = []
        for entry in self._read_jsonl(session["manifest"]):
            try:
                segments.append({
                    "speaker_id": int(entry["speaker_id"]),
                    "start": float(entry["start"]),
                    "end": float(entry["end"]),
                    "file": Path(entry["path"]).name,
                })
            except (KeyError, TypeError, ValueError):
                continue
        segments.sort(key=lambda s: (s["start"], s["end"]))
        return segments

    def segment_audio_path(self, uri: str, filename: str) -> Path | None:
        """分段 wav 的实际路径；拒绝路径穿越（只接受 basename）。

        Args:
            uri: 会话标识。
            filename: 段文件名（必须为纯 basename）。

        Returns:
            分段 wav 路径；非法文件名、无 manifest 或文件不存在返回 None。
        """

        if not filename or Path(filename).name != filename:
            return None
        session = self.get(uri)
        if not session or not session["manifest"]:
            return None
        path = session["manifest"].parent / uri / filename
        return path if path.is_file() else None

    def speakers_json(self, uri: str) -> dict | None:
        """读取 refined 级的 speaker 状态 sidecar（{uri}.speakers.json）。

        sidecar 由管线的 refined 级（RefinedRTTMWriter）与 refined RTTM
        同步原子更新，包含各 speaker 的总时长、uncertain 标记与合并事件；
        不存在（旧产物）时返回 None，前端按无标记展示。

        Args:
            uri: 会话标识。

        Returns:
            sidecar JSON；不存在、会话未知或读取失败返回 None。
        """

        session = self.get(uri)
        if not session:
            return None
        status_path = session.get("speakers")
        if status_path is None or not status_path.exists():
            return None
        try:
            with open(status_path, encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except (OSError, json.JSONDecodeError):
            return None  # sidecar 正在原子替换，下一轮轮询再读


class ArtifactWatcher:
    """产物变化监听线程：mtime/size 签名比对，变化即经 SSE 广播事件。

    每拍先 index.scan()（沿用其节流）发现新会话/新产物路径，再对索引中
    已有会话的产物路径逐个 stat（不 rglob，代价可忽略）；签名变化广播
    artifact 事件，会话集合/能力位/live/音频匹配变化广播 sessions 事件。
    事件只描述"什么变了"，数据仍由客户端走 JSON 端点拉取。

    Attributes:
        index: 会话索引。
        subscribers: SSE 订阅者的消息队列集合。
        lock: subscribers 的互斥锁。
    """

    # 会话字段 -> 事件 kind（manifest 对前端即 segments 数据源）
    _KINDS = (("rttm", "rttm"), ("manifest", "segments"),
              ("transcript", "transcript"), ("speakers", "speakers"))

    def __init__(self, index: SessionIndex):
        """建监听线程（未启动）。

        Args:
            index: 会话索引（scan/list_sessions 另有其内部锁与节流）。
        """

        self.index = index
        self.subscribers: set[queue.Queue] = set()
        self.lock = threading.Lock()
        self._signatures: dict[Path, tuple[int, int]] = {}
        self._session_snapshot: dict[str, tuple] = {}
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="artifact-watcher")

    def start(self) -> None:
        """启动后台监听线程（daemon，随进程退出）。"""

        self._thread.start()

    def subscribe(self) -> queue.Queue:
        """注册 SSE 订阅者，返回其事件队列（有界，慢客户端会被丢弃）。"""

        subscription: queue.Queue = queue.Queue(maxsize=100)
        with self.lock:
            self.subscribers.add(subscription)
        return subscription

    def unsubscribe(self, subscription: queue.Queue) -> None:
        """摘除订阅者（断连/队列满时调用）。"""

        with self.lock:
            self.subscribers.discard(subscription)

    def _broadcast(self, event: dict) -> None:
        """向全部订阅者投递事件；队列满视为慢客户端，直接摘除。"""

        with self.lock:
            subscribers = list(self.subscribers)
        for subscription in subscribers:
            try:
                subscription.put_nowait(event)
            except queue.Full:
                self.unsubscribe(subscription)

    def _run(self) -> None:
        """监听主循环：固定节拍 tick，异常（如产物正在原子替换）下拍重试。"""

        while True:
            try:
                self._tick()
            except OSError:
                pass
            time.sleep(_WATCH_TICK)

    def _tick(self) -> None:
        """一拍：重扫索引并比对会话快照与产物签名，有变化即广播。"""

        self.index.scan()
        sessions = self.index.list_sessions()
        snapshot = {
            s["uri"]: (bool(s["rttm"]), bool(s["manifest"]),
                       bool(s["transcript"]), bool(s["audio"]),
                       self.index.is_live(s))
            for s in sessions
        }
        if snapshot != self._session_snapshot:
            self._session_snapshot = snapshot
            self._broadcast({"type": "sessions"})
        for session in sessions:
            for field, kind in self._KINDS:
                path = session.get(field)
                if path is None:
                    continue
                try:
                    stat_result = path.stat()
                except OSError:
                    continue  # 产物被原子替换的瞬间，下拍再读
                signature = (stat_result.st_mtime_ns, stat_result.st_size)
                if self._signatures.get(path) != signature:
                    self._signatures[path] = signature
                    self._broadcast(
                        {"type": "artifact", "uri": session["uri"], "kind": kind})


class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler：会话/产物 JSON API、静态页与音频流（支持 Range）。"""

    server_version = "RealtimeSDViewer/1.2"
    index: SessionIndex  # 由 main() 注入
    watcher: ArtifactWatcher  # 由 main() 注入

    def log_message(self, fmt, *args):  # 保持安静，按需自行恢复
        pass

    def _send_json(self, obj, status: int = 200) -> None:
        """把对象序列化为 JSON 响应。"""

        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        """整文件响应（仅用于静态小文件）。"""

        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_audio(self, path: Path) -> None:
        """音频流：支持 Range 请求（浏览器 seek 依赖 206 响应）。"""

        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header:
            match = _RANGE_RE.fullmatch(range_header.strip())
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                else:  # bytes=-N：尾部 N 字节
                    start = max(0, size - int(match.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type",
                         mimetypes.guess_type(path.name)[0] or "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        try:
            with open(path, "rb") as file_obj:
                file_obj.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = file_obj.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端中断下载（取消播放等），属正常情况

    def _send_events(self) -> None:
        """SSE 事件流：产物/会话变化即推送，心跳保活，断连摘除订阅。

        无 Content-Length：连接保持到客户端断开（SSE 语义）。线程模型为
        每连接一线程（ThreadingHTTPServer），handler 在此循环内驻留。
        """

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        subscription = self.watcher.subscribe()
        try:
            while True:
                try:
                    event = subscription.get(timeout=_SSE_HEARTBEAT)
                    payload = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    payload = ": ping\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端断开（关页面/切会话），属正常情况
        finally:
            self.watcher.unsubscribe(subscription)

    # stdlib 命名约定（do_GET 不可改成蛇形）
    def do_GET(self) -> None:  # noqa: N802
        """路由分发：静态页/静态资源、会话与产物 JSON、音频流。"""

        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        # 静态资源（style.css / app.js 等）：仅 STATIC_DIR 平铺文件，防路径穿越
        name = route.lstrip("/")
        if (
            not route.startswith("/api/")
            and name
            and Path(name).name == name
            and (STATIC_DIR / name).is_file()
        ):
            self._send_file(
                STATIC_DIR / name,
                mimetypes.guess_type(name)[0] or "application/octet-stream",
            )
            return
        if route == "/api/events":
            self._send_events()  # 长连接：handler 驻留至客户端断开
            return
        if route == "/api/sessions":
            self.index.scan()  # 管线运行中产物会后出现；重扫有节流，间隔内复用缓存
            self._send_json([
                {
                    "uri": s["uri"],
                    "audio_url": f"/api/audio/{s['uri']}" if s["audio"] else None,
                    "live": self.index.is_live(s),
                    "has_rttm": bool(s["rttm"]),
                    "has_segments": bool(s["manifest"]),
                    "has_transcript": bool(s["transcript"]),
                    "rttm_url": f"/api/rttm/{s['uri']}" if s["rttm"] else None,
                    "segments_url": (
                        f"/api/segments/{s['uri']}" if s["manifest"] else None
                    ),
                    "transcript_url": (
                        f"/api/transcript/{s['uri']}" if s["transcript"] else None
                    ),
                }
                for s in self.index.list_sessions()
            ])
            return
        if route.startswith("/api/transcript/"):
            segments = self.index.transcript_json(
                unquote(route[len("/api/transcript/"):])
            )
            if segments is None:
                self._send_json({"error": f"no transcript for uri: {route}"},
                                status=404)
            else:
                self._send_json(segments)
            return
        if route.startswith("/api/rttm/"):
            segments = self.index.rttm_json(unquote(route[len("/api/rttm/"):]))
            if segments is None:
                self._send_json({"error": f"no refined RTTM for uri: {route}"},
                                status=404)
            else:
                self._send_json(segments)
            return
        if route.startswith("/api/segments/"):
            segments = self.index.segments_json(
                unquote(route[len("/api/segments/"):])
            )
            if segments is None:
                self._send_json({"error": f"no segments for uri: {route}"},
                                status=404)
            else:
                self._send_json(segments)
            return
        if route.startswith("/api/segment_audio/"):
            rest = unquote(route[len("/api/segment_audio/"):])
            uri, _, filename = rest.partition("/")
            path = self.index.segment_audio_path(uri, filename)
            if path is None:
                self._send_json({"error": f"no segment audio: {rest}"}, status=404)
            else:
                self._send_audio(path)
            return
        if route.startswith("/api/speakers/"):
            # 无 sidecar 时返回 null（200），前端按无标记展示
            self._send_json(
                self.index.speakers_json(unquote(route[len("/api/speakers/"):]))
            )
            return
        if route.startswith("/api/audio/"):
            session = self.index.get(unquote(route[len("/api/audio/"):]))
            if not session or not session["audio"]:
                self._send_json({"error": f"no audio for uri: {route}"}, status=404)
            else:
                self._send_audio(session["audio"])
            return
        self._send_json({"error": f"not found: {route}"}, status=404)

    # stdlib 命名约定（do_POST 不可改成蛇形）
    def do_POST(self) -> None:  # noqa: N802
        """路由分发：/api/shutdown 延迟触发服务关闭。"""

        route = urlparse(self.path).path
        if route == "/api/shutdown":
            self._send_json({"ok": True})
            # shutdown 必须在 serve_forever 所在线程之外调用；延迟一点触发，
            # 保证上面的响应先送达客户端。
            threading.Timer(0.3, self.server.shutdown).start()
            return
        self._send_json({"error": f"not found: {route}"}, status=404)


def main() -> None:
    """CLI 入口：解析参数、建会话索引并启动 HTTP 服务。"""

    parser = argparse.ArgumentParser(description="管线产物可视化服务器")
    parser.add_argument("--exp_root", action="append", default=None,
                        help="产物扫描根目录（可重复，默认 exp）")
    parser.add_argument("--audio_root", action="append", default=None,
                        help="原始音频扫描根目录（可重复，默认 datasets）")
    parser.add_argument("--audio", action="append", default=[],
                        metavar="URI=PATH", help="显式指定某 uri 的音频路径（可重复）")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    exp_roots = [Path(p) for p in (args.exp_root or ["exp"])]
    audio_roots = [Path(p) for p in (args.audio_root or ["datasets"])]
    overrides = {}
    for item in args.audio:
        uri, _, path = item.partition("=")
        if not path:
            parser.error(f"--audio 需要 URI=PATH 形式，收到: {item}")
        overrides[uri] = Path(path)

    index = SessionIndex(exp_roots, audio_roots, overrides)
    print(f"found {len(index.sessions)} session(s)")
    for uri, session in sorted(index.sessions.items()):
        caps = []
        if session["rttm"]:
            caps.append("rttm")
        if session["manifest"]:
            caps.append("segments")
        if session["transcript"]:
            caps.append("transcript")
        audio = session["audio"] or "(no audio matched)"
        print(f"  {uri}: [{', '.join(caps)}] | audio: {audio}")

    ViewerHandler.index = index
    watcher = ArtifactWatcher(index)
    watcher.start()
    ViewerHandler.watcher = watcher
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    print(f"serving on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("server stopped")


if __name__ == "__main__":
    main()
