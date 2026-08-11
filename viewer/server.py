"""ASR 转写结果可视化服务器（纯标准库，无第三方依赖）。

扫描 exp_root 下的 `{uri}.transcript.jsonl`，并在 audio_root 下按 `{uri}.wav`
（递归）匹配原始整段音频，为前端页面提供会话列表、转写 JSON 和音频流
（支持 HTTP Range，长音频可 seek）。

用法：

    python3 viewer/server.py --exp_root exp --audio_root datasets --port 8000
    python3 viewer/server.py --audio tingshen_6=datasets/examples/tingshen_6.wav
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

_TRANSCRIPT_SUFFIX = ".transcript.jsonl"
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class SessionIndex:
    """uri -> (transcript_path, audio_path|None) 的索引（可重扫，发现新产物）。"""

    def __init__(self, exp_roots: list[Path], audio_roots: list[Path],
                 overrides: dict[str, Path]):
        self.exp_roots = exp_roots
        self.audio_roots = audio_roots
        self.overrides = overrides
        self.sessions: dict[str, dict] = {}
        self.scan()

    def scan(self) -> None:
        for exp_root in self.exp_roots:
            for path in sorted(exp_root.rglob(f"*{_TRANSCRIPT_SUFFIX}")):
                uri = path.name[: -len(_TRANSCRIPT_SUFFIX)]
                if uri not in self.sessions:
                    self.sessions[uri] = {"transcript": path, "audio": None}
                else:
                    self.sessions[uri]["transcript"] = path
        audio_index: dict[str, Path] = {}
        for audio_root in self.audio_roots:
            for path in audio_root.rglob("*.wav"):
                audio_index.setdefault(path.stem, path)
        for uri, session in self.sessions.items():
            if uri in self.overrides:
                session["audio"] = self.overrides[uri]
            elif session["audio"] is None:
                session["audio"] = audio_index.get(uri)

    def transcript_json(self, uri: str) -> list[dict] | None:
        session = self.sessions.get(uri)
        if not session:
            return None
        segments = []
        with open(session["transcript"], encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                segments.append({
                    "speaker_id": int(entry["speaker_id"]),
                    "start": float(entry["start"]),
                    "end": float(entry["end"]),
                    "text": entry["text"],
                })
        segments.sort(key=lambda s: (s["start"], s["end"]))
        return segments


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RealtimeSDViewer/1.0"
    index: SessionIndex  # 由 main() 注入

    def log_message(self, fmt, *args):  # 保持安静，按需自行恢复
        pass

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
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

    # stdlib 命名约定（do_GET 不可改成蛇形）
    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if route == "/api/sessions":
            self.index.scan()  # 管线运行中产物会后出现，每次请求重扫
            self._send_json([
                {
                    "uri": uri,
                    "transcript_url": f"/api/transcript/{uri}",
                    "audio_url": f"/api/audio/{uri}" if s["audio"] else None,
                }
                for uri, s in sorted(self.index.sessions.items())
            ])
            return
        if route.startswith("/api/transcript/"):
            uri = unquote(route[len("/api/transcript/"):])
            if uri not in self.index.sessions:
                self.index.scan()
            segments = self.index.transcript_json(uri)
            if segments is None:
                self._send_json({"error": f"unknown uri: {uri}"}, status=404)
            else:
                self._send_json(segments)
            return
        if route.startswith("/api/audio/"):
            uri = unquote(route[len("/api/audio/"):])
            if uri not in self.index.sessions:
                self.index.scan()
            session = self.index.sessions.get(uri)
            if not session or not session["audio"]:
                self._send_json({"error": f"no audio for uri: {uri}"}, status=404)
            else:
                self._send_audio(session["audio"])
            return
        self._send_json({"error": f"not found: {route}"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR 转写结果可视化服务器")
    parser.add_argument("--exp_root", action="append", default=None,
                        help="transcript 扫描根目录（可重复，默认 exp）")
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
    matched = sum(1 for s in index.sessions.values() if s["audio"])
    print(f"found {len(index.sessions)} session(s), {matched} with audio")
    for uri, session in sorted(index.sessions.items()):
        audio = session["audio"] or "(no audio matched)"
        print(f"  {uri}: {session['transcript']} | audio: {audio}")

    ViewerHandler.index = index
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    print(f"serving on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
