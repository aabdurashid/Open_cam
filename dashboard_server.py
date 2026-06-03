from __future__ import annotations

import json
import os
import glob
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request
from urllib.parse import urlparse


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, stats_path: str, web_root: str, **kwargs):
        self._stats_path = stats_path
        super().__init__(*args, directory=web_root, **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if path == "/api/stats":
            return self._handle_stats()
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/chat":
            return self._handle_chat()
        self.send_error(404, "Not found")

    def _handle_stats(self) -> None:
        if os.path.exists(self._stats_path):
            with open(self._stats_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = _load_latest_summary(self._stats_path)
        self._send_json(payload)

    def _handle_chat(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "Invalid JSON"}, status=400)

        message = str(payload.get("message", "")).strip()
        history = payload.get("history", [])
        stats = payload.get("stats", {})

        if not message:
            return self._send_json({"error": "Empty message"}, status=400)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._send_json({"error": "OPENAI_API_KEY is not set"}, status=400)

        instructions = (
            "You are a dashboard assistant for a computer-vision people-detection "
            "dashboard. Answer in Uzbek unless the user asks for another language. "
            "Use the provided stats JSON when relevant. Be concise and practical. "
            "If a metric is unavailable, say it is unavailable."
        )
        messages = []
        if stats:
            stats_text = json.dumps(stats, ensure_ascii=True)
            messages.append({"role": "user", "content": f"Stats JSON: {stats_text}"})

        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})

        messages.append({"role": "user", "content": message})

        try:
            reply = _call_openai(messages, api_key, instructions)
        except RuntimeError as exc:
            return self._send_json({"error": str(exc)}, status=502)

        self._send_json({"reply": reply})

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _call_openai(messages: list[dict], api_key: str, instructions: str) -> str:
    api_url = os.environ.get(
        "OPENAI_API_URL", "https://api.openai.com/v1/responses"
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "instructions": instructions,
        "input": messages,
        "temperature": 0.2,
        "max_output_tokens": 500,
    }
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(
        api_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI error {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI connection error: {exc.reason}") from exc

    parsed = json.loads(body)
    text = _extract_response_text(parsed)
    if not text:
        raise RuntimeError("OpenAI response was missing text output")
    return text


def _extract_response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()

    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _load_latest_summary(stats_path: str) -> dict:
    report_dir = os.path.dirname(stats_path) or "reports"
    summaries = sorted(glob.glob(os.path.join(report_dir, "summary_*.json")))
    if not summaries:
        return {"status": "waiting"}

    try:
        with open(summaries[-1], "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"status": "waiting"}

    return {
        "status": "finished",
        "timestamp_s": summary.get("duration_s"),
        "count": None,
        "predicted": None,
        "avg_speed": summary.get("avg_speed", 0),
        "anomaly_count": None,
        "avg_count": summary.get("avg_count", 0),
        "max_count": summary.get("max_count", 0),
        "peak_time_s": summary.get("peak_time_s", 0),
        "avg_speed_total": summary.get("avg_speed", 0),
        "max_speed": summary.get("max_speed", 0),
        "anomaly_rate": summary.get("anomaly_rate", 0),
        "total_frames": summary.get("total_frames", 0),
        "series": {"times": [], "counts": [], "preds": [], "anomalies": [], "speeds": []},
    }


def _load_dotenv(base_dir: str) -> None:
    env_path = os.path.join(base_dir, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def start_dashboard_server(
    host: str,
    port: int,
    web_root: str,
    stats_path: str,
) -> ThreadingHTTPServer:
    stats_dir = os.path.dirname(os.path.abspath(stats_path))
    _load_dotenv(os.path.dirname(stats_dir))
    _load_dotenv(stats_dir)
    handler = partial(DashboardRequestHandler, stats_path=stats_path, web_root=web_root)
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
