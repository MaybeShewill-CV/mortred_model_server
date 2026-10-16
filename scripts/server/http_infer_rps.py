#!/usr/bin/env python3
"""Measure HTTP serving RPS / latency for Mortred inference endpoints.

This is not the in-process FPS tool (`mortred-model-benchmark.out`). It POSTs
the unified `{images[], req_id}` envelope over HTTP, closed-loop, and reports
successful RPS plus latency percentiles.

Stdlib only (no locust, no requests). Each worker thread keeps one HTTP/1.1
connection, posts as fast as the server answers (or at a capped --qps), and
records latency plus status-class counts.

Usage:
  python3 scripts/server/http_infer_rps.py --url http://127.0.0.1:9003/... \\
      --image demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG \\
      --concurrency 8 --duration 30s --token SECRET
  python3 scripts/server/http_infer_rps.py --self-test

Wire encodings (--raw selects the second):
  json (default) - unified {"req_id","images":[base64]} envelope, one or more
                   images per request (this tool sends exactly one).
  raw            - the image bytes ARE the request body, Content-Type image/*
                   or application/octet-stream; trace id rides X-Request-ID.
                   ~25% smaller payload, no base64 codec on either side.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# time / percentile helpers
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> float:
    """'30', '30s', '2m', '1h' -> seconds."""
    raw = text.strip().lower()
    if not raw:
        raise ValueError("empty duration")
    unit = 1.0
    if raw.endswith("ms"):
        unit = 0.001
        raw = raw[:-2]
    elif raw[-1] in "smh":
        unit = {"s": 1.0, "m": 60.0, "h": 3600.0}[raw[-1]]
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("invalid duration %r" % text) from exc
    if value < 0:
        raise ValueError("duration must be non-negative")
    return value * unit


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    index = (len(sorted_values) - 1) * q
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return sorted_values[lo]
    frac = index - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def classify_status(code: int) -> str:
    if 200 <= code < 300:
        return "ok"
    if code == 429:
        return "overload"
    if code in (502, 503, 504):
        return "unavailable"
    if code == 401:
        return "unauthorized"
    if 400 <= code < 500:
        return "client_error"
    if 500 <= code < 600:
        return "server_error"
    return "other"


# ---------------------------------------------------------------------------
# config / report
# ---------------------------------------------------------------------------


@dataclass
class LoadConfig:
    url: str
    image_path: Path
    concurrency: int = 4
    duration_s: float = 0.0
    requests: int = 0
    warmup_s: float = 0.0
    qps: float = 0.0
    timeout_s: float = 30.0
    token: str = ""
    raw: bool = False
    follow_retry_after: bool = False
    ready_url: str = ""
    ready_timeout_s: float = 15.0
    progress: bool = True


@dataclass
class LoadReport:
    url: str = ""
    concurrency: int = 0
    wall_s: float = 0.0
    warmup_s: float = 0.0
    requests: int = 0
    ok: int = 0
    errors: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    by_http: dict[str, int] = field(default_factory=dict)
    rps: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    payload_bytes: int = 0
    image_bytes: int = 0
    server_time_ms_p50: float | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "concurrency": self.concurrency,
            "wall_s": round(self.wall_s, 4),
            "warmup_s": round(self.warmup_s, 4),
            "requests": self.requests,
            "ok": self.ok,
            "errors": self.errors,
            "by_class": self.by_class,
            "by_http": self.by_http,
            "rps": round(self.rps, 3),
            "latency_ms": {k: round(v, 3) for k, v in self.latency_ms.items()},
            "payload_bytes": self.payload_bytes,
            "image_bytes": self.image_bytes,
            "server_time_ms_p50": (
                None if self.server_time_ms_p50 is None else round(self.server_time_ms_p50, 3)
            ),
        }


class _Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self.http: Counter[int] = Counter()
        self.classes: Counter[str] = Counter()
        self.transport = 0
        self.server_times: list[float] = []

    def add(self, latency_ms: float, status: int | None, kind: str) -> None:
        with self._lock:
            self.latencies.append(latency_ms)
            self.classes[kind] += 1
            if status is None:
                self.transport += 1
            else:
                self.http[status] += 1

    def add_server_time(self, value: float) -> None:
        with self._lock:
            self.server_times.append(value)

    def snapshot_counts(self) -> tuple[int, int]:
        with self._lock:
            ok = self.classes.get("ok", 0)
            total = len(self.latencies)
        return ok, total


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def server_time_ms_from_body(body: bytes) -> float | None:
    """Top-level numeric server_time_ms. Missing or non-numeric stays out of the p50."""
    if not body:
        return None
    try:
        doc = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or "server_time_ms" not in doc:
        return None
    value = doc["server_time_ms"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def close_client_first(conn: http.client.HTTPConnection | None, drain_s: float = 1.0) -> None:
    """After the response is read, this side sends FIN.

    The active closer owns TIME_WAIT. Doing it here puts TIME_WAIT on the
    ephemeral port. Connection: close would leave it on the listen port, and
    the next bind fails inside the 15s wait.
    """
    if conn is None:
        return
    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        if drain_s > 0:
            try:
                sock.settimeout(drain_s)
                while sock.recv(4096):
                    pass
            except OSError:
                pass
    try:
        conn.close()
    except OSError:
        pass


def wait_ready(url: str, timeout_s: float) -> None:
    parsed = urllib.parse.urlparse(url)
    deadline = time.monotonic() + timeout_s
    last = "no attempt"
    while time.monotonic() < deadline:
        conn = None
        try:
            conn = _connect(parsed, timeout=min(2.0, timeout_s))
            conn.request(
                "GET",
                parsed.path or "/ready",
                headers={"Connection": "keep-alive", "Accept": "*/*"},
            )
            resp = conn.getresponse()
            resp.read()
            status = resp.status
            close_client_first(conn, drain_s=1.0)
            conn = None
            if 200 <= status < 300:
                return
            last = "HTTP %d" % status
        except OSError as exc:
            last = str(exc)
        finally:
            if conn is not None:
                close_client_first(conn, drain_s=0.0)
        time.sleep(0.1)
    raise TimeoutError("ready probe failed for %s (%s)" % (url, last))


def _connect(parsed: urllib.parse.ParseResult, timeout: float) -> http.client.HTTPConnection:
    host = parsed.hostname or "127.0.0.1"
    if host in ("localhost", "::1"):
        host = "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _path_of(parsed: urllib.parse.ParseResult) -> str:
    path = parsed.path or "/"
    if parsed.query:
        path = path + "?" + parsed.query
    return path


def guess_image_media_type(path: Path) -> str:
    """Content-Type for --raw: image/* when we can tell, octet-stream otherwise.
    The server accepts both as the raw-body trigger."""
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def _build_headers(token: str, body_len: int) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(body_len),
        "Connection": "keep-alive",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _build_raw_headers(token: str, body_len: int, media_type: str, req_id: bytes) -> dict[str, str]:
    headers = {
        "Content-Type": media_type,
        "Content-Length": str(body_len),
        "Connection": "keep-alive",
        "Accept": "application/json",
        "X-Request-ID": req_id.decode("ascii"),
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _retry_after_seconds(resp: http.client.HTTPResponse) -> float:
    raw = resp.getheader("Retry-After")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def run_load(cfg: LoadConfig) -> LoadReport:
    if cfg.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if cfg.duration_s <= 0 and cfg.requests <= 0:
        raise ValueError("set --duration and/or --requests")
    image_bytes = cfg.image_path.read_bytes()
    b64 = base64.b64encode(image_bytes)
    # unique req_id without re-encoding the image on every request
    prefix = b'{"req_id":"'
    mid = b'","images":["'
    suffix = b'"]}'
    sample_id = b"0000000000000000"
    payload_bytes = (
        len(image_bytes) if cfg.raw
        else len(prefix) + len(sample_id) + len(mid) + len(b64) + len(suffix)
    )
    media_type = guess_image_media_type(cfg.image_path)

    parsed = urllib.parse.urlparse(cfg.url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("url must be absolute http(s): %s" % cfg.url)
    path = _path_of(parsed)

    if cfg.ready_url:
        wait_ready(cfg.ready_url, cfg.ready_timeout_s)

    stop = threading.Event()
    measuring = threading.Event()
    remaining: queue.Queue[int] | None = None
    if cfg.requests > 0:
        remaining = queue.Queue()
        for _ in range(cfg.requests):
            remaining.put(1)

    rate_lock = threading.Lock()
    next_slot = [time.monotonic()]
    interval = (1.0 / cfg.qps) if cfg.qps > 0 else 0.0
    seq = [0]
    seq_lock = threading.Lock()
    stats = _Stats()

    def next_req_id() -> bytes:
        with seq_lock:
            seq[0] += 1
            n = seq[0]
        return ("%016x" % (n & 0xFFFFFFFFFFFFFFFF)).encode("ascii")

    def wait_rate() -> None:
        if interval <= 0:
            return
        with rate_lock:
            due = next_slot[0]
            next_slot[0] = max(due, time.monotonic()) + interval
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def take_work() -> tuple[bool, bool]:
        """Returns (do_request, record_stats). Warmup traffic is not counted
        and does not consume --requests."""
        if stop.is_set():
            return False, False
        if not measuring.is_set():
            return True, False
        if remaining is None:
            return True, True
        try:
            remaining.get_nowait()
            return True, True
        except queue.Empty:
            return False, False

    def worker() -> None:
        conn: http.client.HTTPConnection | None = None

        while True:
            proceed, record = take_work()
            if not proceed:
                break
            wait_rate()
            req_id = next_req_id()
            if cfg.raw:
                body = image_bytes
                headers = _build_raw_headers(cfg.token, len(body), media_type, req_id)
            else:
                body = prefix + req_id + mid + b64 + suffix
                headers = _build_headers(cfg.token, len(body))
            started = time.perf_counter()
            status: int | None = None
            kind = "transport"
            try:
                if conn is None:
                    conn = _connect(parsed, cfg.timeout_s)
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                resp_body = resp.read()
                status = resp.status
                kind = classify_status(status)
                if record and status == 200:
                    server_ms = server_time_ms_from_body(resp_body)
                    if server_ms is not None:
                        stats.add_server_time(server_ms)
                if cfg.follow_retry_after and status == 429:
                    nap = _retry_after_seconds(resp)
                    if nap > 0:
                        time.sleep(nap)
                # Keep the connection for the next request. If the server
                # already asked to close, FIN from this side now so TIME_WAIT
                # stays on the ephemeral port instead of the listen port.
                if "close" in (resp.getheader("Connection") or "").lower():
                    close_client_first(conn, drain_s=0.05)
                    conn = None
            except (OSError, http.client.HTTPException, TimeoutError):
                kind = "transport"
                status = None
                close_client_first(conn, drain_s=0.0)
                conn = None
            latency_ms = (time.perf_counter() - started) * 1000.0
            if record:
                stats.add(latency_ms, status, kind)
            if stop.is_set():
                break
        close_client_first(conn, drain_s=1.0)

    threads = [threading.Thread(target=worker, name="load-%d" % i, daemon=True)
               for i in range(cfg.concurrency)]
    for thread in threads:
        thread.start()

    if cfg.warmup_s > 0:
        time.sleep(cfg.warmup_s)
    measuring.set()
    t0 = time.perf_counter()

    def progress_loop() -> None:
        while not stop.wait(1.0):
            if not cfg.progress:
                continue
            ok, total = stats.snapshot_counts()
            elapsed = time.perf_counter() - t0
            rps = total / elapsed if elapsed > 0 else 0.0
            sys.stderr.write(
                "\r  %6.1fs  requests=%-7d ok=%-7d rps=%.1f" % (elapsed, total, ok, rps)
            )
            sys.stderr.flush()

    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()

    if cfg.duration_s > 0:
        time.sleep(cfg.duration_s)
        stop.set()
    for thread in threads:
        thread.join()
    stop.set()
    if cfg.progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    wall = time.perf_counter() - t0
    latencies = sorted(stats.latencies)
    ok = stats.classes.get("ok", 0)
    total = len(latencies)
    report = LoadReport(
        url=cfg.url,
        concurrency=cfg.concurrency,
        wall_s=wall,
        warmup_s=cfg.warmup_s,
        requests=total,
        ok=ok,
        errors=total - ok,
        by_class=dict(stats.classes),
        by_http={str(k): v for k, v in sorted(stats.http.items())},
        rps=(ok / wall) if wall > 0 else 0.0,
        latency_ms={
            "min": latencies[0] if latencies else 0.0,
            "mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": latencies[-1] if latencies else 0.0,
        },
        payload_bytes=payload_bytes,
        image_bytes=len(image_bytes),
        server_time_ms_p50=percentile(sorted(stats.server_times), 0.50) if stats.server_times else None,
    )
    return report


def print_report(report: LoadReport) -> None:
    print("url          : %s" % report.url)
    print("concurrency  : %d" % report.concurrency)
    print("wall_s       : %.3f (warmup %.3f)" % (report.wall_s, report.warmup_s))
    print("requests     : %d  ok=%d  errors=%d" % (report.requests, report.ok, report.errors))
    print("rps          : %.2f  (successful / wall)" % report.rps)
    lat = report.latency_ms
    print(
        "latency_ms   : min=%.2f  p50=%.2f  p90=%.2f  p95=%.2f  p99=%.2f  max=%.2f  mean=%.2f"
        % (lat["min"], lat["p50"], lat["p90"], lat["p95"], lat["p99"], lat["max"], lat["mean"])
    )
    if report.by_http:
        print("http         : " + ", ".join("%s=%d" % item for item in report.by_http.items()))
    if report.by_class:
        print("class        : " + ", ".join("%s=%d" % item for item in sorted(report.by_class.items())))
    print("payload      : %d bytes/req (image %d)" % (report.payload_bytes, report.image_bytes))
    if report.server_time_ms_p50 is None:
        print("server_time_ms: p50=n/a")
    else:
        print("server_time_ms: p50=%.3f" % report.server_time_ms_p50)


# ---------------------------------------------------------------------------
# self-test: in-process server with a small stall to exercise concurrency
# ---------------------------------------------------------------------------


class _SelfTestHandler(BaseHTTPRequestHandler):
    stall_s = 0.01

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if ctype.startswith("image/") or ctype == "application/octet-stream":
            ok = length > 0 and (self.headers.get("X-Request-ID") or "") != ""
        else:
            try:
                doc = json.loads(body.decode("utf-8"))
                ok = isinstance(doc.get("images"), list) and doc["images"]
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                ok = False
        time.sleep(self.stall_s)
        if not ok:
            payload = b'{"status":50,"status_str":"bad"}'
            self.send_response(400)
        else:
            payload = (
                b'{"status":0,"status_str":"OK","server_time_ms":12.5,'
                b'"results":[{"status":0,"data":{}}]}'
            )
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _self_test() -> int:
    image = Path("/tmp/mortred-load-selftest.bin")
    image.write_bytes(b"\xff\xd8fakejpeg")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SelfTestHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:%d/infer" % port
    failed = 0
    try:
        report = run_load(
            LoadConfig(
                url=url,
                image_path=image,
                concurrency=4,
                requests=40,
                warmup_s=0.0,
                timeout_s=5.0,
                progress=False,
            )
        )
        if report.ok != 40 or report.errors != 0:
            print("FAIL: expected 40 ok, got %s" % report.to_dict(), file=sys.stderr)
            failed += 1
        if report.rps <= 0:
            print("FAIL: rps should be positive", file=sys.stderr)
            failed += 1
        # 4 workers, 10ms stall => roughly 400 rps theoretical; allow wide band
        if report.latency_ms["p50"] < 5 or report.latency_ms["p50"] > 200:
            print("FAIL: unexpected p50 %s" % report.latency_ms, file=sys.stderr)
            failed += 1
        if parse_duration("2m") != 120.0 or parse_duration("500ms") != 0.5:
            print("FAIL: parse_duration", file=sys.stderr)
            failed += 1
        if parse_duration("0s") != 0.0 or parse_duration("0") != 0.0:
            print("FAIL: parse_duration zero warmup", file=sys.stderr)
            failed += 1
        # raw encoding: image bytes as body, trace id on X-Request-ID
        raw_report = run_load(
            LoadConfig(
                url=url,
                image_path=image,
                concurrency=2,
                requests=20,
                warmup_s=0.0,
                timeout_s=5.0,
                raw=True,
                progress=False,
            )
        )
        if raw_report.ok != 20 or raw_report.errors != 0:
            print("FAIL: raw mode expected 20 ok, got %s" % raw_report.to_dict(), file=sys.stderr)
            failed += 1
        if raw_report.payload_bytes != image.stat().st_size:
            print("FAIL: raw payload_bytes %d != image size %d"
                  % (raw_report.payload_bytes, image.stat().st_size), file=sys.stderr)
            failed += 1
        if report.server_time_ms_p50 != 12.5 or raw_report.server_time_ms_p50 != 12.5:
            print(
                "FAIL: server_time_ms p50 %s %s"
                % (report.server_time_ms_p50, raw_report.server_time_ms_p50),
                file=sys.stderr,
            )
            failed += 1
        if server_time_ms_from_body(b'{"status":0}') is not None:
            print("FAIL: missing server_time_ms entered the p50", file=sys.stderr)
            failed += 1
        try:
            build_model_target("YOLOV8", "", ROOT)
            print("FAIL: uppercase model id must be rejected", file=sys.stderr)
            failed += 1
        except SystemExit:
            pass
        url_s, image_s = build_model_target("yolov8s", "", ROOT)
        if url_s != "http://127.0.0.1:8080/v1/models/yolov8s/infer":
            print("FAIL: yolov8s gateway url %s" % url_s, file=sys.stderr)
            failed += 1
        if not str(image_s).endswith("object_detection/bus.jpg"):
            print("FAIL: yolov8s demo image %s" % image_s, file=sys.stderr)
            failed += 1
    finally:
        server.shutdown()
        try:
            image.unlink()
        except OSError:
            pass
    if failed:
        print("http_infer_rps.py --self-test failed", file=sys.stderr)
        return 1
    print("http_infer_rps.py --self-test passed")
    return 0


def build_model_target(model_id: str, image: str, root: Path,
                        gateway: str = "http://127.0.0.1:8080") -> tuple[str, Path]:
    """Gateway URL for a lowercase product id, plus its demo image when --image is empty."""
    if any(ch.isupper() for ch in model_id):
        raise SystemExit(
            "catalog ids are lowercase (/v1/models/%s/infer). %r is not a catalog id"
            % (model_id.lower(), model_id)
        )
    table_path = root / "conf" / "product_static_bs1.json"
    rows = json.loads(table_path.read_text(encoding="utf-8")).get("models", [])
    by_id = {row["id"]: row for row in rows}
    if model_id not in by_id:
        names = ", ".join(sorted(by_id))
        raise SystemExit("unknown model %r. product ids: %s" % (model_id, names))
    url = gateway.rstrip("/") + "/v1/models/%s/infer" % model_id
    if image:
        path = Path(image)
        if not path.is_absolute():
            path = root / path
    else:
        path = root / by_id[model_id]["demo"]
    return url, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="", help="absolute POST URL (model uri or /v1/models/id/infer)")
    parser.add_argument("--model", default="",
                        help="lowercase product id; sets the gateway URL http://127.0.0.1:8080/v1/models/<id>/infer")
    parser.add_argument("--gateway", default="http://127.0.0.1:8080",
                        help="gateway origin used with --model")
    parser.add_argument("--image", default="", help="input image path (default: the id's demo image)")
    parser.add_argument("-c", "--concurrency", type=int, default=8,
                        help="closed-loop worker threads (keep-alive connections)")
    parser.add_argument("-d", "--duration", default="",
                        help="measure window, e.g. 30s / 2m (stop after this even if --requests remains)")
    parser.add_argument("-n", "--requests", type=int, default=0,
                        help="stop after this many attempted requests (0 = duration only)")
    parser.add_argument("--warmup", default="0s", help="discard samples for this long before measuring")
    parser.add_argument("--qps", type=float, default=0.0,
                        help="optional shared rate cap across workers (0 = as fast as the server)")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request socket timeout seconds")
    parser.add_argument("--token", default=os.environ.get("MORTRED_GATEWAY_AUTH_TOKEN", ""),
                        help="Authorization Bearer (default MORTRED_GATEWAY_AUTH_TOKEN)")
    parser.add_argument("--raw", action="store_true",
                        help="send the image bytes as the raw request body (Content-Type image/*) "
                             "instead of the base64 JSON envelope; one image per request, "
                             "trace id rides X-Request-ID")
    parser.add_argument("--follow-retry-after", action="store_true",
                        help="sleep Retry-After on HTTP 429 instead of immediately issuing the next POST")
    parser.add_argument("--ready-url", default="", help="GET this URL until 2xx before load (e.g. /ready)")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    parser.add_argument("--out", default="", help="write JSON report to this path")
    parser.add_argument("--quiet", action="store_true", help="no per-second progress on stderr")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    # Parse durations before self-test so default --warmup "0s" is exercised.
    duration_s = parse_duration(args.duration) if args.duration else 0.0
    warmup_s = parse_duration(args.warmup) if args.warmup else 0.0
    if args.self_test:
        return _self_test()
    if args.model:
        try:
            built_url, built_image = build_model_target(args.model, args.image, ROOT, args.gateway)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 1
        args.url = built_url
        args.image = str(built_image)
        if not args.ready_url:
            args.ready_url = args.gateway.rstrip("/") + "/healthz"
    if not args.url or not args.image:
        parser.error("--url and --image are required unless --self-test")
    if duration_s <= 0 and args.requests <= 0:
        parser.error("set --duration and/or --requests")
    cfg = LoadConfig(
        url=args.url,
        image_path=Path(args.image),
        concurrency=args.concurrency,
        duration_s=duration_s,
        requests=args.requests,
        warmup_s=warmup_s,
        qps=args.qps,
        timeout_s=args.timeout,
        token=args.token,
        raw=args.raw,
        follow_retry_after=args.follow_retry_after,
        ready_url=args.ready_url,
        ready_timeout_s=args.ready_timeout,
        progress=not args.quiet,
    )
    if not cfg.image_path.is_file():
        print("image not found: %s" % cfg.image_path, file=sys.stderr)
        return 1
    try:
        report = run_load(cfg)
    except (ValueError, TimeoutError, OSError) as exc:
        print("load failed: %s" % exc, file=sys.stderr)
        return 1
    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print("wrote %s" % args.out)
    return 0 if report.ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
