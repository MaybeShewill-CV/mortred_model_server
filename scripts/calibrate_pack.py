#!/usr/bin/env python3
"""Calibrate worker_nums for a machine pack. Never writes conf/server.

For each pack id, sweep worker_nums at concurrency equal to twice that worker
count, POST with http_infer_rps --raw (5s warmup), and sample GPU memory for
that server process (NVML compute-apps by pid, else a unique
mortred-model-server name). Whole-card memory.used is never treated as the
model; if WSL has no process row, occupancy is device.used minus a sample taken
before spawn (gpu_mem_source=device_delta).
w* is the last worker count that still passes the error, p99, and queue gates
and gains at least 10% RPS. With two or more eligible ids, a joint pass may
lower a worker count so the set fits in the memory budget. It does not run a
joint load.

Optional --write-pack updates final worker_nums, occupancy stamps, and the
per-model decode_auto vote (cpu|gpu) on [pack.<ID>] in that pack file only.
Runs shorter than 30s are not written. Git example packs should stay
worker_nums=1 and must not commit gpu_mem_mib.

Usage:
  python3 scripts/calibrate_pack.py --pack /etc/mortred/pack.toml
  python3 scripts/calibrate_pack.py --pack /tmp/machine-pack.toml \\
      --workers 1,2,4 --duration 8s --output logs/calibrate-yolov8.json
  python3 scripts/calibrate_pack.py --pack /etc/mortred/pack.toml --write-pack
  python3 scripts/calibrate_pack.py --self-test
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

from http_infer_rps import LoadConfig, close_client_first, parse_duration, run_load  # noqa: E402
from pack_trt import find_server_toml, pack_ids, pack_server_config, server_listen  # noqa: E402
from repo_toml import load_toml  # noqa: E402
from test_server import build_catalog, resolve_demo_image, resolve_server  # noqa: E402

OOM_MARKERS = ("out of memory", "cuda oom", "cudnn_status_alloc_failed", "std::bad_alloc")
CALIBRATOR_VERSION = "v4"
RELEASE_DURATION_S = 30.0
DECODE_VOTE_DURATION_S = 10.0
LIGHT_THROUGHPUT_QPS = 500.0
LIGHT_SWEEP = [1, 2, 4, 8, 16]
OTHER_SWEEP = [1, 2, 4, 8]
SCALE_NOTES = {"single_point", "approx_linear", "partial_scale"}
# Seed latency budgets in milliseconds. Not a measured SLO.
P99_CAP_MS = {
    "mobilenetv2": 40.0,
    "yolov8s": 50.0,
    "libface": 40.0,
    "pphuman_mobile": 40.0,
    "dbnet": 60.0,
    "dinov2_vits14": 50.0,
    "superpoint": 40.0,
    "realesrgan": 400.0,
    "metric3d_512": 200.0,
}
_THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps", re.IGNORECASE)


def gpu_inventory() -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return {"raw": line}
    try:
        total = float(parts[1])
    except ValueError:
        total = 0.0
    return {"name": parts[0], "memory_total_mib": total, "driver": parts[2]}


def _csv_float(raw: str) -> float | None:
    text = raw.strip().replace(" MiB", "").replace("MiB", "")
    if not text or text.upper() in ("N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def gpu_mem_device_mib() -> float | None:
    """Whole-device used MiB (WSL often has this when per-pid compute-apps is empty)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    total = 0.0
    any_val = False
    for line in out.splitlines():
        val = _csv_float(line.split(",")[0] if line else "")
        if val is None:
            continue
        total += val
        any_val = True
    return total if any_val else None


def gpu_compute_apps() -> list[tuple[int | None, str, float | None]]:
    """NVML compute-apps rows: (pid, process_name, used_mib)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    rows: list[tuple[int | None, str, float | None]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid: int | None
        try:
            pid = int(parts[0])
        except ValueError:
            pid = None
        rows.append((pid, parts[1], _csv_float(parts[2])))
    return rows


def gpu_mem_process(pid: int) -> tuple[float | None, str]:
    """Per-process GPU used MiB from NVML. Never returns whole-device used."""
    apps = gpu_compute_apps()
    for apid, _name, mem in apps:
        if apid == pid and mem is not None:
            return mem, "nvml_pid"
    named = [
        (apid, name, mem)
        for apid, name, mem in apps
        if mem is not None and "mortred-model-server" in name
    ]
    if len(named) == 1:
        return named[0][2], "nvml_name"
    return None, "unavailable"


def occupancy_from_device(now: float | None, baseline: float | None) -> float | None:
    if now is None or baseline is None:
        return None
    return max(0.0, now - baseline)


def gpu_mem_model(pid: int, baseline_device_mib: float | None) -> tuple[float | None, str]:
    """Model occupancy: NVML process row, else idle-to-now device delta.

    Whole-card memory.used is not the model. A delta vs the sample taken
    before this process started is only an estimate (WSL often has no PID
    row); it is labeled device_delta.
    """
    proc, src = gpu_mem_process(pid)
    if proc is not None:
        return proc, src
    delta = occupancy_from_device(gpu_mem_device_mib(), baseline_device_mib)
    if delta is not None:
        return delta, "device_delta"
    return None, "unavailable"


def log_looks_oom(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in OOM_MARKERS)


def wait_port_free(port: int, timeout_s: float = 15.0) -> None:
    """A stopped listener's port can linger (graceful drain / TIME_WAIT);
    the next spawn binds the same port, so wait it out instead of racing."""
    import socket as _socket
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        s = _socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return
        except OSError:
            s.close()
            _time.sleep(0.2)


DECODE_GRID_SIZES = (
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (3840, 2160),
)
DECODE_GRID_QUALITIES = (50, 60, 80, 90)


def generate_decode_grid(out_dir: Path) -> list[dict]:
    """16 JPEGs: 4 resolutions x 4 qualities. Noise + mild blur, fixed seed."""
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(20260922)
    fixtures: list[dict] = []
    for width, height in DECODE_GRID_SIZES:
        noise = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frame = cv2.GaussianBlur(noise, (5, 5), 0)
        for quality in DECODE_GRID_QUALITIES:
            path = out_dir / ("decode_%dx%d_q%d.jpg" % (width, height, quality))
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
                raise RuntimeError("failed to write %s" % path)
            fixtures.append(
                {
                    "path": path,
                    "width": width,
                    "height": height,
                    "quality": quality,
                    "bytes": path.stat().st_size,
                }
            )
    if len(fixtures) != 16:
        raise RuntimeError("decode grid must have 16 fixtures, got %d" % len(fixtures))
    return fixtures


def vote_decode_auto(cases: list[dict]) -> dict:
    """Majority of valid CPU-vs-GPU RPS pairs. CPU wins a case only if strictly higher."""
    cpu_wins = 0
    gpu_wins = 0
    for case in cases:
        cpu = case.get("rps_cpu")
        gpu = case.get("rps_gpu")
        if not isinstance(cpu, (int, float)) or not isinstance(gpu, (int, float)):
            continue
        if float(cpu) > float(gpu):
            cpu_wins += 1
        else:
            gpu_wins += 1
    valid = cpu_wins + gpu_wins
    if valid == 0 or cpu_wins > valid / 2.0:
        winner = "cpu"
    else:
        winner = "gpu"
    return {
        "decode_auto": winner,
        "cpu_wins": cpu_wins,
        "gpu_wins": gpu_wins,
        "valid": valid,
    }


def _http_get(port: int, path: str, token: str, timeout_s: float = 5.0) -> tuple[int, str] | None:
    """GET and close this side first so TIME_WAIT stays off the listen port."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        conn.request(
            "GET",
            path,
            headers={
                "Authorization": "Bearer %s" % token,
                "Accept": "text/plain",
                "Connection": "keep-alive",
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        status = resp.status
        close_client_first(conn, drain_s=1.0)
        conn = None
        return status, body.decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, TimeoutError):
        return None
    finally:
        if conn is not None:
            close_client_first(conn, drain_s=0.0)


def fetch_decode_counters(port: int, token: str) -> dict[str, int] | None:
    got = _http_get(port, "/metrics", token)
    if got is None or got[0] != 200:
        return None
    counts: dict[str, int] = {}
    for match in re.finditer(r'mortred_jpeg_decode_total\{backend="([^"]+)"\} (\d+)', got[1]):
        counts[match.group(1)] = int(match.group(2))
    return counts


def _counter_delta(
    before: dict[str, int] | None, after: dict[str, int] | None, keys: tuple[str, ...]
) -> int | None:
    if before is None or after is None:
        return None
    return sum(after.get(key, 0) - before.get(key, 0) for key in keys)


def load_pinned_rps(
    url: str,
    image: Path,
    workers: int,
    duration_s: float,
    auth_token: str,
    port: int,
    expect_gpu: bool,
) -> tuple[float | None, str]:
    before = fetch_decode_counters(port, auth_token)
    report = run_load(
        LoadConfig(
            url=url,
            image_path=image,
            concurrency=max(1, workers),
            duration_s=duration_s,
            warmup_s=min(2.0, max(0.5, duration_s / 5.0)),
            token=auth_token,
            raw=True,
            progress=False,
        )
    )
    after = fetch_decode_counters(port, auth_token)
    if report.ok <= 0 or report.rps <= 0:
        return None, "load failed ok=%s errors=%s" % (report.ok, report.errors)
    if expect_gpu:
        delta = _counter_delta(before, after, ("jpeggpu",))
        if delta is not None and delta < max(1, report.ok // 2):
            return None, "jpeggpu counter did not advance (delta=%s ok=%s)" % (delta, report.ok)
    else:
        delta = _counter_delta(before, after, ("cpu-reduced", "cpu-full"))
        if delta is not None and delta < max(1, report.ok // 2):
            return None, "cpu counter did not advance (delta=%s ok=%s)" % (delta, report.ok)
    return float(report.rps), "ok=%s" % report.ok


def log_has_jpeggpu_ready(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    return "jpeggpu decoder ready" in log_path.read_text(encoding="utf-8", errors="replace")


def run_decode_path_grid(
    root: Path,
    binary: Path,
    entry: dict,
    workers: int,
    fixtures: list[dict],
    duration_s: float,
    auth_token: str,
    pin: str,
    model_config: str = "",
    server_toml: Path | None = None,
) -> tuple[list[float | None], str]:
    model_id = entry["model_id"]
    found = server_toml if server_toml is not None else find_server_toml(root, model_id)
    if found is None:
        return [None] * len(fixtures), "no conf/server mapping"
    port = int(entry["port"])
    uri = entry["uri"]
    log_path = root / "logs" / ("calibrate-decode-%s-%s.log" % (model_id, pin))
    extra = {"MORTRED_IMAGE_DECODE_BACKEND": pin}
    if model_config:
        extra["MORTRED_MODEL_CONFIG_FILE"] = model_config
    proc = start_model(root, binary, model_id, found, workers, log_path, extra, auth_token)
    scores: list[float | None] = []
    note = "ok"
    try:
        if not wait_http_ready("http://127.0.0.1:%d/ready" % port, 120.0, proc):
            return [None] * len(fixtures), "probe server not ready (see %s)" % log_path
        if pin == "gpu" and not log_has_jpeggpu_ready(log_path):
            return [None] * len(fixtures), "gpu decoder not armed (see %s)" % log_path
        url = "http://127.0.0.1:%d%s" % (port, uri)
        for fixture in fixtures:
            rps, why = load_pinned_rps(
                url, fixture["path"], workers, duration_s, auth_token, port, pin == "gpu"
            )
            scores.append(rps)
            print(
                "    %s %dx%d q%d rps=%s (%s)"
                % (pin, fixture["width"], fixture["height"], fixture["quality"], rps, why),
                flush=True,
            )
            if rps is None:
                note = why
    finally:
        stop_proc(proc)
        wait_port_free(port)
    return scores, note


def calibrate_decode_vote(
    root: Path,
    binary: Path,
    entry: dict,
    workers: int,
    duration_s: float,
    auth_token: str,
    grid_dir: Path,
    model_config: str = "",
    server_toml: Path | None = None,
) -> dict:
    try:
        fixtures = generate_decode_grid(grid_dir)
    except Exception as exc:
        return {"decode_auto": "cpu", "error": "grid generate failed: %s" % exc}
    print("  decode grid: 16 fixtures, w=%d, gpu then cpu" % workers, flush=True)
    gpu_scores, gpu_note = run_decode_path_grid(
        root, binary, entry, workers, fixtures, duration_s, auth_token, "gpu", model_config, server_toml
    )
    if all(score is None for score in gpu_scores):
        return {
            "decode_auto": "cpu",
            "cpu_wins": 0,
            "gpu_wins": 0,
            "valid": 0,
            "error": gpu_note,
            "gpu_note": gpu_note,
        }
    cpu_scores, cpu_note = run_decode_path_grid(
        root, binary, entry, workers, fixtures, duration_s, auth_token, "cpu", model_config, server_toml
    )
    cases = []
    for fixture, cpu_rps, gpu_rps in zip(fixtures, cpu_scores, gpu_scores):
        winner = None
        if isinstance(cpu_rps, (int, float)) and isinstance(gpu_rps, (int, float)):
            winner = "cpu" if float(cpu_rps) > float(gpu_rps) else "gpu"
        cases.append(
            {
                "width": fixture["width"],
                "height": fixture["height"],
                "quality": fixture["quality"],
                "bytes": fixture["bytes"],
                "rps_cpu": cpu_rps,
                "rps_gpu": gpu_rps,
                "winner": winner,
            }
        )
    summary = vote_decode_auto(cases)
    summary["cpu_note"] = cpu_note
    summary["gpu_note"] = gpu_note
    summary["cases"] = cases
    print(
        "  decode vote: cpu=%d gpu=%d valid=%d -> %s"
        % (summary["cpu_wins"], summary["gpu_wins"], summary["valid"], summary["decode_auto"]),
        flush=True,
    )
    return summary


def is_release_duration(duration_s: float) -> bool:
    return duration_s + 1e-9 >= RELEASE_DURATION_S


def benchmark_throughput_qps(root: Path, model_id: str) -> float | None:
    path = root / "docs" / "benchmark-h3" / ("%s.txt" % model_id)
    if not path.is_file():
        return None
    match = _THROUGHPUT_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    return float(match.group(1))


def default_worker_sweep(root: Path, model_id: str) -> list[int]:
    qps = benchmark_throughput_qps(root, model_id)
    if qps is not None and qps >= LIGHT_THROUGHPUT_QPS:
        return list(LIGHT_SWEEP)
    return list(OTHER_SWEEP)


def parse_worker_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        nums = [int(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        nums = [int(item) for item in text.split(",") if item.strip()]
    if not nums or min(nums) < 1:
        raise SystemExit("worker list must be integers >= 1, got %r" % (value,))
    return nums


def pack_table(pack: Path, catalog_id: str | None = None) -> dict:
    table = load_toml(pack)
    pack_body = table.get("pack")
    if catalog_id is None:
        if isinstance(pack_body, dict):
            return pack_body
        return {}
    if isinstance(pack_body, dict) and isinstance(pack_body.get(catalog_id), dict):
        return pack_body[catalog_id]
    dotted = table.get("pack.%s" % catalog_id)
    return dotted if isinstance(dotted, dict) else {}


def workers_for_id(
    root: Path,
    model_id: str,
    cli_workers: list[int] | None,
    pack: Path | None,
) -> list[int]:
    if cli_workers:
        return list(cli_workers)
    if pack is not None:
        parsed = parse_worker_list(pack_table(pack, model_id).get("worker_sweep"))
        if parsed:
            return parsed
    return default_worker_sweep(root, model_id)


def p99_cap_ms(model_id: str, pack: Path | None) -> float | None:
    if pack is not None:
        raw = pack_table(pack, model_id).get("p99_ms_cap")
        if raw is not None and str(raw).strip() != "":
            return float(raw)
    cap = P99_CAP_MS.get(model_id.lower())
    return None if cap is None else float(cap)


def joint_budget_mib(pack: Path, gpu: dict | None) -> float | None:
    raw = pack_table(pack).get("gpu_memory_budget_mib")
    if isinstance(pack_table(pack).get("gpu_memory_budget_mib"), dict):
        raw = None
    if raw is not None and not isinstance(raw, dict) and str(raw).strip() != "":
        return float(raw)
    total = None if not isinstance(gpu, dict) else gpu.get("memory_total_mib")
    if isinstance(total, (int, float)) and float(total) > 0:
        return 0.85 * float(total)
    return None


def run_post_sweep(report: dict, joint_fn, decode_fn) -> dict:
    """Joint residency is decided before the decode vote."""
    joint_fn(report)
    decode_fn(report)
    return report


def fetch_metrics_text(port: int, token: str) -> str | None:
    got = _http_get(port, "/metrics", token)
    if got is None or got[0] != 200:
        return None
    return got[1]


def _metric_series(text: str, name: str, model: str) -> dict | None:
    bucket_re = re.compile(
        r'%s_bucket\{model="%s",le="([^"]+)"\} ([0-9.eE+-]+)'
        % (re.escape(name), re.escape(model))
    )
    count_re = re.compile(
        r'%s_count\{model="%s"\} ([0-9.eE+-]+)' % (re.escape(name), re.escape(model))
    )
    buckets: list[tuple[float, float]] = []
    for match in bucket_re.finditer(text):
        if match.group(1) == "+Inf":
            continue
        buckets.append((float(match.group(1)), float(match.group(2))))
    count_match = count_re.search(text)
    if not buckets and count_match is None:
        return None
    buckets.sort()
    count = float(count_match.group(1)) if count_match else (buckets[-1][1] if buckets else 0.0)
    return {"buckets": buckets, "count": count}


def _hist_delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    previous = {le: count for le, count in before["buckets"]}
    buckets = [
        (le, max(0.0, count - previous.get(le, 0.0))) for le, count in after["buckets"]
    ]
    return {"buckets": buckets, "count": max(0.0, after["count"] - before["count"])}


def histogram_p50(hist: dict | None) -> float | None:
    if not hist or hist["count"] <= 0:
        return None
    target = 0.5 * float(hist["count"])
    prev_le = 0.0
    prev_count = 0.0
    for le, count in hist["buckets"]:
        if count >= target:
            width = count - prev_count
            if width <= 0:
                return float(le)
            return prev_le + (target - prev_count) / width * (float(le) - prev_le)
        prev_le = float(le)
        prev_count = count
    return prev_le


def _point_rates(point: dict) -> tuple[int, int, int]:
    total = int(point.get("requests") or 0)
    ok = int(point.get("ok") or 0)
    if "errors" in point and point.get("errors") is not None:
        errors = int(point.get("errors") or 0)
    else:
        errors = max(0, total - ok)
    overload = int(point.get("overload") or 0)
    return total, errors, overload


def assess_point(
    point: dict, last_good: dict | None, p99_cap: float | None
) -> tuple[str | None, str | None]:
    """Return (stop_reason, scale_note). A stop_reason rejects this point."""
    if not point.get("started") or point.get("oom"):
        why = "oom_or_start_failed" if point.get("oom") else "start_failed"
        return why, None
    total, errors, overload = _point_rates(point)
    if total <= 0:
        return "error_rate", None
    non_429 = max(0, errors - overload)
    if non_429 / total >= 0.01:
        return "error_rate", None
    if overload / total >= 0.01:
        return "overload", None
    if p99_cap is not None:
        p99 = point.get("p99_ms")
        if p99 is None or float(p99) > float(p99_cap):
            return "latency_budget", None
    if point.get("metrics_missing"):
        return "metrics_missing", None
    queue_p50 = point.get("queue_p50_ms")
    infer_p50 = point.get("inference_p50_ms")
    if queue_p50 is not None and infer_p50 is not None and float(queue_p50) > float(infer_p50):
        return "queue_dominates", None
    if last_good is not None:
        prev_rps = max(float(last_good.get("rps") or 0.0), 1e-6)
        gain = float(point.get("rps") or 0.0) / prev_rps
        w_ratio = float(point["worker_nums"]) / float(last_good["worker_nums"])
        if gain < 1.10:
            return "rps_gain_below_10pct", None
        if gain < 0.5 * w_ratio:
            return "saturating", None
        note = "approx_linear" if gain >= 0.8 * w_ratio else "partial_scale"
        return None, note
    return None, "single_point"


def pick_w_star(
    points: list[dict], p99_cap: float | None = None
) -> tuple[int, str, bool, list[int]]:
    """Last point that passed every gate.

    Returns worker_nums, reason, eligible, and the worker counts that passed.
    The first point failing a gate is not eligible and does not become w*.
    """
    last_good: dict | None = None
    note = "single_point"
    passed: list[int] = []
    for point in points:
        stop, scale = assess_point(point, last_good, p99_cap)
        if stop:
            if last_good is None:
                return int(point.get("worker_nums") or 1), stop, False, passed
            return int(last_good["worker_nums"]), stop, True, passed
        note = scale or note
        last_good = point
        passed.append(int(point["worker_nums"]))
    if last_good is None:
        return 1, "start_failed", False, passed
    return int(last_good["worker_nums"]), note, True, passed


def pick_demotion(
    sizes: dict[str, float],
    current: dict[str, int],
    grids: dict[str, list[int]],
    passed: dict[str, set[int]],
) -> tuple[str, int] | None:
    """Drop the largest resident one grid step, onto a worker count that passed."""
    ranked = sorted(sizes, key=lambda model_id: (-float(sizes.get(model_id) or 0.0), model_id))
    for model_id in ranked:
        grid = list(grids.get(model_id) or [])
        w = int(current[model_id])
        if w not in grid:
            continue
        index = grid.index(w)
        if index <= 0:
            continue
        target = int(grid[index - 1])
        if target not in passed.get(model_id, set()):
            continue
        return model_id, target
    return None


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _ceil_positive_mib(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return max(1, int(math.ceil(number)))


def _fmt_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, 3))
    text = str(value)
    return '"%s"' % text.replace("\\", "\\\\").replace('"', '\\"')


def _section_pack_id(header: str) -> str | None:
    """[pack] -> ''; [pack.YOLOV8] -> YOLOV8; other -> None."""
    if header == "[pack]":
        return ""
    if header.startswith("[pack.") and header.endswith("]"):
        return header[6:-1].strip()
    return None


def write_pack_fields(
    pack: Path,
    id_fields: dict[str, dict[str, object]],
    pack_fields: dict[str, object] | None = None,
) -> list[dict]:
    """Upsert keys on [pack] / [pack.<ID>]. Refuses conf/server paths."""
    if _under(pack, ROOT / "conf" / "server"):
        raise ValueError("refusing to write conf/server; pass a pack toml")
    if _under(pack, ROOT / "conf" / "packs"):
        raise ValueError(
            "refusing to write git example pack %s; copy it to a machine-local file" % pack
        )
    cleaned_ids: dict[str, dict[str, object]] = {}
    for catalog_id, fields in id_fields.items():
        kept = {k: v for k, v in fields.items() if v is not None}
        if kept:
            cleaned_ids[catalog_id] = kept
    cleaned_pack = {k: v for k, v in (pack_fields or {}).items() if v is not None}
    if not cleaned_ids and not cleaned_pack:
        return []
    original = pack.read_text(encoding="utf-8")
    newline = "\n" if "\r\n" not in original else "\r\n"
    lines = original.splitlines()
    out: list[str] = []
    current: str | None = None
    pending: dict[str, object] | None = None
    saw_pack_global = False
    remaining_ids = {k: dict(v) for k, v in cleaned_ids.items()}
    remaining_pack = dict(cleaned_pack)
    applied: list[dict] = []

    def close_section() -> None:
        nonlocal pending
        if pending:
            for key, value in list(pending.items()):
                out.append("%s = %s" % (key, _fmt_toml_value(value)))
            pending.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            close_section()
            current = _section_pack_id(stripped)
            if current == "":
                saw_pack_global = True
                pending = remaining_pack
            elif current is not None:
                pending = remaining_ids.get(current)
            else:
                pending = None
            out.append(line)
            continue
        if pending is not None:
            replaced = False
            for key in list(pending):
                match = re.match(r"^(\s*" + re.escape(key) + r"\s*=\s*).*$", line)
                if match:
                    out.append("%s%s" % (match.group(1), _fmt_toml_value(pending.pop(key))))
                    replaced = True
                    break
            if replaced:
                continue
        out.append(line)
    close_section()
    for leftover in remaining_ids:
        if remaining_ids[leftover]:
            applied.append({"id": leftover, "error": "no [pack.%s] table" % leftover})
    if remaining_pack and not saw_pack_global:
        block = ["[pack]"]
        for key, value in remaining_pack.items():
            block.append("%s = %s" % (key, _fmt_toml_value(value)))
        block.append("")
        insert_at = 0
        for i, line in enumerate(out):
            if line.strip().startswith("[pack."):
                insert_at = i
                break
        else:
            insert_at = len(out)
        out[insert_at:insert_at] = block
        remaining_pack.clear()
        applied.append({"section": "pack", "action": "insert"})
    for catalog_id, fields in cleaned_ids.items():
        if any(a.get("id") == catalog_id and a.get("error") for a in applied):
            continue
        applied.append({"id": catalog_id, "action": "write", **fields})
    text = newline.join(out)
    if original.endswith(newline) or original.endswith("\n"):
        if not text.endswith(newline):
            text += newline
    pack.write_text(text, encoding="utf-8")
    return applied


def write_pack_worker_nums(pack: Path, updates: dict[str, int]) -> list[dict]:
    """Set worker_nums on [pack.<ID>] tables. Refuses conf/server paths."""
    return write_pack_fields(pack, {k: {"worker_nums": v} for k, v in updates.items()})


def _chosen_worker_nums(row: dict) -> int | None:
    w = row.get("final_worker_nums")
    if not isinstance(w, int):
        w = row.get("suggested_worker_nums")
    if isinstance(w, int) and 1 <= w <= 256:
        return w
    return None


def occupancy_fields_from_report(
    report: dict,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    id_fields: dict[str, dict[str, object]] = {}
    if report.get("release_measurement") is False:
        return {}, {}
    for row in report.get("models") or []:
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if row.get("eligible") is False or row.get("reason") == "start_failed":
            continue
        points = row.get("points") or []
        if not any(p.get("started") for p in points):
            continue
        w = _chosen_worker_nums(row)
        if w is None:
            continue
        fields: dict[str, object] = {"worker_nums": w, "gpu_mem_at_workers": w}
        match = next(
            (p for p in points if p.get("started") and int(p.get("worker_nums") or 0) == w),
            None,
        )
        if match is None:
            match = next((p for p in reversed(points) if p.get("started")), None)
        if match is not None:
            mib = _ceil_positive_mib(match.get("gpu_mem_mib_peak"))
            if mib is not None:
                fields["gpu_mem_mib"] = mib
            src = match.get("gpu_mem_source")
            if isinstance(src, str) and src and src != "unavailable":
                fields["gpu_mem_source"] = src
            rss = _ceil_positive_mib(match.get("rss_mib"))
            if rss is not None:
                fields["rss_mib"] = rss
            thr = match.get("thread_count")
            if isinstance(thr, int) and thr > 0:
                fields["thread_count"] = thr
            fds = match.get("fd_count")
            if isinstance(fds, int) and fds > 0:
                fields["fd_count"] = fds
        decode = row.get("decode")
        if isinstance(decode, dict):
            winner = decode.get("decode_auto")
            if winner in ("cpu", "gpu"):
                fields["decode_auto"] = winner
        id_fields[model_id] = fields
    if not id_fields:
        # No eligible row: --write-pack must leave the pack untouched (the
        # self-test pins both the empty return and the unchanged file). The
        # machine-level fields below (GPU identity, host RAM) are only recorded
        # together with at least one eligible id, so an all-ineligible
        # calibration cannot half-update a pack.
        return id_fields, {}
    pack_fields: dict[str, object] = {}
    gpu = report.get("gpu")
    if isinstance(gpu, dict):
        name = gpu.get("name")
        if isinstance(name, str) and name.strip():
            pack_fields["gpu_name"] = name.strip()
        total = _ceil_positive_mib(gpu.get("memory_total_mib"))
        if total is not None:
            pack_fields["gpu_memory_total_mib"] = total
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    pack_fields["host_memory_total_mib"] = max(1, (int(parts[1]) + 1023) // 1024)
                break
    except OSError:
        pass
    return id_fields, pack_fields


def pack_writes_from_report(report: dict) -> dict[str, int]:
    updates: dict[str, int] = {}
    if report.get("release_measurement") is False:
        return updates
    for row in report.get("models") or []:
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if row.get("eligible") is False or row.get("reason") == "start_failed":
            continue
        points = row.get("points") or []
        if not any(p.get("started") for p in points):
            continue
        w = _chosen_worker_nums(row)
        if w is not None:
            updates[model_id] = w
    return updates


def listen_port_blocked(port: int) -> str | None:
    """Return an error string if 127.0.0.1:port cannot be bound (already served)."""
    if port <= 0:
        return "invalid port"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        return str(exc)
    finally:
        sock.close()
    return None


def find_server_bin(root: Path) -> Path | None:
    for rel in ("_bin/mortred-model-server.out", "bin/mortred-model-server.out"):
        path = root / rel
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def _close_popen_stdio(proc: subprocess.Popen[bytes]) -> None:
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def _wait_quiet(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    try:
        proc.wait(timeout=timeout_s)
        return True
    except Exception:
        return False


def stop_proc(proc: subprocess.Popen[bytes] | None) -> None:
    """Never raise: HTTP/calibrate drivers must survive a stuck server."""
    try:
        if proc is None:
            return
        if proc.poll() is not None:
            _close_popen_stdio(proc)
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                _close_popen_stdio(proc)
                return
        if _wait_quiet(proc, 8.0):
            _close_popen_stdio(proc)
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        if not _wait_quiet(proc, 5.0):
            subprocess.run(["killall", "-9", "-q", "mortred-model-server.out"], check=False)
            _wait_quiet(proc, 3.0)
        _close_popen_stdio(proc)
    except BaseException:
        try:
            subprocess.run(["killall", "-9", "-q", "mortred-model-server.out"], check=False)
        except Exception:
            pass
        try:
            if proc is not None:
                _close_popen_stdio(proc)
        except Exception:
            pass


def calib_auth_token() -> str:
    # Per-run random token shared by every spawned calibration instance and
    # the load driver. The server has no anonymous mode (an EMPTY configured
    # token rejects every request), so the spawn MUST carry a token and the
    # client MUST present the same one - ambient MORTRED_AUTH_TOKEN is never
    # inherited (it would 401 the token-less history of this script).
    import secrets

    return secrets.token_urlsafe(24)


def start_model(
    root: Path,
    binary: Path,
    model_id: str,
    server_toml: Path,
    workers: int,
    log_path: Path,
    extra_env: dict[str, str],
    auth_token: str,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("MORTRED_LISTEN_PORT", None)
    env["MORTRED_AUTH_TOKEN"] = auth_token
    env["MORTRED_LISTEN_HOST"] = "127.0.0.1"
    env["MORTRED_PROJECT_ROOT"] = str(root)
    env["MORTRED_WORKER_NUMS"] = str(workers)
    env["LD_LIBRARY_PATH"] = "%s:%s:%s" % (
        root / "_lib",
        root / "3rd_party" / "libs",
        env.get("LD_LIBRARY_PATH", ""),
    )
    env.update(extra_env)
    handle = log_path.open("wb")
    handle.write(
        (
            "spawn model=%s server_toml=%s model_config=%s workers=%s\n"
            % (
                model_id,
                server_toml,
                extra_env.get("MORTRED_MODEL_CONFIG_FILE", ""),
                workers,
            )
        ).encode()
    )
    handle.flush()
    return subprocess.Popen(
        [str(binary), "--model", model_id, str(server_toml)],
        cwd=str(binary.parent),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_http_ready(url: str, timeout_s: float, proc: subprocess.Popen[bytes]) -> bool:
    from http_infer_rps import wait_ready

    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            wait_ready(url, min(1.5, deadline - time.monotonic()))
            return True
        except TimeoutError as exc:
            last = str(exc)
        time.sleep(0.2)
    return False



def process_floors(pid: int) -> dict:
    """Sample RSS MiB, Threads, and open fd count for a live server pid."""
    out: dict = {"rss_mib": None, "thread_count": None, "fd_count": None}
    if pid <= 0:
        return out
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        text = status.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out["rss_mib"] = max(1, (int(parts[1]) + 1023) // 1024)
            elif line.startswith("Threads:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out["thread_count"] = int(parts[1])
    fd_dir = Path(f"/proc/{pid}/fd")
    if fd_dir.is_dir():
        try:
            out["fd_count"] = sum(1 for _ in fd_dir.iterdir())
        except OSError:
            pass
    return out


def sample_mem_during(
    pid: int,
    baseline: float | None,
    stop: threading.Event,
    bucket: list[tuple[float, str]],
) -> None:
    while not stop.is_set():
        val, scope = gpu_mem_model(pid, baseline)
        if val is not None:
            bucket.append((val, scope))
        stop.wait(0.4)


def catalog_entry(root: Path, model_id: str) -> dict:
    entries = build_catalog(root)
    entry, err = resolve_server(model_id, entries)
    if entry is None:
        raise SystemExit("unknown catalog id %s: %s" % (model_id, err))
    return entry


def run_one_point(
    root: Path,
    binary: Path,
    entry: dict,
    image: Path,
    workers: int,
    duration_s: float,
    concurrency: int,
    model_config: str,
    auth_token: str,
    server_toml: Path | None = None,
) -> dict:
    model_id = entry["model_id"]
    log_path = root / "logs" / ("calibrate-%s-w%d.log" % (model_id, workers))
    extra = {}
    if model_config:
        extra["MORTRED_MODEL_CONFIG_FILE"] = model_config
    found = server_toml if server_toml is not None else find_server_toml(root, model_id)
    if found is None:
        return {
            "worker_nums": workers,
            "started": False,
            "oom": False,
            "error": "no conf/server mapping",
        }
    port, uri = server_listen(found)
    if port <= 0 or not uri:
        return {
            "worker_nums": workers,
            "started": False,
            "oom": False,
            "error": "invalid port/server_uri in %s" % found,
        }
    blocked = listen_port_blocked(port)
    if blocked:
        return {
            "worker_nums": workers,
            "started": False,
            "oom": False,
            "error": (
                "port %d already in use (%s); stop mortred-supervisor / "
                "another mortred-model-server before calibrate"
                % (port, blocked)
            ),
        }
    baseline = gpu_mem_device_mib()
    proc = start_model(root, binary, model_id, found, workers, log_path, extra, auth_token)
    ready_url = "http://127.0.0.1:%d/ready" % port
    infer_url = "http://127.0.0.1:%d%s" % (port, uri)
    point: dict = {
        "worker_nums": workers,
        "started": False,
        "oom": False,
        "log": str(log_path.relative_to(root)),
        "url": infer_url,
    }
    try:
        if not wait_http_ready(ready_url, 90.0, proc):
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            point["oom"] = log_looks_oom(text)
            tail = "\n".join(text.strip().splitlines()[-12:])
            if "Cannot start server" in text:
                point["error"] = (
                    "HTTP listen failed after model init (exit=%s) on %s; "
                    "engine loaded — usually port %d is already bound "
                    "(stop mortred-supervisor / another model server)"
                    % (proc.poll(), ready_url, port)
                )
            else:
                point["error"] = "not ready (exit=%s) probing %s" % (proc.poll(), ready_url)
            if tail:
                point["log_tail"] = tail
            return point
        point["started"] = True
        ready_mem, ready_scope = gpu_mem_model(proc.pid, baseline)
        point["gpu_mem_mib_ready"] = ready_mem
        point["gpu_mem_source"] = ready_scope
        point["gpu_mem_device_used_mib"] = gpu_mem_device_mib()
        floors = process_floors(proc.pid)
        point["rss_mib"] = floors.get("rss_mib")
        point["thread_count"] = floors.get("thread_count")
        point["fd_count"] = floors.get("fd_count")
        samples: list[tuple[float, str]] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=sample_mem_during, args=(proc.pid, baseline, stop, samples), daemon=True
        )
        sampler.start()
        conc = max(1, int(concurrency))
        before_metrics = fetch_metrics_text(port, auth_token)
        # Worker sweep only. Decode vote uses load_pinned_rps and its own warmup.
        sweep_warmup_s = 5.0
        # duration+10 is 70s at duration=60 and cuts off the 5s warmup.
        sweep_timeout_s = max(30.0, duration_s + sweep_warmup_s + 45.0)
        report = run_load(
            LoadConfig(
                url=infer_url,
                image_path=image,
                concurrency=conc,
                duration_s=duration_s,
                warmup_s=sweep_warmup_s,
                timeout_s=sweep_timeout_s,
                token=auth_token,
                raw=True,
                progress=False,
            )
        )
        after_metrics = fetch_metrics_text(port, auth_token)
        stop.set()
        sampler.join(timeout=2)
        if samples:
            peak = max(v for v, _s in samples)
            if any(s.startswith("nvml_") for _v, s in samples):
                point["gpu_mem_source"] = next(s for _v, s in samples if s.startswith("nvml_"))
            elif ready_scope == "device_delta":
                point["gpu_mem_source"] = "device_delta"
        else:
            peak = ready_mem
        overload = int((report.by_class or {}).get("overload", 0))
        total = int(report.requests)
        non_429 = max(0, int(report.errors) - overload)

        def _before_hist(name: str) -> dict:
            series = _metric_series(before_metrics or "", name, uri)
            if series is None:
                return {"buckets": [], "count": 0.0}
            return series

        queue_delta = _hist_delta(
            _before_hist("mortred_queue_wait_duration_ms"),
            _metric_series(after_metrics or "", "mortred_queue_wait_duration_ms", uri),
        )
        infer_delta = _hist_delta(
            _before_hist("mortred_inference_duration_ms"),
            _metric_series(after_metrics or "", "mortred_inference_duration_ms", uri),
        )
        queue_p50 = histogram_p50(queue_delta)
        infer_p50 = histogram_p50(infer_delta)
        point["gpu_mem_mib_peak"] = peak
        point["rps"] = round(report.rps, 3)
        point["ok"] = report.ok
        point["errors"] = report.errors
        point["requests"] = total
        point["overload"] = overload
        point["error_rate"] = round((non_429 / total) if total else 0.0, 6)
        point["overload_rate"] = round((overload / total) if total else 0.0, 6)
        point["p50_ms"] = round(report.latency_ms.get("p50") or 0.0, 3)
        point["p99_ms"] = round(report.latency_ms.get("p99") or 0.0, 3)
        point["queue_p50_ms"] = None if queue_p50 is None else round(queue_p50, 3)
        point["inference_p50_ms"] = None if infer_p50 is None else round(infer_p50, 3)
        point["concurrency"] = conc
        queue_count = 0.0 if queue_delta is None else float(queue_delta["count"])
        infer_count = 0.0 if infer_delta is None else float(infer_delta["count"])
        point["metrics_missing"] = bool(
            report.ok > 0 and (queue_count < report.ok / 2.0 or infer_count < report.ok / 2.0)
        )
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        point["oom"] = log_looks_oom(text)
        return point
    finally:
        stop_proc(proc)
        wait_port_free(port)


def pack_model_config(pack: Path, model_id: str, root: Path) -> str:
    from pack_trt import pack_model_config as _pmc

    path = _pmc(pack, model_id, root)
    return str(path) if path is not None else ""


def resolve_pack_server(root: Path, pack: Path, model_id: str) -> Path | None:
    path = pack_server_config(pack, model_id, root)
    if path is not None and path.is_file():
        return path
    return find_server_toml(root, model_id)


def _sweep_memory(row: dict, workers: int) -> float:
    for point in row.get("points") or []:
        if int(point.get("worker_nums") or 0) == workers and isinstance(
            point.get("gpu_mem_mib_peak"), (int, float)
        ):
            return float(point["gpu_mem_mib_peak"])
    return 0.0


def measure_joint_once(
    root: Path,
    binary: Path,
    pack: Path,
    auth_token: str,
    worker_map: dict[str, int],
) -> dict:
    """Start eligible ids together and read memory. Does not send inference load."""
    procs: list[tuple[str, subprocess.Popen[bytes], int]] = []
    per_model: dict[str, dict] = {}
    port_blocked = False
    oom = False
    start_failed = False
    baseline = gpu_mem_device_mib()
    try:
        for model_id, workers in worker_map.items():
            found = resolve_pack_server(root, pack, model_id)
            if found is None:
                per_model[model_id] = {"started": False, "error": "no conf/server mapping"}
                start_failed = True
                break
            log_path = root / "logs" / ("calibrate-joint-%s.log" % model_id)
            override = pack_model_config(pack, model_id, root)
            extra = {}
            if override:
                extra["MORTRED_MODEL_CONFIG_FILE"] = override
            port, _uri = server_listen(found)
            blocked = listen_port_blocked(port)
            if blocked:
                per_model[model_id] = {
                    "started": False,
                    "port_blocked": True,
                    "error": "port %d already in use (%s)" % (port, blocked),
                }
                port_blocked = True
                break
            proc = start_model(root, binary, model_id, found, workers, log_path, extra, auth_token)
            ready_url = "http://127.0.0.1:%d/ready" % port
            if not wait_http_ready(ready_url, 90.0, proc):
                text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
                is_oom = log_looks_oom(text)
                per_model[model_id] = {
                    "started": False,
                    "oom": is_oom,
                    "error": "not ready",
                }
                oom = oom or is_oom
                start_failed = start_failed or not is_oom
                stop_proc(proc)
                wait_port_free(port)
                break
            procs.append((model_id, proc, port))
            mem, src = gpu_mem_process(proc.pid)
            per_model[model_id] = {
                "started": True,
                "pid": proc.pid,
                "gpu_mem_mib": mem,
                "gpu_mem_source": src,
            }
        nvml_sum = 0.0
        nvml_n = 0
        for row in per_model.values():
            if str(row.get("gpu_mem_source") or "").startswith("nvml_") and isinstance(
                row.get("gpu_mem_mib"), (int, float)
            ):
                nvml_sum += float(row["gpu_mem_mib"])
                nvml_n += 1
        started = [row for row in per_model.values() if row.get("started")]
        result: dict = {
            "per_model": per_model,
            "port_blocked": port_blocked,
            "oom": oom,
            "start_failed": start_failed,
            "gpu_mem_mib_sum": None,
            "gpu_mem_pack_delta_mib": None,
        }
        if started and nvml_n == len(started) and nvml_n == len(worker_map):
            result["gpu_mem_mib_sum"] = nvml_sum
        else:
            after = gpu_mem_device_mib()
            if baseline is not None and after is not None:
                result["gpu_mem_pack_delta_mib"] = max(0.0, after - baseline)
            result["gpu_mem_note"] = (
                "per-model NVML pid rows unavailable; pack_delta is "
                "device.used now minus device.used before the joint spawn, "
                "not a sum of isolated model footprints"
            )
        return result
    finally:
        for _model_id, proc, port in procs:
            stop_proc(proc)
            wait_port_free(port)


def _joint_sizes(rows: dict[str, dict], measured: dict, current: dict[str, int]) -> dict[str, float]:
    sizes: dict[str, float] = {}
    per_model = measured.get("per_model") or {}
    for model_id, workers in current.items():
        live = (per_model.get(model_id) or {}).get("gpu_mem_mib")
        if isinstance(live, (int, float)):
            sizes[model_id] = float(live)
        else:
            sizes[model_id] = _sweep_memory(rows[model_id], workers)
    return sizes


def _used_mib(measured: dict) -> float | None:
    total = measured.get("gpu_mem_mib_sum")
    if isinstance(total, (int, float)):
        return float(total)
    delta = measured.get("gpu_mem_pack_delta_mib")
    if isinstance(delta, (int, float)):
        return float(delta)
    return None


def calibrate(
    root: Path,
    pack: Path,
    cli_workers: list[int] | None,
    duration_s: float,
    skip_joint: bool,
) -> dict:
    binary = find_server_bin(root)
    if binary is None:
        raise SystemExit("mortred-model-server.out not found under _bin/ or bin/")
    auth_token = calib_auth_token()
    ids = pack_ids(pack)
    if not ids:
        raise SystemExit("pack has no [pack.<ID>] tables: %s" % pack)
    gpu = gpu_inventory()
    report: dict = {
        "pack": str(pack),
        "gpu": gpu,
        "duration_s": duration_s,
        "calibrator_version": CALIBRATOR_VERSION,
        "release_measurement": is_release_duration(duration_s),
        "workers_swept": cli_workers,
        "wrote_conf_server": False,
        "wrote_pack": False,
        "models": [],
        "joint": None,
    }
    by_id: dict[str, dict] = {}
    for model_id in ids:
        entry = catalog_entry(root, model_id)
        image = resolve_demo_image(entry, root)
        if image is None:
            report["models"].append({"id": model_id, "error": "no demo image", "eligible": False})
            continue
        sweep = workers_for_id(root, model_id, cli_workers, pack)
        cap = p99_cap_ms(model_id, pack)
        print("== %s image=%s sweep=%s ==" % (model_id, image.relative_to(root), sweep), flush=True)
        override = pack_model_config(pack, model_id, root)
        server_toml = resolve_pack_server(root, pack, model_id)
        print(
            "  model_config=%s server=%s p99_cap_ms=%s"
            % (override or "(product)", server_toml, cap),
            flush=True,
        )
        points: list[dict] = []
        for w in sweep:
            conc = w * 2
            print("  w=%d concurrency=%d ..." % (w, conc), flush=True)
            point = run_one_point(
                root, binary, entry, image, w, duration_s, conc, override, auth_token, server_toml
            )
            points.append(point)
            print(
                "    started=%s rps=%s p99=%s queue_p50=%s infer_p50=%s mem_peak=%s oom=%s"
                % (
                    point.get("started"),
                    point.get("rps"),
                    point.get("p99_ms"),
                    point.get("queue_p50_ms"),
                    point.get("inference_p50_ms"),
                    point.get("gpu_mem_mib_peak"),
                    point.get("oom"),
                ),
                flush=True,
            )
            _w, why, _eligible, _passed = pick_w_star(points, cap)
            if why not in SCALE_NOTES:
                print("    stop %s" % why, flush=True)
                break
        w_star, why, eligible, passed = pick_w_star(points, cap)
        row = {
            "id": model_id,
            "image": str(image.relative_to(root)),
            "points": points,
            "workers_swept": sweep,
            "p99_cap_ms": cap,
            "suggested_worker_nums": w_star,
            "final_worker_nums": w_star,
            "passed_worker_nums": passed,
            "eligible": eligible,
            "reason": why,
        }
        report["models"].append(row)
        by_id[model_id] = row

    def apply_joint(state: dict) -> None:
        eligible_rows = [row for row in state["models"] if row.get("eligible") and row.get("id") in by_id]
        for row in state["models"]:
            if row.get("final_worker_nums") is None and row.get("suggested_worker_nums") is not None:
                row["final_worker_nums"] = row["suggested_worker_nums"]
        if skip_joint:
            state["joint"] = {
                "joint_ok": True,
                "joint_adjusted": False,
                "attempts": [],
                "skipped": True,
            }
            return
        if len(eligible_rows) < 2:
            state["joint"] = {
                "joint_ok": True,
                "joint_adjusted": False,
                "attempts": [],
                "note": "single_or_none",
            }
            return
        budget = joint_budget_mib(pack, state.get("gpu") if isinstance(state.get("gpu"), dict) else None)
        current = {row["id"]: int(row["suggested_worker_nums"]) for row in eligible_rows}
        grids = {row["id"]: list(row["workers_swept"]) for row in eligible_rows}
        passed = {row["id"]: set(int(w) for w in row["passed_worker_nums"]) for row in eligible_rows}
        attempts: list[dict] = []
        joint_adjusted = False
        joint_ok = False
        outcome = "round_limit"
        measured: dict = {}
        for round_index in range(len(eligible_rows)):
            measured = measure_joint_once(root, binary, pack, auth_token, current)
            used = _used_mib(measured)
            over_budget = budget is not None and used is not None and used > budget
            over = bool(measured.get("oom")) or over_budget
            attempt = {
                "worker_nums": dict(current),
                "gpu_mem_mib_sum": measured.get("gpu_mem_mib_sum"),
                "gpu_mem_pack_delta_mib": measured.get("gpu_mem_pack_delta_mib"),
                "oom": bool(measured.get("oom")),
                "port_blocked": bool(measured.get("port_blocked")),
                "over_budget": over_budget,
            }
            if measured.get("port_blocked"):
                attempt["demoted"] = None
                attempts.append(attempt)
                outcome = "port_blocked"
                break
            if measured.get("start_failed") and not measured.get("oom"):
                attempt["demoted"] = None
                attempts.append(attempt)
                outcome = "start_failed"
                break
            if not over:
                attempt["demoted"] = None
                attempts.append(attempt)
                joint_ok = True
                outcome = "fit"
                break
            if round_index + 1 >= len(eligible_rows):
                attempt["demoted"] = None
                attempts.append(attempt)
                outcome = "round_limit"
                for row in eligible_rows:
                    row["eligible"] = False
                break
            step = pick_demotion(
                _joint_sizes(by_id, measured, current), current, grids, passed
            )
            if step is None:
                attempt["demoted"] = None
                attempts.append(attempt)
                outcome = "no_lower_step"
                for row in eligible_rows:
                    row["eligible"] = False
                break
            model_id, target = step
            attempt["demoted"] = {"id": model_id, "to": target}
            attempts.append(attempt)
            current[model_id] = target
            joint_adjusted = True
        if joint_ok:
            for row in eligible_rows:
                row["final_worker_nums"] = current[row["id"]]
        state["joint"] = {
            "joint_ok": joint_ok,
            "joint_adjusted": joint_adjusted,
            "attempts": attempts,
            "outcome": outcome,
            "budget_mib": budget,
            "worker_nums": dict(current),
            "per_model": measured.get("per_model") or {},
        }
        if measured.get("gpu_mem_note"):
            state["joint"]["gpu_mem_note"] = measured["gpu_mem_note"]

    def apply_decode(state: dict) -> None:
        for row in state["models"]:
            model_id = row.get("id")
            if not row.get("eligible") or model_id not in by_id:
                continue
            if not any(point.get("started") for point in row.get("points") or []):
                continue
            entry = catalog_entry(root, model_id)
            final_w = int(row["final_worker_nums"])
            override = pack_model_config(pack, model_id, root)
            server_toml = resolve_pack_server(root, pack, model_id)
            print("== %s decode vote at w=%d ==" % (model_id, final_w), flush=True)
            grid_dir = Path(tempfile.mkdtemp(prefix="mortred-decode-grid-"))
            try:
                row["decode"] = calibrate_decode_vote(
                    root,
                    binary,
                    entry,
                    final_w,
                    DECODE_VOTE_DURATION_S,
                    auth_token,
                    grid_dir,
                    override,
                    server_toml,
                )
            except Exception as exc:
                row["decode"] = {
                    "decode_auto": "cpu",
                    "cpu_wins": 0,
                    "gpu_wins": 0,
                    "valid": 0,
                    "error": "decode vote failed: %s" % exc,
                }
                print("  decode vote failed: %s" % exc, flush=True)
            finally:
                shutil.rmtree(grid_dir, ignore_errors=True)

    run_post_sweep(report, apply_joint, apply_decode)
    return report


def _qualified_point(workers: int, rps: float, **overrides: object) -> dict:
    point = {
        "worker_nums": workers,
        "started": True,
        "oom": False,
        "ok": 100,
        "requests": 100,
        "errors": 0,
        "overload": 0,
        "rps": rps,
        "p99_ms": 20.0,
        "queue_p50_ms": 1.0,
        "inference_p50_ms": 10.0,
        "metrics_missing": False,
    }
    point.update(overrides)
    return point


def self_test() -> int:
    points = [_qualified_point(1, 10.0), _qualified_point(2, 19.0), _qualified_point(4, 20.0)]
    w, why, eligible, _passed = pick_w_star(points, 100.0)
    if w != 2 or why != "rps_gain_below_10pct" or not eligible:
        print("self-test: expected w*=2 rps_gain_below_10pct, got", w, why, eligible, file=sys.stderr)
        return 1
    latency_pts = [
        _qualified_point(1, 10.0, p99_ms=20.0),
        _qualified_point(2, 19.0, p99_ms=25.0),
        _qualified_point(4, 40.0, p99_ms=80.0),
    ]
    w, why, eligible, _passed = pick_w_star(latency_pts, 50.0)
    if w != 2 or why != "latency_budget" or not eligible:
        print("self-test: expected latency stop at 2, got", w, why, eligible, file=sys.stderr)
        return 1
    queue_pts = [
        _qualified_point(1, 10.0),
        _qualified_point(2, 19.0),
        _qualified_point(4, 40.0, queue_p50_ms=30.0, inference_p50_ms=10.0),
    ]
    w, why, eligible, _passed = pick_w_star(queue_pts, 100.0)
    if w != 2 or why != "queue_dominates" or not eligible:
        print("self-test: expected queue_dominates at 2, got", w, why, eligible, file=sys.stderr)
        return 1
    error_pts = [_qualified_point(1, 10.0, ok=98, errors=2, overload=0)]
    w, why, eligible, _passed = pick_w_star(error_pts, 100.0)
    if w != 1 or why != "error_rate" or eligible:
        print("self-test: expected error_rate, got", w, why, eligible, file=sys.stderr)
        return 1
    overload_pts = [_qualified_point(1, 10.0, ok=98, errors=2, overload=2)]
    w, why, eligible, _passed = pick_w_star(overload_pts, 100.0)
    if w != 1 or why != "overload" or eligible:
        print("self-test: expected overload, not error_rate, got", w, why, eligible, file=sys.stderr)
        return 1
    only_w1 = [_qualified_point(1, 10.0, p99_ms=80.0)]
    w, why, eligible, _passed = pick_w_star(only_w1, 50.0)
    if w != 1 or why != "latency_budget" or eligible:
        print("self-test: expected ineligible w=1, got", w, why, eligible, file=sys.stderr)
        return 1
    oom_pts = [
        _qualified_point(1, 5.0, ok=50, requests=50),
        {"worker_nums": 2, "started": False, "oom": True, "ok": 0, "requests": 0, "rps": 0.0},
    ]
    w, why, eligible, _passed = pick_w_star(oom_pts)
    if w != 1 or why != "oom_or_start_failed" or not eligible:
        print("self-test: expected w*=1 on oom, got", w, why, eligible, file=sys.stderr)
        return 1
    fail_pts = [
        {"worker_nums": 1, "started": False, "oom": False, "ok": 0, "requests": 0, "rps": 0.0},
    ]
    w, why, eligible, _passed = pick_w_star(fail_pts)
    if w != 1 or why != "start_failed" or eligible:
        print("self-test: expected start_failed, got", w, why, eligible, file=sys.stderr)
        return 1
    huge = [
        _qualified_point(1, 10.0, gpu_mem_mib_peak=7000.0),
        _qualified_point(2, 19.0, gpu_mem_mib_peak=7500.0),
        _qualified_point(4, 20.0, gpu_mem_mib_peak=7900.0),
    ]
    w, why, _eligible, _passed = pick_w_star(huge, 100.0)
    if w != 2 or why != "rps_gain_below_10pct":
        print("self-test: memory ratio must not stop the sweep, got", w, why, file=sys.stderr)
        return 1
    if 16 not in default_worker_sweep(ROOT, "superpoint"):
        print("self-test: superpoint sweep missing 16", default_worker_sweep(ROOT, "superpoint"), file=sys.stderr)
        return 1
    if default_worker_sweep(ROOT, "no-such-model") != OTHER_SWEEP:
        print("self-test: missing benchmark sweep", default_worker_sweep(ROOT, "no-such-model"), file=sys.stderr)
        return 1
    if workers_for_id(ROOT, "superpoint", [1, 2, 4], None) != [1, 2, 4]:
        print("self-test: CLI workers did not override", file=sys.stderr)
        return 1
    if not is_release_duration(30.0) or is_release_duration(8.0):
        print("self-test: release duration", file=sys.stderr)
        return 1
    demoted = pick_demotion(
        {"big": 3000.0, "small": 1000.0},
        {"big": 4, "small": 4},
        {"big": [1, 2, 4], "small": [1, 2, 4]},
        {"big": {1, 2, 4}, "small": {1, 2, 4}},
    )
    if demoted != ("big", 2):
        print("self-test: pick_demotion", demoted, file=sys.stderr)
        return 1
    if (
        pick_demotion(
            {"big": 3000.0, "small": 1000.0},
            {"big": 1, "small": 1},
            {"big": [1, 2, 4], "small": [1, 2, 4]},
            {"big": {1}, "small": {1}},
        )
        is not None
    ):
        print("self-test: lowest grid must not demote", file=sys.stderr)
        return 1
    phases: list[str] = []
    run_post_sweep({}, lambda _state: phases.append("joint"), lambda _state: phases.append("decode"))
    if phases != ["joint", "decode"]:
        print("self-test: phase order", phases, file=sys.stderr)
        return 1
    if log_looks_oom("CUDA out of memory") is False:
        print("self-test: oom marker missed", file=sys.stderr)
        return 1
    if _csv_float("N/A") is not None or _csv_float("512") != 512.0:
        print("self-test: _csv_float", file=sys.stderr)
        return 1
    if occupancy_from_device(4096.0, 1024.0) != 3072.0:
        print("self-test: occupancy_from_device", file=sys.stderr)
        return 1
    if occupancy_from_device(800.0, 1024.0) != 0.0:
        print("self-test: occupancy_from_device clamp", file=sys.stderr)
        return 1
    cpu_majority = vote_decode_auto(
        [{"rps_cpu": 300.0, "rps_gpu": 200.0}] * 13
        + [{"rps_cpu": 200.0, "rps_gpu": 300.0}] * 12
    )
    if cpu_majority["decode_auto"] != "cpu" or cpu_majority["cpu_wins"] != 13:
        print("self-test: vote cpu majority", cpu_majority, file=sys.stderr)
        return 1
    gpu_majority = vote_decode_auto(
        [{"rps_cpu": 200.0, "rps_gpu": 300.0}] * 13
        + [{"rps_cpu": 300.0, "rps_gpu": 200.0}] * 12
    )
    if gpu_majority["decode_auto"] != "gpu" or gpu_majority["gpu_wins"] != 13:
        print("self-test: vote gpu majority", gpu_majority, file=sys.stderr)
        return 1
    empty_vote = vote_decode_auto([{"rps_cpu": None, "rps_gpu": 1.0}] * 16)
    if empty_vote["decode_auto"] != "cpu" or empty_vote["valid"] != 0:
        print("self-test: vote empty", empty_vote, file=sys.stderr)
        return 1
    from repo_toml import load_toml

    cfg = ROOT / "conf" / "server" / "classification" / "mobilenetv2" / "mobilenetv2_server_config.toml"
    table = load_toml(cfg)
    server = table.get("MOBILENETV2_CLASSIFICATION_SERVER") or {}
    if int(server.get("port") or 0) != 9002:
        print("self-test: expected unquoted port=9002, got", server.get("port"), file=sys.stderr)
        return 1
    port, uri = server_listen(cfg)
    if port != 9002 or not uri.startswith("/"):
        print("self-test: server_listen failed", port, uri, file=sys.stderr)
        return 1
    tmp = Path(tempfile.mkdtemp()) / "pack.toml"
    tmp.write_text(
        "# machine pack\n[pack.MOBILENETV2]\nworker_nums = 1\n\n[pack.YOLOV8]\n",
        encoding="utf-8",
    )
    fake = {
        "models": [
            {
                "id": "MOBILENETV2",
                "suggested_worker_nums": 4,
                "reason": "rps_gain_below_10pct",
                "points": [{"started": True}],
            },
            {
                "id": "YOLOV8",
                "suggested_worker_nums": 2,
                "reason": "single_point",
                "points": [{"started": True}],
            },
            {
                "id": "SKIPME",
                "suggested_worker_nums": 8,
                "reason": "start_failed",
                "points": [{"started": False}],
            },
        ]
    }
    updates = pack_writes_from_report(fake)
    if updates != {"MOBILENETV2": 4, "YOLOV8": 2}:
        print("self-test: pack_writes_from_report", updates, file=sys.stderr)
        return 1
    applied = write_pack_worker_nums(tmp, updates)
    body = tmp.read_text(encoding="utf-8")
    if "worker_nums = 4" not in body or "worker_nums = 2" not in body:
        print("self-test: pack write body", body, file=sys.stderr)
        return 1
    if any(a.get("error") for a in applied):
        print("self-test: unexpected pack write error", applied, file=sys.stderr)
        return 1
    occ = Path(tempfile.mkdtemp()) / "occ.toml"
    occ.write_text("[pack.YOLOV8]\nworker_nums = 1\n", encoding="utf-8")
    fake_occ = {
        "gpu": {"name": "Fake GPU", "memory_total_mib": 8192.1},
        "models": [
            {
                "id": "YOLOV8",
                "suggested_worker_nums": 2,
                "reason": "single_point",
                "points": [
                    {
                        "started": True,
                        "worker_nums": 2,
                        "gpu_mem_mib_peak": 1800.2,
                        "gpu_mem_source": "nvml_pid",
                    }
                ],
                "decode": {"decode_auto": "gpu", "cpu_wins": 6, "gpu_wins": 10, "valid": 16},
            }
        ],
    }
    id_fields, pack_fields = occupancy_fields_from_report(fake_occ)
    write_pack_fields(occ, id_fields, pack_fields)
    occ_body = occ.read_text(encoding="utf-8")
    if "gpu_mem_mib = 1801" not in occ_body or "gpu_mem_at_workers = 2" not in occ_body:
        print("self-test: occupancy stamp body", occ_body, file=sys.stderr)
        return 1
    if 'gpu_mem_source = "nvml_pid"' not in occ_body or 'gpu_mem_source = "seed"' in occ_body:
        print("self-test: occupancy source must be positive non-seed", occ_body, file=sys.stderr)
        return 1
    if 'gpu_name = "Fake GPU"' not in occ_body or "gpu_memory_total_mib = 8193" not in occ_body:
        print("self-test: gpu fingerprint body", occ_body, file=sys.stderr)
        return 1
    if 'decode_auto = "gpu"' not in occ_body:
        print("self-test: decode_auto stamp body", occ_body, file=sys.stderr)
        return 1
    example_pack = ROOT / "conf" / "packs" / "yolov8.toml"
    example_before = example_pack.read_text(encoding="utf-8") if example_pack.is_file() else ""
    try:
        write_pack_worker_nums(ROOT / "conf" / "packs" / "demo.toml", {"mobilenetv2": 4})
        print("self-test: should refuse conf/packs", file=sys.stderr)
        return 1
    except ValueError:
        pass
    if example_pack.is_file() and example_pack.read_text(encoding="utf-8") != example_before:
        print("self-test: conf/packs/yolov8.toml mutated", file=sys.stderr)
        return 1

    try:
        write_pack_worker_nums(cfg, {"MOBILENETV2": 4})
        print("self-test: should refuse conf/server", file=sys.stderr)
        return 1
    except ValueError:
        pass
    if int(load_toml(cfg)["MOBILENETV2_CLASSIFICATION_SERVER"].get("worker_nums") or 0) != 1:
        print("self-test: conf/server worker_nums mutated", file=sys.stderr)
        return 1
    blocked = Path(tempfile.mkdtemp()) / "blocked.toml"
    blocked.write_text("[pack.YOLOV8]\nworker_nums = 1\n", encoding="utf-8")
    blocked_before = blocked.read_text(encoding="utf-8")
    blocked_fields, blocked_pack = occupancy_fields_from_report(
        {
            "models": [
                {
                    "id": "YOLOV8",
                    "eligible": False,
                    "suggested_worker_nums": 1,
                    "final_worker_nums": 1,
                    "reason": "latency_budget",
                    "points": [{"started": True, "worker_nums": 1}],
                }
            ]
        }
    )
    if blocked_fields or blocked_pack:
        print("self-test: ineligible row produced pack fields", blocked_fields, file=sys.stderr)
        return 1
    write_pack_fields(blocked, blocked_fields, blocked_pack)
    if blocked.read_text(encoding="utf-8") != blocked_before:
        print("self-test: ineligible write changed pack", file=sys.stderr)
        return 1
    short = occupancy_fields_from_report(
        {
            "release_measurement": False,
            "models": [
                {
                    "id": "YOLOV8",
                    "eligible": True,
                    "suggested_worker_nums": 4,
                    "points": [{"started": True, "worker_nums": 4}],
                }
            ],
        }
    )
    if short != ({}, {}):
        print("self-test: short run wrote fields", short, file=sys.stderr)
        return 1
    final_report = {
        "models": [
            {
                "id": "BIG",
                "eligible": True,
                "suggested_worker_nums": 4,
                "final_worker_nums": 2,
                "reason": "rps_gain_below_10pct",
                "points": [
                    {
                        "started": True,
                        "worker_nums": 2,
                        "gpu_mem_mib_peak": 1000,
                        "gpu_mem_source": "nvml_pid",
                    }
                ],
            },
            {
                "id": "SMALL",
                "eligible": True,
                "suggested_worker_nums": 4,
                "final_worker_nums": 4,
                "reason": "approx_linear",
                "points": [{"started": True, "worker_nums": 4}],
            },
        ]
    }
    if pack_writes_from_report(final_report) != {"BIG": 2, "SMALL": 4}:
        print("self-test: final worker write", pack_writes_from_report(final_report), file=sys.stderr)
        return 1
    final_pack = Path(tempfile.mkdtemp()) / "final.toml"
    final_pack.write_text(
        "[pack.BIG]\nworker_nums = 1\n\n[pack.SMALL]\nworker_nums = 1\n",
        encoding="utf-8",
    )
    id_fields, _pack_fields = occupancy_fields_from_report(final_report)
    write_pack_fields(final_pack, id_fields, {})
    final_body = final_pack.read_text(encoding="utf-8")
    if "worker_nums = 2" not in final_body or "worker_nums = 4" not in final_body:
        print("self-test: demoted pack body", final_body, file=sys.stderr)
        return 1
    if pack_writes_from_report(
        {
            "models": [
                {
                    "id": "BIG",
                    "eligible": False,
                    "suggested_worker_nums": 1,
                    "final_worker_nums": 1,
                    "points": [{"started": True, "worker_nums": 1}],
                }
            ]
        }
    ):
        print("self-test: unfit joint wrote workers", file=sys.stderr)
        return 1
    uri = "/mortred_ai_server_v1/classification/mobilenetv2"
    metrics = (
        'mortred_queue_wait_duration_ms_bucket{model="%s",le="1"} 4\n'
        'mortred_queue_wait_duration_ms_bucket{model="%s",le="+Inf"} 4\n'
        'mortred_queue_wait_duration_ms_count{model="%s"} 4\n'
    ) % (uri, uri, uri)
    series = _metric_series(metrics, "mortred_queue_wait_duration_ms", uri)
    if series is None or series["count"] != 4.0:
        print("self-test: server_uri histogram", series, file=sys.stderr)
        return 1
    if _metric_series(metrics, "mortred_queue_wait_duration_ms", "mobilenetv2") is not None:
        print("self-test: directory id must not match the histogram", file=sys.stderr)
        return 1
    delta = _hist_delta({"buckets": [], "count": 0.0}, series)
    if delta is None or delta["count"] != series["count"]:
        print("self-test: empty before delta", delta, file=sys.stderr)
        return 1
    if _hist_delta(series, None) is not None:
        print("self-test: empty after must stay empty", file=sys.stderr)
        return 1
    print("calibrate_pack.py self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="machine pack toml")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--workers",
        default="",
        help="comma-separated worker_nums for every id; empty uses the per-id default grid",
    )
    parser.add_argument("--duration", default="30s")
    parser.add_argument("--output", type=Path, help="write JSON report")
    parser.add_argument(
        "--from-json",
        type=Path,
        help="load an existing calibrate JSON instead of running the sweep",
    )
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument(
        "--write-pack",
        action="store_true",
        help="write final worker_nums, occupancy stamps, and decode_auto into the pack file (never conf/server; skipped when duration < 30s)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.pack is None:
        parser.error("--pack is required unless --self-test")
    pack = args.pack if args.pack.is_absolute() else args.project_root / args.pack
    cli_workers = parse_worker_list(args.workers)
    duration_s = parse_duration(args.duration)
    if args.from_json:
        src = args.from_json if args.from_json.is_absolute() else args.project_root / args.from_json
        report = json.loads(src.read_text(encoding="utf-8"))
    else:
        report = calibrate(args.project_root, pack, cli_workers, duration_s, args.skip_joint)
    if args.write_pack and report.get("release_measurement") is False:
        report["wrote_pack"] = False
        report["pack_writes"] = []
        print("duration < 30s; not writing pack", file=sys.stderr)
    elif args.write_pack:
        try:
            id_fields, pack_fields = occupancy_fields_from_report(report)
            applied = write_pack_fields(pack, id_fields, pack_fields)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        report["wrote_pack"] = any(a.get("action") for a in applied if not a.get("error"))
        report["pack_writes"] = applied
        if report["wrote_pack"]:
            print("wrote occupancy stamps into %s" % pack, file=sys.stderr)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % args.output, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
