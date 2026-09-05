#!/usr/bin/env python3
"""Calibrate worker_nums for a machine pack. Report only; never writes conf/server.

For each pack id, sweep worker_nums, POST with http_infer_rps, and sample GPU
memory for that server process (NVML compute-apps by pid, else a unique
mortred-model-server name). Whole-card memory.used is never treated as the
model; if WSL has no process row, occupancy is device.used minus a sample
taken before spawn (gpu_mem_source=device_delta). Suggests w* from the RPS
curve. A joint pass starts every id at its suggested w; it sums only real
NVML per-process rows, otherwise reports one pack device delta.

Usage:
  python3 scripts/calibrate_pack.py --pack conf/packs/demo.toml
  python3 scripts/calibrate_pack.py --pack conf/packs/yolov8.toml \\
      --workers 1,2,4 --duration 8s --output logs/calibrate-yolov8.json
  python3 scripts/calibrate_pack.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

from http_infer_rps import LoadConfig, parse_duration, run_load  # noqa: E402
from pack_trt import find_server_toml, pack_ids, server_listen  # noqa: E402
from test_server import build_catalog, resolve_demo_image, resolve_server  # noqa: E402

OOM_MARKERS = ("out of memory", "cuda oom", "cudnn_status_alloc_failed", "std::bad_alloc")


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


def pick_w_star(points: list[dict]) -> tuple[int, str]:
    """Last good worker_nums. OOM / start fail / flat RPS keep the previous w."""
    last_good: dict | None = None
    note = "single_point"
    for point in points:
        if not point.get("started") or point.get("oom"):
            why = "oom_or_start_failed" if point.get("oom") else "start_failed"
            return (int(last_good["worker_nums"]) if last_good else 1, why)
        ok = int(point.get("ok") or 0)
        total = int(point.get("requests") or 0)
        if total > 0 and ok / total < 0.9:
            return (int(last_good["worker_nums"]) if last_good else int(point["worker_nums"]),
                    "error_rate")
        if last_good is not None:
            prev_rps = max(float(last_good.get("rps") or 0.0), 1e-6)
            rps = float(point.get("rps") or 0.0)
            gain = rps / prev_rps
            w_ratio = float(point["worker_nums"]) / float(last_good["worker_nums"])
            if gain < 1.10:
                return int(last_good["worker_nums"]), "rps_gain_below_10pct"
            if gain < 0.5 * w_ratio:
                return int(last_good["worker_nums"]), "saturating"
            note = "approx_linear" if gain >= 0.8 * w_ratio else "partial_scale"
        last_good = point
    return int(last_good["worker_nums"]) if last_good else 1, note


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


def stop_proc(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def start_model(
    root: Path,
    binary: Path,
    model_id: str,
    server_toml: Path,
    workers: int,
    log_path: Path,
    extra_env: dict[str, str],
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("MORTRED_LISTEN_PORT", None)
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
) -> dict:
    model_id = entry["model_id"]
    log_path = root / "logs" / ("calibrate-%s-w%d.log" % (model_id, workers))
    extra = {}
    if model_config:
        extra["MORTRED_MODEL_CONFIG_FILE"] = model_config
    found = find_server_toml(root, model_id)
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
    proc = start_model(root, binary, model_id, found, workers, log_path, extra)
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
        samples: list[tuple[float, str]] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=sample_mem_during, args=(proc.pid, baseline, stop, samples), daemon=True
        )
        sampler.start()
        conc = max(1, concurrency)
        report = run_load(
            LoadConfig(
                url=infer_url,
                image_path=image,
                concurrency=conc,
                duration_s=duration_s,
                warmup_s=min(1.0, duration_s / 5.0),
                timeout_s=max(30.0, duration_s + 10.0),
                progress=False,
            )
        )
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
        point["gpu_mem_mib_peak"] = peak
        point["rps"] = round(report.rps, 3)
        point["ok"] = report.ok
        point["errors"] = report.errors
        point["requests"] = report.requests
        point["p50_ms"] = round(report.latency_ms.get("p50") or 0.0, 3)
        point["p99_ms"] = round(report.latency_ms.get("p99") or 0.0, 3)
        point["concurrency"] = conc
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        point["oom"] = log_looks_oom(text)
        return point
    finally:
        stop_proc(proc)


def pack_model_config(pack: Path, model_id: str, root: Path) -> str:
    from pack_trt import pack_model_config as _pmc

    path = _pmc(pack, model_id, root)
    return str(path) if path is not None else ""


def calibrate(
    root: Path,
    pack: Path,
    workers: list[int],
    duration_s: float,
    skip_joint: bool,
) -> dict:
    binary = find_server_bin(root)
    if binary is None:
        raise SystemExit("mortred-model-server.out not found under _bin/ or bin/")
    ids = pack_ids(pack)
    if not ids:
        raise SystemExit("pack has no [pack.<ID>] tables: %s" % pack)
    report: dict = {
        "pack": str(pack),
        "gpu": gpu_inventory(),
        "duration_s": duration_s,
        "workers_swept": workers,
        "wrote_conf_server": False,
        "models": [],
        "joint": None,
    }
    suggested: dict[str, int] = {}
    for model_id in ids:
        entry = catalog_entry(root, model_id)
        image = resolve_demo_image(entry, root)
        if image is None:
            report["models"].append({"id": model_id, "error": "no demo image"})
            continue
        print("== %s image=%s ==" % (model_id, image.relative_to(root)), flush=True)
        override = pack_model_config(pack, model_id, root)
        points: list[dict] = []
        for w in workers:
            print("  w=%d ..." % w, flush=True)
            conc = max(4, min(16, 4 * w))
            point = run_one_point(
                root, binary, entry, image, w, duration_s, conc, override
            )
            points.append(point)
            print(
                "    started=%s rps=%s p99=%s mem_peak=%s source=%s oom=%s"
                % (
                    point.get("started"),
                    point.get("rps"),
                    point.get("p99_ms"),
                    point.get("gpu_mem_mib_peak"),
                    point.get("gpu_mem_source"),
                    point.get("oom"),
                ),
                flush=True,
            )
            if not point.get("started") or point.get("oom"):
                break
        w_star, why = pick_w_star(points)
        suggested[model_id] = w_star
        report["models"].append(
            {
                "id": model_id,
                "image": str(image.relative_to(root)),
                "points": points,
                "suggested_worker_nums": w_star,
                "reason": why,
            }
        )

    any_started = any(
        any(p.get("started") for p in row.get("points") or [])
        for row in report["models"]
    )
    if skip_joint or not any_started:
        return report

    print("== joint residency at suggested w ==")
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    joint: dict = {"worker_nums": dict(suggested), "per_model": {}}
    baseline = gpu_mem_device_mib()
    try:
        for model_id, w in suggested.items():
            found = find_server_toml(root, model_id)
            if found is None:
                continue
            log_path = root / "logs" / ("calibrate-joint-%s.log" % model_id)
            override = pack_model_config(pack, model_id, root)
            extra = {}
            if override:
                extra["MORTRED_MODEL_CONFIG_FILE"] = override
            port, _uri = server_listen(found)
            blocked = listen_port_blocked(port)
            if blocked:
                joint["per_model"][model_id] = {
                    "started": False,
                    "error": "port %d already in use (%s)" % (port, blocked),
                }
                continue
            proc = start_model(root, binary, model_id, found, w, log_path, extra)
            ready_url = "http://127.0.0.1:%d/ready" % port
            if not wait_http_ready(ready_url, 90.0, proc):
                joint["per_model"][model_id] = {"started": False, "error": "not ready"}
                stop_proc(proc)
                continue
            procs.append((model_id, proc))
            mem, src = gpu_mem_process(proc.pid)
            row = {
                "started": True,
                "pid": proc.pid,
                "gpu_mem_mib": mem,
                "gpu_mem_source": src,
            }
            joint["per_model"][model_id] = row
        nvml_sum = 0.0
        nvml_n = 0
        for row in joint["per_model"].values():
            if row.get("gpu_mem_source", "").startswith("nvml_") and isinstance(
                row.get("gpu_mem_mib"), (int, float)
            ):
                nvml_sum += float(row["gpu_mem_mib"])
                nvml_n += 1
        started = [r for r in joint["per_model"].values() if r.get("started")]
        if started and nvml_n == len(started):
            joint["gpu_mem_mib_sum"] = nvml_sum
        else:
            joint["gpu_mem_mib_sum"] = None
            after = gpu_mem_device_mib()
            joint["gpu_mem_device_used_mib"] = after
            if baseline is not None and after is not None:
                joint["gpu_mem_pack_delta_mib"] = max(0.0, after - baseline)
            joint["gpu_mem_note"] = (
                "per-model NVML pid rows unavailable; pack_delta is "
                "device.used now minus device.used before the joint spawn, "
                "not a sum of isolated model footprints"
            )
        report["joint"] = joint
    finally:
        for _, proc in procs:
            stop_proc(proc)
    return report


def self_test() -> int:
    points = [
        {"worker_nums": 1, "started": True, "oom": False, "ok": 100, "requests": 100, "rps": 10.0},
        {"worker_nums": 2, "started": True, "oom": False, "ok": 100, "requests": 100, "rps": 19.0},
        {"worker_nums": 4, "started": True, "oom": False, "ok": 100, "requests": 100, "rps": 20.0},
    ]
    w, why = pick_w_star(points)
    if w != 2 or why != "rps_gain_below_10pct":
        print("self-test: expected w*=2 saturating, got", w, why, file=sys.stderr)
        return 1
    oom_pts = [
        {"worker_nums": 1, "started": True, "oom": False, "ok": 50, "requests": 50, "rps": 5.0},
        {"worker_nums": 2, "started": False, "oom": True, "ok": 0, "requests": 0, "rps": 0.0},
    ]
    w, why = pick_w_star(oom_pts)
    if w != 1 or why != "oom_or_start_failed":
        print("self-test: expected w*=1 on oom, got", w, why, file=sys.stderr)
        return 1
    fail_pts = [
        {"worker_nums": 1, "started": False, "oom": False, "ok": 0, "requests": 0, "rps": 0.0},
    ]
    w, why = pick_w_star(fail_pts)
    if w != 1 or why != "start_failed":
        print("self-test: expected start_failed, got", w, why, file=sys.stderr)
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
    print("calibrate_pack.py self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="machine pack toml")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--workers", default="1,2,4", help="comma-separated worker_nums sweep")
    parser.add_argument("--duration", default="8s")
    parser.add_argument("--output", type=Path, help="write JSON report")
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.pack is None:
        parser.error("--pack is required unless --self-test")
    pack = args.pack if args.pack.is_absolute() else args.project_root / args.pack
    workers = [int(x) for x in args.workers.split(",") if x.strip()]
    if not workers or min(workers) < 1:
        parser.error("workers must be >= 1")
    duration_s = parse_duration(args.duration)
    report = calibrate(args.project_root, pack, workers, duration_s, args.skip_joint)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % args.output, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
