#!/usr/bin/env python3
"""Calibrate worker_nums for a machine pack. Report only; never writes conf/server.

For each pack id, sweep worker_nums, POST with http_infer_rps, and sample the
process GPU memory via nvidia-smi. Suggests w* from the RPS curve. A joint
pass then starts every id at its suggested w and records residency.

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


def gpu_mem_mib_for_pid(pid: int) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    peak = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) != pid:
                continue
        except ValueError:
            continue
        val = _csv_float(parts[1])
        if val is None:
            continue
        peak = val if peak is None else max(peak, val)
    return peak


def gpu_mem_sample(pid: int) -> tuple[float | None, str]:
    proc = gpu_mem_mib_for_pid(pid)
    if proc is not None:
        return proc, "process"
    device = gpu_mem_device_mib()
    if device is not None:
        return device, "device"
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
            return (int(last_good["worker_nums"]) if last_good else 1, "oom_or_start_failed")
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


def sample_mem_during(pid: int, stop: threading.Event, bucket: list[tuple[float, str]]) -> None:
    while not stop.is_set():
        val, scope = gpu_mem_sample(pid)
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
            point["error"] = "not ready (exit=%s) probing %s" % (proc.poll(), ready_url)
            if tail:
                point["log_tail"] = tail
            return point
        point["started"] = True
        ready_mem, ready_scope = gpu_mem_sample(proc.pid)
        point["gpu_mem_mib_ready"] = ready_mem
        point["gpu_mem_scope"] = ready_scope
        samples: list[tuple[float, str]] = []
        stop = threading.Event()
        sampler = threading.Thread(target=sample_mem_during, args=(proc.pid, stop, samples), daemon=True)
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
            if any(s == "process" for _v, s in samples):
                point["gpu_mem_scope"] = "process"
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
                "    started=%s rps=%s p99=%s mem_peak=%s oom=%s"
                % (
                    point.get("started"),
                    point.get("rps"),
                    point.get("p99_ms"),
                    point.get("gpu_mem_mib_peak"),
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

    if skip_joint or len(suggested) < 1:
        return report

    print("== joint residency at suggested w ==")
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    joint: dict = {"worker_nums": dict(suggested), "per_model": {}}
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
            proc = start_model(root, binary, model_id, found, w, log_path, extra)
            port, _uri = server_listen(found)
            ready_url = "http://127.0.0.1:%d/ready" % port
            if not wait_http_ready(ready_url, 90.0, proc):
                joint["per_model"][model_id] = {"started": False, "error": "not ready"}
                stop_proc(proc)
                continue
            procs.append((model_id, proc))
            mem, scope = gpu_mem_sample(proc.pid)
            joint["per_model"][model_id] = {
                "started": True,
                "pid": proc.pid,
                "gpu_mem_mib": mem,
                "gpu_mem_scope": scope,
            }
        total = 0.0
        any_mem = False
        for row in joint["per_model"].values():
            mem = row.get("gpu_mem_mib")
            if isinstance(mem, (int, float)):
                total += float(mem)
                any_mem = True
        joint["gpu_mem_mib_sum"] = total if any_mem else None
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
    if log_looks_oom("CUDA out of memory") is False:
        print("self-test: oom marker missed", file=sys.stderr)
        return 1
    if _csv_float("N/A") is not None or _csv_float("512") != 512.0:
        print("self-test: _csv_float", file=sys.stderr)
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
