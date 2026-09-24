#!/usr/bin/env python3
"""Unattended Set A TRT benchmark driver (WSL)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import MODELS, unique_converts  # noqa: E402

LOG = ROOT / "logs" / "bench" / "set_a"
TRTEXEC = ROOT / "3rd_party" / "bin" / "trtexec"
SERVER = ROOT / "_bin" / "mortred-model-server.out"
# trtexec 10.3 suffixes are B/K/M/G. "6GiB" is parsed as 6 bytes because the trailing B wins.
WS = "--memPoolSize=workspace:6G"
STATUS = LOG / "status.json"


def log(msg: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with (LOG / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(**kwargs) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    prev = {}
    if STATUS.is_file():
        try:
            prev = json.loads(STATUS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    prev.update(kwargs)
    prev["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(prev, indent=2), encoding="utf-8")


def trt_env() -> dict[str, str]:
    env = os.environ.copy()
    extra = ["/usr/lib/wsl/lib", str(ROOT / "3rd_party" / "libs")]
    env["LD_LIBRARY_PATH"] = ":".join(extra + [env.get("LD_LIBRARY_PATH", "")])
    return env


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def ensure_jpeg(src: Path) -> Path:
    dest_dir = LOG / "jpeg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (src.stem + ".jpg")
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    raw = src.read_bytes()[:3]
    if raw[:2] == b"\xff\xd8":
        dest.write_bytes(src.read_bytes())
        return dest
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imread failed")
        cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return dest
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        Image.open(src).convert("RGB").save(dest, "JPEG", quality=95)
        return dest
    except Exception as exc:
        log("jpeg convert failed %s: %s; using original" % (src, exc))
        return src


class SmiSampler:
    def __init__(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.handle = csv_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,nounits",
                "-l",
                "1",
            ],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.smi_cmd = " ".join(self.proc.args)  # type: ignore[arg-type]

    def stop(self) -> dict:
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except OSError:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.handle.close()
        utils: list[float] = []
        mems: list[float] = []
        if self.csv_path.is_file():
            with self.csv_path.open(encoding="utf-8", errors="replace") as fh:
                reader = csv.reader(fh)
                header = None
                for row in reader:
                    if not row:
                        continue
                    if header is None:
                        header = [c.strip() for c in row]
                        continue
                    try:
                        # timestamp, index, util.gpu, util.mem, mem.used, power
                        utils.append(float(row[2]))
                        mems.append(float(row[4]))
                    except (IndexError, ValueError):
                        continue
        out = {
            "smi_cmd": self.smi_cmd,
            "smi_samples": len(utils),
            "gpu_util_avg_pct": round(sum(utils) / len(utils), 1) if utils else None,
            "gpu_util_max_pct": max(utils) if utils else None,
            "mem_used_max_mib": max(mems) if mems else None,
        }
        return out


def parse_trtexec(text: str) -> dict:
    out: dict = {}
    m = re.search(r"Throughput:\s+([0-9.]+)\s+qps", text)
    if m:
        out["trtexec_rps"] = float(m.group(1))
    m = re.search(r"GPU Compute Time:.*mean =\s+([0-9.]+)", text)
    if m:
        out["gpu_compute_ms"] = float(m.group(1))
    m = re.search(r"Latency:.*mean =\s+([0-9.]+)", text)
    if m:
        out["latency_mean_ms"] = float(m.group(1))
    out["cuda_graph"] = "yes" if re.search(r"Successfully.*[Cc]uda.?[Gg]raph", text) else "no"
    return out


def shape_flags(profile_rel: str | None) -> list[str]:
    if not profile_rel:
        return []
    path = ROOT / profile_rel
    if not path.is_file():
        return []
    prof = json.loads(path.read_text(encoding="utf-8"))
    mins, opts, maxs = [], [], []
    for b in prof:
        def dims(v: list) -> str:
            return "x".join(str(d) for d in v)
        mins.append("%s:%s" % (b["name"], dims(b["min"])))
        opts.append("%s:%s" % (b["name"], dims(b["opt"])))
        maxs.append("%s:%s" % (b["name"], dims(b["max"])))
    return [
        "--minShapes=" + ",".join(mins),
        "--optShapes=" + ",".join(opts),
        "--maxShapes=" + ",".join(maxs),
    ]


def point_overlays_at_engine(cid: str, engine_rel: str) -> None:
    """Point every HTTP overlay for this convert_id at the engine that actually exists."""
    if not engine_rel:
        return
    rel = engine_rel.replace("\\", "/")
    for other in MODELS:
        if other["convert_id"] != cid or not other["http"]:
            continue
        overlay = ROOT / ("conf/bench/set_a/%s.toml" % other["id"])
        if not overlay.is_file():
            continue
        text = overlay.read_text(encoding="utf-8")
        updated = re.sub(
            r'model_file_path\s*=\s*"[^"]+"',
            'model_file_path = "../%s"' % rel,
            text,
            count=1,
        )
        if updated != text:
            overlay.write_text(updated, encoding="utf-8")


def _invalidate_after_reconvert(cid: str) -> None:
    """Drop profile/calibrate/HTTP artifacts so a rebuilt engine is measured again."""
    for name in ("%s.json" % cid, "%s.times.json" % cid, "%s.profile.json" % cid):
        p = LOG / "trtexec" / name
        if p.is_file():
            p.unlink()
    for other in MODELS:
        if other["convert_id"] != cid:
            continue
        for p in (
            LOG / "calibrate" / ("%s.json" % other["id"]),
            LOG / "calibrate" / ("%s.driver.json" % other["id"]),
            LOG / "http" / ("%s.summary.json" % other["id"]),
        ):
            if p.is_file():
                p.unlink()


def convert_one(row: dict, profiles: dict[str, str | None]) -> dict:
    cid = row["convert_id"]
    onnx = ROOT / row["onnx"]
    engine16 = ROOT / row["engine"]
    engine32 = Path(str(engine16).replace(".fp16", ".fp32"))
    log_path = LOG / "convert" / ("%s.log" % cid)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = LOG / "convert" / ("%s.json" % cid)
    layout = "hwc" if row.get("layout") == "nhwc" else "chw"
    io_flag = "--inputIOFormats=fp16:%s" % layout
    flags = shape_flags(profiles.get(cid))

    def stamp_matches(dest: Path, fp: str) -> dict | None:
        if not stamp.is_file() or not dest.is_file() or dest.stat().st_size <= 0:
            return None
        try:
            obj = json.loads(stamp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if obj.get("fp") != fp or obj.get("status") not in ("ok", "exists"):
            return None
        cmd = [str(x) for x in (obj.get("cmd") or [])]
        if fp == "fp16" and io_flag not in cmd:
            return None
        if any(f not in cmd for f in flags):
            return None
        obj["id"] = cid
        obj["status"] = "exists"
        obj["engine"] = str(dest.relative_to(ROOT))
        obj["fp"] = fp
        return obj

    existing16 = stamp_matches(engine16, "fp16")
    if existing16:
        return existing16
    if engine16.is_file():
        engine16.unlink()
    env = trt_env()
    last_err = ""
    for fp, dest in (("fp16", engine16), ("fp32", engine32)):
        dest.parent.mkdir(parents=True, exist_ok=True)
        args = [
            str(TRTEXEC),
            "--onnx=%s" % onnx,
            "--saveEngine=%s" % dest,
            "--skipInference",
            WS,
        ]
        if fp == "fp16":
            args.append("--fp16")
            args.append(io_flag)
        args.extend(flags)
        log("convert %s %s" % (cid, fp))
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                args, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)
            )
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if fp == "fp16" and proc.returncode != 0:
            (LOG / "convert" / ("%s.fp16.log" % cid)).write_text(text, encoding="utf-8")
        if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            _invalidate_after_reconvert(cid)
            return {
                "id": cid,
                "status": "ok",
                "engine": str(dest.relative_to(ROOT)),
                "fp": fp,
                "cmd": args,
            }
        last_err = text[-2000:]
        log("convert %s %s failed rc=%s" % (cid, fp, proc.returncode))
        if fp == "fp16":
            kept = stamp_matches(engine32, "fp32")
            if kept:
                log("convert %s keep existing fp32 after fp16 failure" % cid)
                return kept
        if dest.is_file() and dest.stat().st_size == 0:
            dest.unlink()
    if engine32.is_file() and engine32.stat().st_size > 0:
        return {"id": cid, "status": "exists", "engine": str(engine32.relative_to(ROOT)), "fp": "fp32"}
    return {"id": cid, "status": "FAIL", "error": last_err[-500:], "fp": None, "engine": None}


def profile_one(row: dict, conv: dict, profiles: dict[str, str | None]) -> dict:
    cid = row["convert_id"]
    engine = conv.get("engine")
    if not engine:
        return {"id": cid, "status": "SKIP", "reason": "no engine"}
    engine_path = ROOT / engine
    out_dir = LOG / "trtexec"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / ("%s.json" % cid)
    if existing.is_file():
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        if prev.get("status") == "ok" and prev.get("trtexec_rps"):
            return prev
    times = out_dir / ("%s.times.json" % cid)
    prof = out_dir / ("%s.profile.json" % cid)
    smi_csv = out_dir / ("%s.smi.csv" % cid)
    args = [
        str(TRTEXEC),
        "--loadEngine=%s" % engine_path,
        "--warmUp=500",
        "--duration=15",
        "--avgRuns=100",
        "--useSpinWait",
        "--dumpProfile",
        "--separateProfileRun",
        "--exportTimes=%s" % times,
        "--exportProfile=%s" % prof,
    ]
    if conv.get("fp") == "fp16":
        args.append("--fp16")
    flags = shape_flags(profiles.get(cid))
    if flags:
        # inference-time shapes use opt
        opt = [f.replace("--optShapes=", "--shapes=") for f in flags if f.startswith("--optShapes=")]
        args.extend(opt)
    cmd_txt = out_dir / ("%s.cmd.txt" % cid)
    cmd_line = " ".join(args)
    cmd_txt.write_text(cmd_line + "\n", encoding="utf-8")
    sampler = SmiSampler(smi_csv)
    log("profile %s" % cid)
    try:
        proc = subprocess.run(args, capture_output=True, text=True, env=trt_env(), cwd=str(ROOT))
    finally:
        smi = sampler.stop()
    text = (proc.stdout or "") + (proc.stderr or "")
    (out_dir / ("%s.log" % cid)).write_text(text, encoding="utf-8")
    parsed = parse_trtexec(text)
    result = {
        "id": cid,
        "status": "ok" if proc.returncode == 0 else "FAIL",
        "rc": proc.returncode,
        "trtexec_cmd": args,
        "trtexec_cmd_file": str(cmd_txt.relative_to(ROOT)),
        **parsed,
        **smi,
    }
    dump_json(out_dir / ("%s.json" % cid), result)
    return result


def kill_servers() -> None:
    subprocess.run(["killall", "-q", "mortred-model-server.out"], check=False)
    time.sleep(1)


def _decode_report_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    models = report.get("models") or []
    if not models:
        return False
    decode = models[0].get("decode") or {}
    return int(decode.get("valid") or 0) > 0 and decode.get("decode_auto") in ("cpu", "gpu")


def _calibrate_ready_for_http(row: dict, calib: dict) -> bool:
    out = LOG / "calibrate" / ("%s.json" % row["id"])
    if _decode_report_complete(out):
        return True
    if calib.get("status") != "ok":
        return False
    if not out.is_file():
        return False
    try:
        report = json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    models = report.get("models") or []
    if not models:
        return False
    if models[0].get("reason") == "start_failed":
        return False
    return any(p.get("started") for p in models[0].get("points") or [])


def calibrate_one(row: dict) -> dict:
    if not row["http"]:
        return {"id": row["id"], "status": "n/a-bench"}
    pack = ROOT / ("logs/bench/set_a/packs/%s.toml" % row["id"])
    jpeg = ensure_jpeg(ROOT / row["demo"])
    out = LOG / "calibrate" / ("%s.json" % row["id"])
    env = trt_env()
    env["MORTRED_DEMO_IMAGE"] = str(jpeg)
    env["MORTRED_PROJECT_ROOT"] = str(ROOT)
    base = [
        sys.executable,
        str(ROOT / "scripts" / "calibrate_pack.py"),
        "--pack",
        str(pack),
        "--project-root",
        str(ROOT),
        "--write-pack",
        "--output",
        str(out),
    ]
    if _decode_report_complete(out):
        cmd = base + ["--from-json", str(out)]
        log("calibrate %s reuse %s" % (row["id"], out.relative_to(ROOT)))
        write = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
        (LOG / "calibrate" / ("%s.write.log" % row["id"])).write_text(
            (write.stdout or "") + (write.stderr or ""), encoding="utf-8"
        )
        return {
            "id": row["id"],
            "status": "ok" if write.returncode == 0 else "write-pack-fail",
            "report": str(out.relative_to(ROOT)),
            "cmd": cmd,
            "write_rc": write.returncode,
            "reused": True,
        }
    cmd = base + [
        "--workers",
        "1,2,4,8,16",
        "--duration",
        "15s",
        "--skip-joint",
    ]
    log("calibrate %s" % row["id"])
    kill_servers()
    log_path = LOG / "calibrate" / ("%s.log" % row["id"])
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(ROOT))
    kill_servers()
    status = "ok" if proc.returncode == 0 else "FAIL"
    if status == "ok" and out.is_file():
        try:
            report = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
        models = report.get("models") or []
        if models and models[0].get("reason") == "start_failed":
            status = "FAIL"
        elif not _decode_report_complete(out) and not any(
            p.get("started") for p in (models[0].get("points") if models else []) or []
        ):
            status = "FAIL"
    return {
        "id": row["id"],
        "status": status,
        "report": str(out.relative_to(ROOT)),
        "cmd": cmd,
        "write_rc": proc.returncode,
    }


def read_wstar(row: dict) -> int:
    pack = ROOT / ("logs/bench/set_a/packs/%s.toml" % row["id"])
    if not pack.is_file():
        return 1
    for line in pack.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("worker_nums"):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return 1
    return 1


def parse_server_listen(server_toml: str | Path) -> tuple[int, str]:
    from pack_trt import server_listen

    path = Path(server_toml)
    if not path.is_absolute():
        path = ROOT / path
    port, uri = server_listen(path)
    return int(port), str(uri)


def http_one(row: dict) -> dict:
    if not row["http"]:
        return {"id": row["id"], "status": "n/a-bench"}
    from calibrate_pack import (
        calib_auth_token,
        start_model,
        stop_proc,
        wait_http_ready,
        wait_port_free,
        find_server_bin,
    )
    from http_infer_rps import LoadConfig, run_load

    binary = find_server_bin(ROOT)
    if binary is None:
        return {"id": row["id"], "status": "FAIL", "error": "no server binary"}
    wstar = read_wstar(row)
    jpeg = ensure_jpeg(ROOT / row["demo"])
    overlay = str(ROOT / ("conf/bench/set_a/%s.toml" % row["id"]))
    server = ROOT / ("conf/bench/set_a/%s.server.toml" % row["id"])
    if not server.is_file():
        server = ROOT / row["server"]
    port, uri = parse_server_listen(server)
    conc = max(8, min(64, 4 * wstar))
    token = calib_auth_token()
    result: dict = {
        "id": row["id"],
        "backend": "tensorrt",
        "w_star": wstar,
        "conc": conc,
        "image": str(jpeg.relative_to(ROOT)),
    }
    for pin in ("cpu", "gpu"):
        kill_servers()
        wait_port_free(port)
        extra = {
            "MORTRED_MODEL_CONFIG_FILE": overlay,
            "MORTRED_IMAGE_DECODE_BACKEND": pin,
            "MORTRED_PACK": str(ROOT / ("logs/bench/set_a/packs/%s.toml" % row["id"])),
        }
        slog = LOG / "http" / ("%s.%s.server.log" % (row["id"], pin))
        slog.parent.mkdir(parents=True, exist_ok=True)
        log("http %s pin=%s w=%s server=%s overlay=%s" % (row["id"], pin, wstar, server, overlay))
        proc = start_model(
            ROOT, binary, row["catalog"], server, wstar, slog, extra, token
        )
        ready = "http://127.0.0.1:%d/ready" % port
        url = "http://127.0.0.1:%d%s" % (port, uri)
        started = wait_http_ready(ready, 180.0, proc)
        text = slog.read_text(encoding="utf-8", errors="replace") if slog.is_file() else ""
        m = re.search(r"decode fork \[.*?\]: configured=(\w+) resolved=(\w+) gpu_path_eligible=(yes|no)", text)
        fork = {"configured": None, "resolved": None, "eligible": None}
        if m:
            fork = {"configured": m.group(1), "resolved": m.group(2), "eligible": m.group(3)}
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "server" / "http_infer_rps.py"),
            "--url",
            url,
            "--image",
            str(jpeg),
            "--concurrency",
            str(conc),
            "--warmup",
            "5s",
            "--duration",
            "15s",
            "--token",
            token,
            "--ready-url",
            ready,
            "--out",
            str(LOG / "http" / ("%s.%s.json" % (row["id"], pin))),
        ]
        (LOG / "http" / ("%s.%s.cmd.txt" % (row["id"], pin))).write_text(
            " ".join(cmd) + "\n", encoding="utf-8"
        )
        load = None
        if started:
            load = run_load(
                LoadConfig(
                    url=url,
                    image_path=jpeg,
                    concurrency=conc,
                    duration_s=15.0,
                    warmup_s=5.0,
                    token=token,
                    progress=False,
                    timeout_s=60.0,
                )
            ).to_dict()
        try:
            stop_proc(proc)
        except Exception as exc:
            log("stop_proc %s pin=%s ignored: %s" % (row["id"], pin, exc))
            try:
                proc.kill()
            except OSError:
                pass
        wait_port_free(port)
        pin_row = {
            "started": started,
            "fork": fork,
            "jpeggpu_ready": "jpeggpu decoder ready" in text,
            "report": load,
            "cmd_file": "logs/bench/set_a/http/%s.%s.cmd.txt" % (row["id"], pin),
        }
        if pin == "gpu" and fork.get("eligible") == "no":
            pin_row["http_rps"] = "fallback-cpu"
        elif load:
            pin_row["http_rps"] = load.get("rps")
            pin_row["p99_ms"] = (load.get("latency_ms") or {}).get("p99")
        else:
            pin_row["http_rps"] = None
        result[pin] = pin_row
    dump_json(LOG / "http" / ("%s.summary.json" % row["id"]), result)
    return result


def write_docs(converts: dict, profiles: dict, calibs: dict, https: dict) -> None:
    host = {}
    hp = LOG / "host.json"
    if hp.is_file():
        host = json.loads(hp.read_text(encoding="utf-8"))
    smi = str(host.get("nvidia_smi", "")).replace("\n", " | ")

    def cell(v: object) -> str:
        if v is None or v == "":
            return "—"
        if isinstance(v, float):
            return "%.2f" % v
        return str(v)

    lines = [
        "# A 集合 TensorRT 静态 batch=1 基准",
        "",
        "实测表。空单元格为未测或失败。trtexec Throughput 不是 HTTP RPS。表 3 不是发布 HTTP 值。",
        "",
        "## 机器",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| GPU | %s |" % smi,
        "| 日期 | %s |" % time.strftime("%Y-%m-%d"),
        "| workspace | 6G (8 GB card; trtexec suffix G, not GiB) |",
        "| 产品 toml | 未改；标定/HTTP 使用 conf/bench/set_a/<id>.server.toml → <id>.toml |",
        "",
        "## 表 1 — 身份 / 转换",
        "",
        "| id | catalog | fp | engine | convert |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for row in MODELS:
        cid = row["convert_id"]
        if cid in seen:
            conv = converts.get(cid, {})
            lines.append(
                "| %s | %s | (same engine %s) | — | — |" % (row["id"], row["catalog"], cid)
            )
            continue
        seen.add(cid)
        conv = converts.get(cid, {})
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row["id"], row["catalog"], cell(conv.get("fp")), cell(conv.get("engine")), cell(conv.get("status")))
        )
    lines += [
        "",
        "## 表 2 — trtexec profile",
        "",
        "| id | trtexec_rps | gpu_compute_ms | gpu_util_avg% | gpu_util_max% | cmd |",
        "|---|---:|---:|---:|---:|---|",
    ]
    seen = set()
    for row in unique_converts():
        cid = row["convert_id"]
        p = profiles.get(cid, {})
        cmd = p.get("trtexec_cmd_file") or "—"
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                cid,
                cell(p.get("trtexec_rps")),
                cell(p.get("gpu_compute_ms")),
                cell(p.get("gpu_util_avg_pct")),
                cell(p.get("gpu_util_max_pct")),
                cmd,
            )
        )
    lines += [
        "",
        "## 表 3 — 标定（非发布 RPS）",
        "",
        "`--workers 1,2,4,8,16 --duration 15s --skip-joint`",
        "",
        "| id | status | w* | decode_auto | rps@w* | report |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in MODELS:
        c = calibs.get(row["id"], {})
        w_star = decode_auto = rps_w = None
        report = c.get("report")
        rp = ROOT / report if report else LOG / "calibrate" / ("%s.json" % row["id"])
        if rp.is_file():
            try:
                rep = json.loads(rp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                rep = {}
            models = rep.get("models") or []
            if models:
                m0 = models[0]
                w_star = m0.get("suggested_worker_nums")
                decode_auto = (m0.get("decode") or {}).get("decode_auto")
                for pt in m0.get("points") or []:
                    if pt.get("worker_nums") == w_star:
                        rps_w = pt.get("rps")
                        break
        status_cell = c.get("status") or ("ok" if decode_auto else None)
        if rp.is_file():
            try:
                reason = ((json.loads(rp.read_text(encoding="utf-8")).get("models") or [{}])[0]).get("reason")
            except json.JSONDecodeError:
                reason = None
            if reason == "start_failed" or (not decode_auto and status_cell == "ok"):
                status_cell = "FAIL"
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                row["id"],
                cell(status_cell),
                cell(w_star),
                cell(decode_auto),
                cell(rps_w),
                cell(report or (str(rp.relative_to(ROOT)) if rp.is_file() else None)),
            )
        )
    lines += [
        "",
        "## 表 4 — 真实 HTTP（发布值）",
        "",
        "| id | w* | conc | http_rps_cpu | p99_cpu | http_rps_gpu | p99_gpu | eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    def pub_rps(pin: dict) -> object:
        if not pin.get("started"):
            return None
        report = pin.get("report") or {}
        if int(report.get("ok") or 0) <= 0:
            return None
        return pin.get("http_rps")

    for row in MODELS:
        h = https.get(row["id"], {})
        if h.get("status") in ("SKIP", "FAIL", "n/a-bench"):
            lines.append("| %s | — | — | — | — | — | — | — |" % row["id"])
            continue
        cpu = h.get("cpu") or {}
        gpu = h.get("gpu") or {}
        elig = (gpu.get("fork") or {}).get("eligible") or (cpu.get("fork") or {}).get("eligible")
        cpu_rps = pub_rps(cpu)
        gpu_rps = pub_rps(gpu)
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["id"],
                cell(h.get("w_star") if (cpu_rps is not None or gpu_rps is not None) else None),
                cell(h.get("conc") if (cpu_rps is not None or gpu_rps is not None) else None),
                cell(cpu_rps),
                cell(cpu.get("p99_ms") if cpu_rps is not None else None),
                cell(gpu_rps),
                cell(gpu.get("p99_ms") if gpu_rps is not None else None),
                cell(elig if (cpu_rps is not None or gpu_rps is not None) else None),
            )
        )
    gap_lines = []
    for row in unique_converts():
        cid = row["convert_id"]
        conv = converts.get(cid, {})
        if conv.get("status") == "FAIL":
            err = str(conv.get("error") or "").replace("\n", " ")
            if len(err) > 240:
                err = err[:240] + "…"
            gap_lines.append("- convert `%s`: %s" % (cid, err or "FAIL"))
        if not row.get("http"):
            gap_lines.append("- `%s`: convert+profile only (`http=False`)" % cid)
    if gap_lines:
        lines += ["", "## Gaps（失败/非 HTTP，来自 convert JSON）", ""] + gap_lines
    body = "\n".join(lines) + "\n"
    (ROOT / "docs" / "benchmark-set-a.md").write_text(body, encoding="utf-8")
    zh = body.replace("# A 集合 TensorRT 静态 batch=1 基准", "# A 集合 TensorRT 静态 batch=1 基准（中文）")
    (ROOT / "docs" / "benchmark-set-a.zh-cn.md").write_text(zh, encoding="utf-8")
    log("wrote docs/benchmark-set-a.md")


def load_profiles() -> dict[str, str | None]:
    man = ROOT / "conf" / "trt_engines.set_a.json"
    if not man.is_file():
        return {}
    data = json.loads(man.read_text(encoding="utf-8"))
    return {e["model"]: e.get("profile") for e in data.get("engines", [])}


def pipeline(only: str | None, start_from: str | None, stages: list[str]) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    jobs = list(MODELS)
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        jobs = [j for j in jobs if j["id"] in wanted or j["convert_id"] in wanted]
    if start_from:
        keep = False
        filtered = []
        for j in MODELS:
            if j["id"] == start_from or j["convert_id"] == start_from:
                keep = True
            if keep:
                filtered.append(j)
        jobs = filtered
    profiles = load_profiles()
    converts: dict = {}
    profs: dict = {}
    calibs: dict = {}
    https: dict = {}
    for p in (LOG / "convert").glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            converts[obj["id"]] = obj
    for p in (LOG / "trtexec").glob("*.json"):
        if p.name.endswith(".times.json") or p.name.endswith(".profile.json"):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            profs[obj["id"]] = obj
    for p in (LOG / "calibrate").glob("*.driver.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            calibs[obj["id"]] = obj
    for p in (LOG / "http").glob("*.summary.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            https[obj["id"]] = obj

    sequential = stages == ["convert", "profile", "serve", "table"]
    if sequential:
        converted: set[str] = set()
        for row in jobs:
            cid = row["convert_id"]
            if cid not in converted:
                write_status(stage="convert", current=cid)
                converts[cid] = convert_one(row, profiles)
                dump_json(LOG / "convert" / ("%s.json" % cid), converts[cid])
                point_overlays_at_engine(cid, converts[cid].get("engine") or "")
                write_status(stage="profile", current=cid)
                profs[cid] = profile_one(row, converts.get(cid, {}), profiles)
                converted.add(cid)
            write_status(stage="calibrate", current=row["id"])
            calibs[row["id"]] = calibrate_one(row)
            dump_json(LOG / "calibrate" / ("%s.driver.json" % row["id"]), calibs[row["id"]])
            write_status(stage="http", current=row["id"])
            if _calibrate_ready_for_http(row, calibs[row["id"]]):
                https[row["id"]] = http_one(row)
            else:
                log("skip http %s: calibrate %s" % (row["id"], calibs[row["id"]].get("status")))
                https[row["id"]] = {
                    "id": row["id"],
                    "status": "SKIP",
                    "error": "calibrate did not write a complete report",
                }
            write_docs(converts, profs, calibs, https)
        write_status(stage="idle", current="")
        return 0

    if "convert" in stages:
        done: set[str] = set()
        for row in jobs:
            cid = row["convert_id"]
            if cid in done:
                continue
            done.add(cid)
            write_status(stage="convert", current=cid)
            converts[cid] = convert_one(row, profiles)
            dump_json(LOG / "convert" / ("%s.json" % cid), converts[cid])
            point_overlays_at_engine(cid, converts[cid].get("engine") or "")
    if "profile" in stages:
        done = set()
        for row in jobs:
            cid = row["convert_id"]
            if cid in done:
                continue
            done.add(cid)
            write_status(stage="profile", current=cid)
            profs[cid] = profile_one(row, converts.get(cid, {}), profiles)
    if "serve" in stages:
        for row in jobs:
            write_status(stage="calibrate", current=row["id"])
            calibs[row["id"]] = calibrate_one(row)
            dump_json(LOG / "calibrate" / ("%s.driver.json" % row["id"]), calibs[row["id"]])
            write_status(stage="http", current=row["id"])
            if _calibrate_ready_for_http(row, calibs[row["id"]]):
                https[row["id"]] = http_one(row)
            else:
                log("skip http %s: calibrate %s" % (row["id"], calibs[row["id"]].get("status")))
                https[row["id"]] = {
                    "id": row["id"],
                    "status": "SKIP",
                    "error": "calibrate did not write a complete report",
                }
            write_docs(converts, profs, calibs, https)
    if "table" in stages:
        write_docs(converts, profs, calibs, https)
    write_status(stage="idle", current="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="all",
                        help="generate|convert|profile|serve|table|all|smoke")
    parser.add_argument("--only", default="")
    parser.add_argument("--from", dest="start_from", default="")
    args = parser.parse_args()
    if args.stage in ("generate", "all", "smoke"):
        subprocess.check_call([sys.executable, str(Path(__file__).with_name("generate.py"))])
    if args.stage == "generate":
        return 0
    if args.stage == "smoke":
        return pipeline("yolov8s", "", ["convert", "profile", "serve", "table"])
    if args.stage == "all":
        return pipeline(args.only or None, args.start_from or None, ["convert", "profile", "serve", "table"])
    stages = {
        "convert": ["convert"],
        "profile": ["profile"],
        "serve": ["serve"],
        "table": ["table"],
    }[args.stage]
    return pipeline(args.only or None, args.start_from or None, stages)


if __name__ == "__main__":
    raise SystemExit(main())
