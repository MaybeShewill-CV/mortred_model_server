#!/usr/bin/env python3
"""Set A HTTP --raw pins (CPU then GPU). One catalog row per model; does not overwrite JSON summaries."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

from catalog import MODELS  # noqa: E402
from run import (  # noqa: E402
    LOG,
    SmiSampler,
    dump_json,
    ensure_jpeg,
    kill_servers,
    parse_server_listen,
    read_wstar,
)
from calibrate_pack import (  # noqa: E402
    calib_auth_token,
    find_server_bin,
    start_model,
    stop_proc,
    wait_http_ready,
    wait_port_free,
)
from http_infer_rps import LoadConfig, run_load  # noqa: E402

HTTP_MODELS = [m for m in MODELS if m.get("http", True)]
TABLE = LOG / "http" / "json_vs_raw.json"
RAW_LOG = LOG / "http" / "raw.log"


def rlog(msg: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RAW_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def r4(x: float | None) -> float | None:
    if x is None:
        return None
    return float(f"{x:.4g}")


def http_rps(block: dict | None) -> float | None:
    if not isinstance(block, dict) or not block.get("started"):
        return None
    r = block.get("http_rps")
    if num(r):
        return float(r)
    r = (block.get("report") or {}).get("rps")
    return float(r) if num(r) else None


def trtexec_qps(row: dict) -> float | None:
    mid, cid = row["id"], row["convert_id"]
    path = LOG / "trtexec" / ("%s.json" % mid)
    if not path.is_file() and cid != mid:
        path = LOG / "trtexec" / ("%s.json" % cid)
    if not path.is_file():
        return None
    t = json.loads(path.read_text(encoding="utf-8"))
    q = t.get("trtexec_rps")
    return float(q) if num(q) else None


def json_side(row: dict) -> dict:
    mid = row["id"]
    wstar = read_wstar(row)
    conc = max(8, min(64, 4 * wstar))
    decode = None
    cal = LOG / "calibrate" / ("%s.json" % mid)
    if cal.is_file():
        m0 = (json.loads(cal.read_text(encoding="utf-8")).get("models") or [{}])[0]
        wstar = m0.get("suggested_worker_nums") or wstar
        conc = max(8, min(64, 4 * int(wstar)))
        decode = (m0.get("decode") or {}).get("decode_auto")
    cpu = gpu = None
    hp = LOG / "http" / ("%s.summary.json" % mid)
    if hp.is_file():
        h = json.loads(hp.read_text(encoding="utf-8"))
        if num(h.get("conc")):
            conc = int(h["conc"])
        cpu = http_rps(h.get("cpu") or {})
        gpu = http_rps(h.get("gpu") or {})
        wstar = h.get("w_star") or wstar
    trt = trtexec_qps(row)
    ach = None
    if cpu is not None and gpu is not None and trt:
        ach = r4(max(cpu, gpu) / trt)
    return {
        "id": mid,
        "trtexec_qps": trt,
        "w_star": wstar,
        "decode_auto": decode,
        "conc": conc,
        "json_cpu_rps": cpu,
        "json_gpu_rps": gpu,
        "json_cpu_util_avg_pct": None,
        "json_cpu_util_max_pct": None,
        "json_gpu_util_avg_pct": None,
        "json_gpu_util_max_pct": None,
        "json_achievement": ach,
        "raw_cpu_rps": None,
        "raw_gpu_rps": None,
        "raw_cpu_util_avg_pct": None,
        "raw_cpu_util_max_pct": None,
        "raw_gpu_util_avg_pct": None,
        "raw_gpu_util_max_pct": None,
        "raw_achievement": None,
    }


def merge_raw(base: dict, raw_summary: dict) -> dict:
    out = dict(base)
    cpu = raw_summary.get("cpu") or {}
    gpu = raw_summary.get("gpu") or {}
    out["raw_cpu_rps"] = http_rps(cpu)
    out["raw_gpu_rps"] = http_rps(gpu)
    out["raw_cpu_util_avg_pct"] = cpu.get("gpu_util_avg_pct")
    out["raw_cpu_util_max_pct"] = cpu.get("gpu_util_max_pct")
    out["raw_gpu_util_avg_pct"] = gpu.get("gpu_util_avg_pct")
    out["raw_gpu_util_max_pct"] = gpu.get("gpu_util_max_pct")
    trt = out.get("trtexec_qps")
    rc, rg = out["raw_cpu_rps"], out["raw_gpu_rps"]
    if rc is not None and rg is not None and trt:
        out["raw_achievement"] = r4(max(rc, rg) / float(trt))
    if num(raw_summary.get("conc")):
        out["conc"] = int(raw_summary["conc"])
    if raw_summary.get("w_star") is not None:
        out["w_star"] = raw_summary["w_star"]
    return out


def raw_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        h = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return http_rps(h.get("cpu") or {}) is not None and http_rps(h.get("gpu") or {}) is not None


def write_table() -> dict:
    rows = []
    done = 0
    for row in HTTP_MODELS:
        rec = json_side(row)
        raw_path = LOG / "http" / ("%s.raw.summary.json" % row["id"])
        if raw_path.is_file():
            rec = merge_raw(rec, json.loads(raw_path.read_text(encoding="utf-8")))
        if rec["raw_cpu_rps"] is not None and rec["raw_gpu_rps"] is not None:
            done += 1
        rows.append(rec)
    payload = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": done,
        "total": len(HTTP_MODELS),
        "rows": rows,
    }
    dump_json(TABLE, payload)
    return payload


def raw_one(row: dict) -> dict:
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
        "encoding": "raw",
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
        slog = LOG / "http" / ("%s.%s.raw.server.log" % (row["id"], pin))
        rlog("raw http %s pin=%s w=%s conc=%s" % (row["id"], pin, wstar, conc))
        proc = start_model(ROOT, binary, row["catalog"], server, wstar, slog, extra, token)
        ready = "http://127.0.0.1:%d/ready" % port
        url = "http://127.0.0.1:%d%s" % (port, uri)
        started = wait_http_ready(ready, 180.0, proc)
        smi_csv = LOG / "http" / ("%s.%s.raw.smi.csv" % (row["id"], pin))
        load = None
        smi: dict = {}
        if started:
            sampler = SmiSampler(smi_csv)
            try:
                load = run_load(
                    LoadConfig(
                        url=url,
                        image_path=jpeg,
                        concurrency=conc,
                        duration_s=15.0,
                        warmup_s=5.0,
                        token=token,
                        raw=True,
                        progress=False,
                        timeout_s=60.0,
                    )
                ).to_dict()
            finally:
                smi = sampler.stop()
            dump_json(LOG / "http" / ("%s.%s.raw.json" % (row["id"], pin)), load)
        try:
            stop_proc(proc)
        except Exception as exc:
            rlog("stop_proc %s pin=%s ignored: %s" % (row["id"], pin, exc))
            try:
                proc.kill()
            except OSError:
                pass
        wait_port_free(port)
        pin_row = {
            "started": started,
            "report": load,
            "gpu_util_avg_pct": smi.get("gpu_util_avg_pct"),
            "gpu_util_max_pct": smi.get("gpu_util_max_pct"),
            "smi_samples": smi.get("smi_samples"),
            "smi_csv": str(smi_csv.relative_to(ROOT)) if smi_csv.is_file() else None,
        }
        if load:
            pin_row["http_rps"] = load.get("rps")
            pin_row["p99_ms"] = (load.get("latency_ms") or {}).get("p99")
        else:
            pin_row["http_rps"] = None
        result[pin] = pin_row
    dump_json(LOG / "http" / ("%s.raw.summary.json" % row["id"]), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated catalog ids")
    args = parser.parse_args()
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    binary = find_server_bin(ROOT)
    if binary is None:
        rlog("no server binary")
        return 1
    write_table()
    todo = []
    for i, row in enumerate(HTTP_MODELS, 1):
        if wanted and row["id"] not in wanted:
            continue
        raw_path = LOG / "http" / ("%s.raw.summary.json" % row["id"])
        if raw_complete(raw_path):
            rlog("skip complete %s" % row["id"])
            continue
        todo.append((i, row))
    rlog("raw queue %d / %d" % (len(todo), len(HTTP_MODELS)))
    for i, row in todo:
        rlog("start %s (%d/%d)" % (row["id"], i, len(HTTP_MODELS)))
        try:
            raw_one(row)
        except Exception as exc:
            rlog("FAIL %s: %s" % (row["id"], exc))
            dump_json(
                LOG / "http" / ("%s.raw.summary.json" % row["id"]),
                {"id": row["id"], "status": "FAIL", "error": str(exc)},
            )
        table = write_table()
        rec = next(r for r in table["rows"] if r["id"] == row["id"])
        payload = {
            "id": row["id"],
            "index": i,
            "total": len(HTTP_MODELS),
            "complete": table["complete"],
            "row": rec,
        }
        print("AGENT_LOOP_WAKE_raw_http %s" % json.dumps(payload, ensure_ascii=False), flush=True)
    rlog("raw pass finished complete=%s/%s" % (write_table()["complete"], len(HTTP_MODELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
