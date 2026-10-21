#!/usr/bin/env python3
"""HTTP RPS checklist for the coverage set and the full product catalog.

Default catalog is the nine coverage-set ids
(docs/perf/coverage_set_http_rps.md). ``--catalog product_static`` reads all
ids from conf/product_static_bs1.json and uses
docs/perf/product_static_http_rps.md.

--check reads the checklist and does not start processes or write it.
Coverage-set --check requires every section to be a qualified PASS row
(libface may be blank). Product-static --check requires the same for all 39
ids and exits non-zero while any section is blocked/missing. It never treats
a missing row as PASS.
--check-skeleton checks heading order and that each section is either an
explicit blocked/missing row or a qualified PASS row.
--self-test runs the in-memory judge fixtures (concurrency, error rate, caps).
A coverage-set live run with no arguments clears the coverage checklist,
then for each id: prepare, calibrate --write-pack, deploy that pack alone,
official --raw load, and append the row before the next id.
--from <id> keeps rows already in that file and continues at <id>.
--id <id> runs one id and replaces that section only.
Product-static live runs require --id or --from so the skeleton is not wiped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

from calibrate_pack import (  # noqa: E402
    _hist_delta,
    _metric_series,
    format_missing_p99_table,
    gpu_inventory,
    histogram_p50,
    missing_p99_caps,
    pack_table,
)
from http_infer_rps import close_client_first  # noqa: E402
from pack_trt import server_listen  # noqa: E402
from repo_toml import load_toml  # noqa: E402

import http.client  # noqa: E402

REPORT = ROOT / "docs" / "perf" / "coverage_set_http_rps.md"
PRODUCT_REPORT = ROOT / "docs" / "perf" / "product_static_http_rps.md"
BASELINE_DIR = ROOT / "docs" / "perf" / "baselines"
CATALOG_NAMES = ("coverage_set", "product_static")
FACE_IMAGE = "demo_data/model_test_input/scene_segmentation/face_w_mask.jpg"
DBNET_IMAGE = "demo_data/model_test_input/ocr/coco_text_test.jpg"
SUPERPOINT_IMAGE = "demo_data/model_test_input/feature_point/test.jpg"
METRIC3D_IMAGE = "demo_data/model_test_input/mono_depth_estimation/0000000005.jpg"
REALESRGAN_IMAGE = "logs/p2-11-realesrgan-256.jpg"
GATEWAY = "http://127.0.0.1:8080"
STAGES = ("gw_in", "route", "auth", "fwd", "upstream")
TOKEN_KEYS = (
    "MORTRED_API_TOKEN",
    "MORTRED_GATEWAY_AUTH_TOKEN",
    "MORTRED_METRICS_TOKEN",
)
YOLO_THROUGHPUT = 400.677

MODELS = (
    {
        "id": "mobilenetv2",
        "conf_server": "conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/classification/mobilenetv2",
        "port": 9002,
    },
    {
        "id": "yolov8s",
        "conf_server": "conf/server/object_detection/yolov8/yolov8s_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/obj_detection/yolov8s",
        "port": 9056,
    },
    {
        "id": "libface",
        "conf_server": "conf/server/object_detection/libface_det/libface_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/obj_detection/libface",
        "port": 9051,
    },
    {
        "id": "pphuman_mobile",
        "conf_server": "conf/server/scene_segmentation/pphuman_seg/pphuman_mobile_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/scene_segmentation/pphuman_mobile",
        "port": 9072,
    },
    {
        "id": "dbnet",
        "conf_server": "conf/server/ocr/dbnet/dbnet_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/ocr/dbnet",
        "port": 9060,
    },
    {
        "id": "dinov2_vits14",
        "conf_server": "conf/server/feature_embedding/dinov2/dinov2_vits14_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/feature_embedding/dinov2_vits14",
        "port": 9111,
    },
    {
        "id": "superpoint",
        "conf_server": "conf/server/feature_point/superpoint/superpoint_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/feature_point/superpoint",
        "port": 9020,
    },
    {
        "id": "realesrgan",
        "conf_server": "conf/server/enhancement/real_esrgan/realesrgan_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/enhancement/realesrgan",
        "port": 9012,
    },
    {
        "id": "metric3d_512",
        "conf_server": "conf/server/mono_depth_estimation/metric3d/metric3d_512_server_config.toml",
        "server_uri": "/mortred_ai_server_v1/mono_depth_estimation/metric3d_512",
        "port": 9041,
    },
)
MODEL_BY_ID = {row["id"]: row for row in MODELS}
_PRODUCT_SPECS: tuple[dict, ...] | None = None


def product_static_specs() -> tuple[dict, ...]:
    global _PRODUCT_SPECS
    if _PRODUCT_SPECS is not None:
        return _PRODUCT_SPECS
    doc = json.loads((ROOT / "conf" / "product_static_bs1.json").read_text(encoding="utf-8"))
    models = doc.get("models")
    if not isinstance(models, list) or not models:
        raise RuntimeError("conf/product_static_bs1.json has no models")
    rows = []
    seen = set()
    for item in models:
        if not isinstance(item, dict) or not item.get("id"):
            raise RuntimeError("product catalog row is missing id")
        model_id = str(item["id"])
        if model_id in seen:
            raise RuntimeError("duplicate product id %s" % model_id)
        seen.add(model_id)
        rows.append(
            {
                "id": model_id,
                "conf_server": str(item["server_config"]),
                "server_uri": str(item["server_uri"]),
                "port": int(item["port"]),
                "demo": str(item.get("demo") or ""),
                "family": str(item.get("family") or ""),
                "catalog": "product_static",
            }
        )
    _PRODUCT_SPECS = tuple(rows)
    return _PRODUCT_SPECS


def specs_for(catalog: str) -> tuple[dict, ...]:
    if catalog == "coverage_set":
        return MODELS
    if catalog == "product_static":
        return product_static_specs()
    raise RuntimeError("unknown catalog %s" % catalog)


def report_path_for(catalog: str) -> Path:
    if catalog == "coverage_set":
        return REPORT
    if catalog == "product_static":
        return PRODUCT_REPORT
    raise RuntimeError("unknown catalog %s" % catalog)


def spec_map_for(specs: tuple[dict, ...] | list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in specs}


def load_image_rel(spec: dict) -> str:
    model_id = spec["id"]
    if model_id == "pphuman_mobile":
        return FACE_IMAGE
    if model_id == "dbnet":
        return DBNET_IMAGE
    if model_id == "superpoint":
        return SUPERPOINT_IMAGE
    if model_id == "realesrgan":
        return REALESRGAN_IMAGE
    if model_id == "metric3d_512":
        return METRIC3D_IMAGE
    if spec.get("catalog") == "product_static":
        return str(spec.get("demo") or "")
    return ""


def solo_pack_text(model_id: str) -> str:
    """One-id pack text. Prefer conf/packs/<id>.toml when that file is solo."""
    candidate = ROOT / "conf" / "packs" / ("%s.toml" % model_id)
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
        found = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[pack.") and stripped.endswith("]"):
                found.append(stripped[6:-1].strip())
        if found != [model_id]:
            raise RuntimeError(
                "conf/packs/%s.toml is not a solo pack for %s (ids=%s); "
                "refusing to deploy more than one id" % (model_id, model_id, found)
            )
        return text
    return "[pack.%s]\nworker_nums = 1\n" % model_id


def _fail(message: str) -> int:
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")
    return 1


def jpeg_sof0_size(data: bytes):
    """Return (width, height) for a baseline SOF0 scan, else None."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 1 < len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        i += 2
        if marker in (0xD8, 0xD9):
            return None
        if marker == 0x00 or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            return None
        seglen = int.from_bytes(data[i : i + 2], "big")
        if seglen < 2 or i + seglen > len(data):
            return None
        if marker == 0xC0:
            if seglen < 7:
                return None
            height = int.from_bytes(data[i + 3 : i + 5], "big")
            width = int.from_bytes(data[i + 5 : i + 7], "big")
            return width, height
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return None
        i += seglen
    return None


def read_server_row(rel: str) -> tuple[str, int, int]:
    doc = load_toml(ROOT / rel)
    for section in doc.values():
        if isinstance(section, dict) and "server_uri" in section:
            return str(section["server_uri"]), int(section["worker_nums"]), int(section["port"])
    raise RuntimeError("no server_uri in %s" % rel)


def contract_ok(specs: tuple[dict, ...] | list[dict] | None = None) -> str | None:
    rows = MODELS if specs is None else specs
    for row in rows:
        try:
            uri, workers, port = read_server_row(row["conf_server"])
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            return "%s: %s" % (row["id"], exc)
        if uri != row["server_uri"] or workers != 1 or port != row["port"]:
            return "%s conf/server is %s worker_nums=%s port=%s" % (row["id"], uri, workers, port)
        listened = server_listen(ROOT / row["conf_server"])
        if listened != (row["port"], row["server_uri"]):
            return "%s server_listen %s" % (row["id"], listened)
    return None


def read_throughput(model_id: str) -> float:
    text = (ROOT / "docs" / "benchmark-h3" / ("%s.txt" % model_id)).read_text(
        encoding="utf-8", errors="replace"
    )
    match = re.search(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps", text, re.IGNORECASE)
    if match is None:
        raise RuntimeError("no Throughput in docs/benchmark-h3/%s.txt" % model_id)
    if model_id == "yolov8s" and match.group(1) != "400.677":
        raise RuntimeError("yolov8s Throughput is %s" % match.group(1))
    return float(match.group(1))


def hist_count_delta(before_text: str, after_text: str, name: str, model: str):
    before = _metric_series(before_text or "", name, model)
    after = _metric_series(after_text or "", name, model)
    if before is None:
        before = {"buckets": [], "count": 0.0}
    delta = _hist_delta(before, after)
    if delta is None:
        return None
    return float(delta["count"])


def _gauge(text: str, name: str, model: str):
    match = re.search(
        r'%s\{model="%s"\} ([0-9.eE+-]+)' % (re.escape(name), re.escape(model)),
        text or "",
    )
    if match is None:
        return None
    return float(match.group(1))


def jpeg_counts(text: str) -> dict[str, float]:
    return {
        match.group(1): float(match.group(2))
        for match in re.finditer(
            r'mortred_jpeg_decode_total\{backend="([^"]+)"\} ([0-9.eE+-]+)',
            text or "",
        )
    }


def jpeg_delta(before_text: str, after_text: str) -> dict[str, float]:
    before = jpeg_counts(before_text)
    after = jpeg_counts(after_text)
    keys = ("jpeggpu", "cpu-reduced", "cpu-full")
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys}


def stage_series(text: str, stage: str):
    bucket_re = re.compile(
        r'mortred_gateway_stage_duration_ms_bucket\{model="gateway",stage="%s",le="([^"]+)"\} ([0-9.eE+-]+)'
        % re.escape(stage)
    )
    count_re = re.compile(
        r'mortred_gateway_stage_duration_ms_count\{model="gateway",stage="%s"\} ([0-9.eE+-]+)'
        % re.escape(stage)
    )
    buckets = []
    for match in bucket_re.finditer(text or ""):
        if match.group(1) == "+Inf":
            continue
        buckets.append((float(match.group(1)), float(match.group(2))))
    count_match = count_re.search(text or "")
    if not buckets and count_match is None:
        return None
    buckets.sort()
    count = float(count_match.group(1)) if count_match else (buckets[-1][1] if buckets else 0.0)
    return {"buckets": buckets, "count": count}


def dominant_name(p50s: dict[str, float]) -> str:
    winner = STAGES[0]
    for name in STAGES[1:]:
        if p50s[name] >= p50s[winner]:
            winner = name
    return winner


def dominant_gauge(text: str):
    match = re.search(
        r'mortred_gateway_stage_dominant\{model="gateway",stage="([^"]+)"\} 1',
        text or "",
    )
    if match is None:
        return None
    return match.group(1)


def jpeg_landing(decode_auto: str, counts: dict[str, float]) -> str | None:
    gpu = float(counts.get("jpeggpu", 0.0))
    cpu = float(counts.get("cpu-reduced", 0.0)) + float(counts.get("cpu-full", 0.0))
    if decode_auto == "gpu" and gpu > 0.0 and gpu > cpu:
        return "gpu"
    if decode_auto == "cpu" and cpu > 0.0 and cpu > gpu:
        return "cpu"
    return None


def _near(left, right) -> bool:
    return abs(float(left) - float(right)) <= 1e-6


def judge_row(row: dict, specs: dict | None = None) -> tuple[bool, str]:
    if row.get("blank"):
        if row.get("id") != "libface":
            return False, "only libface may be blank"
        return True, "blank"
    model_id = row.get("id")
    table = MODEL_BY_ID if specs is None else specs
    spec = table.get(model_id)
    if spec is None:
        return False, "unknown id"
    if row.get("server_uri") != spec["server_uri"]:
        return False, "server_uri"
    if row.get("conf_server") != spec["conf_server"]:
        return False, "conf_server"
    if row.get("conf_server_worker_nums") != 1:
        return False, "conf/server worker_nums"
    workers = row.get("worker_nums")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        return False, "worker_nums"
    decode_auto = row.get("decode_auto")
    if decode_auto not in ("cpu", "gpu"):
        return False, "decode_auto"
    if row.get("mortred_worker_nums") != workers or row.get("mortred_decode_auto") != decode_auto:
        return False, "process env"
    if row.get("process_workers_source") != "after_load":
        return False, "process worker source"
    note = str(row.get("process_workers_note") or "")
    if "不是压测前刮取" not in note:
        return False, "process worker note"
    available = row.get("workers_available_after")
    busy = row.get("workers_busy_after")
    if not isinstance(available, (int, float)) or not isinstance(busy, (int, float)):
        return False, "worker gauges"
    if isinstance(available, bool) or isinstance(busy, bool):
        return False, "worker gauges"
    if not _near(row.get("process_workers"), available + busy) or not _near(
        row.get("process_workers"), workers
    ):
        return False, "process workers"
    if _near(row.get("process_workers"), 0) and workers != 0:
        return False, "preload zero used as process workers"
    calibrate = str(row.get("calibrate") or "")
    if "mortredctl calibrate" not in calibrate or "--write-pack" not in calibrate:
        return False, "calibrate command"
    if "--duration 60s" not in calibrate or "--from-json" in calibrate or "--workers" in calibrate:
        return False, "calibrate flags"
    if "model_config" in calibrate:
        return False, "calibrate model_config"
    if "--raw" not in str(row.get("calibrate_raw") or ""):
        return False, "calibrate raw"
    command = str(row.get("load_command") or "")
    if "--raw" not in command or "-d 60s" not in command or "--warmup 5s" not in command:
        return False, "load flags"
    if "--qps" in command:
        return False, "load qps"
    match = re.search(r"(?:^|\s)-c\s+(\d+)(?:\s|$)", command)
    if match is None or int(match.group(1)) != workers * 2:
        return False, "load concurrency"
    url = str(row.get("load_url") or "")
    if url != "http://127.0.0.1:8080/v1/models/%s/infer" % model_id:
        return False, "load url"
    if not isinstance(row.get("wall_s"), (int, float)) or float(row["wall_s"]) < 60.0:
        return False, "wall_s"
    if not _near(row.get("warmup_s"), 5.0):
        return False, "warmup"
    requests = row.get("requests")
    errors = row.get("errors")
    http_429 = row.get("http_429")
    if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
        return False, "requests"
    if not isinstance(errors, int) or isinstance(errors, bool) or errors < 0:
        return False, "errors"
    if not isinstance(http_429, int) or isinstance(http_429, bool) or http_429 < 0:
        return False, "http_429"
    if http_429 > errors:
        return False, "http_429"
    expected_rate = (errors - http_429) / requests
    if not isinstance(row.get("error_rate"), (int, float)) or isinstance(row.get("error_rate"), bool):
        return False, "error_rate"
    if abs(float(row["error_rate"]) - expected_rate) > 1e-6:
        return False, "error_rate folded"
    if float(row["error_rate"]) > 0.01 + 1e-12:
        return False, "error_rate above 1%"
    rps = row.get("success_rps")
    if not isinstance(rps, (int, float)) or isinstance(rps, bool) or float(rps) <= 0:
        return False, "success rps"
    throughput = row.get("throughput_qps")
    if not isinstance(throughput, (int, float)) or isinstance(throughput, bool) or float(throughput) <= 0:
        return False, "throughput"
    if model_id == "yolov8s" and not _near(throughput, YOLO_THROUGHPUT):
        return False, "yolov8s throughput"
    ratio = row.get("ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        return False, "ratio"
    if abs(float(ratio) - (float(rps) / float(throughput))) > 1e-4:
        return False, "ratio"
    if not isinstance(row.get("p50_ms"), (int, float)) or not isinstance(row.get("p99_ms"), (int, float)):
        return False, "latency"
    if row.get("queue_model") != spec["server_uri"] or row.get("infer_model") != spec["server_uri"]:
        return False, "histogram model"
    queue_delta = row.get("queue_count_delta")
    infer_delta = row.get("infer_count_delta")
    if not isinstance(queue_delta, (int, float)) or not isinstance(infer_delta, (int, float)):
        return False, "histogram delta"
    if isinstance(queue_delta, bool) or isinstance(infer_delta, bool):
        return False, "histogram delta"
    if float(queue_delta) <= 0 or float(infer_delta) <= 0:
        return False, "histogram delta"
    landing = jpeg_landing(
        decode_auto,
        {
            "jpeggpu": row.get("jpeg_jpeggpu"),
            "cpu-reduced": row.get("jpeg_cpu_reduced"),
            "cpu-full": row.get("jpeg_cpu_full"),
        },
    )
    if landing != decode_auto or row.get("jpeg_landing") != decode_auto:
        return False, "jpeg landing"
    stages = row.get("stage_p50_ms")
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        return False, "gateway stages"
    try:
        p50s = {name: float(stages[name]) for name in STAGES}
    except (TypeError, ValueError):
        return False, "gateway stages"
    if not _near(row.get("stage_p50_sum"), sum(p50s.values())):
        return False, "gateway stage sum"
    server_p50 = row.get("server_time_ms_p50")
    if not isinstance(server_p50, (int, float)) or isinstance(server_p50, bool):
        return False, "server_time_ms"
    winner = dominant_name(p50s)
    if float(row["stage_p50_sum"]) > 0.5 * float(server_p50):
        if row.get("dominant_stage") != winner:
            return False, "dominant stage"
    elif row.get("dominant_stage") not in (None, ""):
        return False, "dominant stage"
    gauge = row.get("dominant_gauge")
    if gauge not in (None, "", "absent") and gauge != row.get("dominant_stage"):
        return False, "dominant gauge"
    if not str(row.get("hardware") or "").strip() or "MiB" not in str(row.get("hardware")):
        return False, "hardware"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(row.get("date") or "")):
        return False, "date"
    if "Bearer " in json.dumps(row):
        return False, "token"
    image = str(row.get("image") or "")
    if model_id == "pphuman_mobile":
        if not image.endswith("face_w_mask.jpg") or "human_image" in image:
            return False, "pphuman image"
    if model_id == "dbnet":
        if not image.endswith("coco_text_test.jpg") or "railway_ticket" in image:
            return False, "dbnet image"
    if model_id == "superpoint":
        if not image.endswith("feature_point/test.jpg") or "test.png" in image:
            return False, "superpoint image"
    if model_id == "metric3d_512":
        if not image.endswith("mono_depth_estimation/0000000005.jpg") or "0000000005.png" in image:
            return False, "metric3d image"
    if model_id == "realesrgan":
        if "p2-11-realesrgan-256.jpg" not in image:
            return False, "realesrgan image"
        if row.get("decoded_width") != 256 or row.get("decoded_height") != 256:
            return False, "realesrgan size"
    if "static" not in str(row.get("engine") or "") or "batch=1" not in str(row.get("engine") or ""):
        return False, "engine"
    return True, "ok"


def judge_document(rows: list[dict], specs: tuple[dict, ...] | list[dict] | None = None) -> tuple[bool, str]:
    catalog = MODELS if specs is None else specs
    if [row.get("id") for row in rows] != [spec["id"] for spec in catalog]:
        return False, "id order"
    table = spec_map_for(catalog)
    for row in rows:
        ok, why = judge_row(row, table)
        if not ok:
            return False, "%s: %s" % (row.get("id"), why)
    return True, "ok"


def _fmt(key: str, value) -> str:
    text = str(value)
    if any(ch in text for ch in ' |"'):
        text = '"%s"' % text.replace("\\", "\\\\").replace('"', '\\"')
    return "%s=%s" % (key, text)


def render_row(row: dict) -> str:
    if row.get("blank"):
        return "## %s\nchecked=no\n" % row["id"]
    stages = row["stage_p50_ms"]
    parts = [
        _fmt("checked", "yes"),
        _fmt("worktree", row["worktree"]),
        _fmt("hardware", row["hardware"]),
        _fmt("date", row["date"]),
        _fmt("server_uri", row["server_uri"]),
        _fmt("conf_server", row["conf_server"]),
        _fmt("conf_server_worker_nums", row["conf_server_worker_nums"]),
        _fmt("worker_nums", row["worker_nums"]),
        _fmt("decode_auto", row["decode_auto"]),
        _fmt("prepare", row["prepare"]),
        _fmt("calibrate", row["calibrate"]),
        _fmt("calibrate_raw", row["calibrate_raw"]),
        _fmt("MORTRED_WORKER_NUMS", row["mortred_worker_nums"]),
        _fmt("MORTRED_DECODE_AUTO", row["mortred_decode_auto"]),
        _fmt("process_workers", row["process_workers"]),
        _fmt("process_workers_available", row["workers_available_after"]),
        _fmt("process_workers_busy", row["workers_busy_after"]),
        _fmt("process_workers_source", row["process_workers_source"]),
        _fmt("process_workers_note", row["process_workers_note"]),
        _fmt("load_command", row["load_command"]),
        _fmt("load_url", row["load_url"]),
        _fmt("wall_s", row["wall_s"]),
        _fmt("warmup_s", row["warmup_s"]),
        _fmt("success_rps", row["success_rps"]),
        _fmt("p50_ms", row["p50_ms"]),
        _fmt("p99_ms", row["p99_ms"]),
        _fmt("http_429", row["http_429"]),
        _fmt("error_rate", row["error_rate"]),
        _fmt("requests", row["requests"]),
        _fmt("errors", row["errors"]),
        _fmt("throughput_qps", row["throughput_qps"]),
        _fmt("ratio", row["ratio"]),
        _fmt("mortred_queue_wait_duration_ms_count_delta", row["queue_count_delta"]),
        _fmt("mortred_queue_wait_duration_ms_model", row["queue_model"]),
        _fmt("mortred_inference_duration_ms_count_delta", row["infer_count_delta"]),
        _fmt("mortred_inference_duration_ms_model", row["infer_model"]),
        _fmt("jpeg_jpeggpu", row["jpeg_jpeggpu"]),
        _fmt("jpeg_cpu_reduced", row["jpeg_cpu_reduced"]),
        _fmt("jpeg_cpu_full", row["jpeg_cpu_full"]),
        _fmt("jpeg_landing", row["jpeg_landing"]),
        _fmt("gateway_gw_in_p50_ms", stages["gw_in"]),
        _fmt("gateway_route_p50_ms", stages["route"]),
        _fmt("gateway_auth_p50_ms", stages["auth"]),
        _fmt("gateway_fwd_p50_ms", stages["fwd"]),
        _fmt("gateway_upstream_p50_ms", stages["upstream"]),
        _fmt("gateway_stage_p50_sum", row["stage_p50_sum"]),
        _fmt("server_time_ms_p50", row["server_time_ms_p50"]),
        _fmt("dominant_stage", row["dominant_stage"] if row["dominant_stage"] else "none"),
        _fmt("dominant_gauge", row["dominant_gauge"] or "absent"),
        _fmt("image", row.get("image") or "-"),
        _fmt("decoded_width", row.get("decoded_width") if row.get("decoded_width") is not None else "-"),
        _fmt("decoded_height", row.get("decoded_height") if row.get("decoded_height") is not None else "-"),
        _fmt("engine", row["engine"]),
    ]
    return "## %s\n%s\n" % (row["id"], " | ".join(parts))


def _sample_row(model_id: str, spec: dict | None = None, **overrides) -> dict:
    spec = MODEL_BY_ID[model_id] if spec is None else spec
    workers = 4
    throughput = YOLO_THROUGHPUT if model_id == "yolov8s" else 100.0
    rps = 50.0
    stages = {name: 0.2 for name in STAGES}
    stages["upstream"] = 1.5
    stage_sum = sum(stages.values())
    server_p50 = 1.0
    row = {
        "id": model_id,
        "blank": False,
        "worktree": "/mnt/g/Codex/mortred_wt_r1_p2_11",
        "hardware": "NVIDIA Test, 8192 MiB",
        "date": "2026-10-14",
        "server_uri": spec["server_uri"],
        "conf_server": spec["conf_server"],
        "conf_server_worker_nums": 1,
        "worker_nums": workers,
        "decode_auto": "cpu",
        "prepare": "mortredctl prepare --id %s" % model_id,
        "calibrate": "mortredctl calibrate --pack /tmp/pack.toml --write-pack --duration 60s",
        "calibrate_raw": "worker sweep and jpeg decode vote use --raw",
        "mortred_worker_nums": workers,
        "mortred_decode_auto": "cpu",
        "process_workers": workers,
        "process_workers_source": "after_load",
        "process_workers_note": (
            "官方压测产生完成响应且进程仍在时，mortred_workers_available 与 "
            "mortred_workers_busy 之和；不是压测前刮取"
        ),
        "workers_available_after": float(workers),
        "workers_busy_after": 0.0,
        "load_command": (
            "python3 scripts/server/http_infer_rps.py --raw --model %s "
            "--gateway http://127.0.0.1:8080 -c %d -d 60s --warmup 5s"
            % (model_id, workers * 2)
        ),
        "load_url": "http://127.0.0.1:8080/v1/models/%s/infer" % model_id,
        "wall_s": 60.01,
        "warmup_s": 5.0,
        "success_rps": rps,
        "p50_ms": 2.0,
        "p99_ms": 8.0,
        "http_429": 0,
        "error_rate": 0.0,
        "requests": 1000,
        "errors": 0,
        "throughput_qps": throughput,
        "ratio": rps / throughput,
        "queue_count_delta": 1000.0,
        "queue_model": spec["server_uri"],
        "infer_count_delta": 1000.0,
        "infer_model": spec["server_uri"],
        "jpeg_jpeggpu": 0.0,
        "jpeg_cpu_reduced": 1000.0,
        "jpeg_cpu_full": 0.0,
        "jpeg_landing": "cpu",
        "stage_p50_ms": stages,
        "stage_p50_sum": stage_sum,
        "server_time_ms_p50": server_p50,
        "dominant_stage": "upstream",
        "dominant_gauge": "absent",
        "image": "-",
        "decoded_width": None,
        "decoded_height": None,
        "engine": "static shape batch=1",
    }
    if model_id == "pphuman_mobile":
        row["image"] = FACE_IMAGE
    if model_id == "dbnet":
        row["image"] = DBNET_IMAGE
    if model_id == "superpoint":
        row["image"] = SUPERPOINT_IMAGE
    if model_id == "metric3d_512":
        row["image"] = METRIC3D_IMAGE
    if model_id == "realesrgan":
        row["image"] = REALESRGAN_IMAGE
        row["decoded_width"] = 256
        row["decoded_height"] = 256
    row.update(overrides)
    if "stage_p50_ms" in overrides:
        row["stage_p50_sum"] = sum(float(overrides["stage_p50_ms"][name]) for name in STAGES)
    if "success_rps" in overrides or "throughput_qps" in overrides:
        row["ratio"] = float(row["success_rps"]) / float(row["throughput_qps"])
    return row


def _expect_fail(row: dict, label: str, specs: dict | None = None) -> str | None:
    ok, _why = judge_row(row, specs)
    if ok:
        return "expected fail: %s" % label
    return None


def parse_kv_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i] in " |":
            i += 1
        if i >= n:
            break
        eq = line.find("=", i)
        if eq < 0:
            break
        key = line[i:eq].strip()
        if not key or any(ch.isspace() for ch in key):
            break
        i = eq + 1
        if i < n and line[i] == '"':
            i += 1
            buf: list[str] = []
            while i < n:
                if line[i] == "\\" and i + 1 < n:
                    buf.append(line[i + 1])
                    i += 2
                    continue
                if line[i] == '"':
                    i += 1
                    break
                buf.append(line[i])
                i += 1
            fields[key] = "".join(buf)
            continue
        end = line.find(" | ", i)
        if end < 0:
            fields[key] = line[i:].strip()
            break
        fields[key] = line[i:end].strip()
        i = end
    return fields


def parse_report_sections(text: str) -> list[tuple[str, dict[str, str]]]:
    matches = list(re.finditer(r"(?m)^## ([A-Za-z0-9_]+)\s*$", text))
    sections = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        line = ""
        for raw in body.splitlines():
            if raw.strip():
                line = raw.strip()
                break
        sections.append((match.group(1), parse_kv_line(line) if line else {}))
    return sections


def section_kind(fields: dict[str, str]) -> str:
    """blocked, blank, pass-claim, or invalid.

    A blocked row that also carries a latency or RPS field is not blocked:
    that would hide a number that was not judged.
    """
    checked = fields.get("checked", "")
    status = fields.get("status", "")
    evidence = fields.get("evidence", "")
    has_metric = any(key in fields for key in ("success_rps", "p50_ms", "p99_ms", "throughput_qps"))
    if status == "blocked" and evidence == "missing" and checked == "no" and not has_metric:
        if fields.get("pf1", "not-pass") == "not-pass":
            return "blocked"
        return "invalid"
    if checked == "no" and not has_metric and status == "" and evidence == "":
        return "blank"
    if checked == "yes" or status in ("pass", "PASS") or has_metric:
        return "pass-claim"
    return "invalid"


def _as_int(text: str) -> int:
    value = float(text)
    if value != int(value):
        raise ValueError(text)
    return int(value)


def _as_float(text: str) -> float:
    return float(text)


def row_from_fields(model_id: str, fields: dict[str, str]) -> dict:
    required = (
        "worktree",
        "hardware",
        "date",
        "server_uri",
        "conf_server",
        "conf_server_worker_nums",
        "worker_nums",
        "decode_auto",
        "prepare",
        "calibrate",
        "calibrate_raw",
        "MORTRED_WORKER_NUMS",
        "MORTRED_DECODE_AUTO",
        "process_workers",
        "process_workers_available",
        "process_workers_busy",
        "process_workers_source",
        "process_workers_note",
        "load_command",
        "load_url",
        "wall_s",
        "warmup_s",
        "success_rps",
        "p50_ms",
        "p99_ms",
        "http_429",
        "error_rate",
        "requests",
        "errors",
        "throughput_qps",
        "ratio",
        "mortred_queue_wait_duration_ms_count_delta",
        "mortred_queue_wait_duration_ms_model",
        "mortred_inference_duration_ms_count_delta",
        "mortred_inference_duration_ms_model",
        "jpeg_jpeggpu",
        "jpeg_cpu_reduced",
        "jpeg_cpu_full",
        "jpeg_landing",
        "gateway_gw_in_p50_ms",
        "gateway_route_p50_ms",
        "gateway_auth_p50_ms",
        "gateway_fwd_p50_ms",
        "gateway_upstream_p50_ms",
        "gateway_stage_p50_sum",
        "server_time_ms_p50",
        "dominant_stage",
        "dominant_gauge",
        "image",
        "decoded_width",
        "decoded_height",
        "engine",
    )
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError("missing " + ",".join(missing))
    dominant = fields["dominant_stage"]
    if dominant == "none":
        dominant = ""
    width = fields["decoded_width"]
    height = fields["decoded_height"]
    return {
        "id": model_id,
        "blank": False,
        "worktree": fields["worktree"],
        "hardware": fields["hardware"],
        "date": fields["date"],
        "server_uri": fields["server_uri"],
        "conf_server": fields["conf_server"],
        "conf_server_worker_nums": _as_int(fields["conf_server_worker_nums"]),
        "worker_nums": _as_int(fields["worker_nums"]),
        "decode_auto": fields["decode_auto"],
        "prepare": fields["prepare"],
        "calibrate": fields["calibrate"],
        "calibrate_raw": fields["calibrate_raw"],
        "mortred_worker_nums": _as_int(fields["MORTRED_WORKER_NUMS"]),
        "mortred_decode_auto": fields["MORTRED_DECODE_AUTO"],
        "process_workers": _as_float(fields["process_workers"]),
        "process_workers_source": fields["process_workers_source"],
        "process_workers_note": fields["process_workers_note"],
        "workers_available_after": _as_float(fields["process_workers_available"]),
        "workers_busy_after": _as_float(fields["process_workers_busy"]),
        "load_command": fields["load_command"],
        "load_url": fields["load_url"],
        "wall_s": _as_float(fields["wall_s"]),
        "warmup_s": _as_float(fields["warmup_s"]),
        "success_rps": _as_float(fields["success_rps"]),
        "p50_ms": _as_float(fields["p50_ms"]),
        "p99_ms": _as_float(fields["p99_ms"]),
        "http_429": _as_int(fields["http_429"]),
        "error_rate": _as_float(fields["error_rate"]),
        "requests": _as_int(fields["requests"]),
        "errors": _as_int(fields["errors"]),
        "throughput_qps": _as_float(fields["throughput_qps"]),
        "ratio": _as_float(fields["ratio"]),
        "queue_count_delta": _as_float(fields["mortred_queue_wait_duration_ms_count_delta"]),
        "queue_model": fields["mortred_queue_wait_duration_ms_model"],
        "infer_count_delta": _as_float(fields["mortred_inference_duration_ms_count_delta"]),
        "infer_model": fields["mortred_inference_duration_ms_model"],
        "jpeg_jpeggpu": _as_float(fields["jpeg_jpeggpu"]),
        "jpeg_cpu_reduced": _as_float(fields["jpeg_cpu_reduced"]),
        "jpeg_cpu_full": _as_float(fields["jpeg_cpu_full"]),
        "jpeg_landing": fields["jpeg_landing"],
        "stage_p50_ms": {
            "gw_in": _as_float(fields["gateway_gw_in_p50_ms"]),
            "route": _as_float(fields["gateway_route_p50_ms"]),
            "auth": _as_float(fields["gateway_auth_p50_ms"]),
            "fwd": _as_float(fields["gateway_fwd_p50_ms"]),
            "upstream": _as_float(fields["gateway_upstream_p50_ms"]),
        },
        "stage_p50_sum": _as_float(fields["gateway_stage_p50_sum"]),
        "server_time_ms_p50": _as_float(fields["server_time_ms_p50"]),
        "dominant_stage": dominant,
        "dominant_gauge": fields["dominant_gauge"],
        "image": fields["image"],
        "decoded_width": None if width == "-" else _as_int(width),
        "decoded_height": None if height == "-" else _as_int(height),
        "engine": fields["engine"],
    }


def record_pf5_baseline(row: dict, source: str, dest: Path | None = None) -> str:
    """Write docs/perf/baselines/<id>.json the first time. Keep it afterwards.

    Same hardware keeps the recorded numbers. A different hardware string
    refuses and does not rewrite the file. Schema matches pf5-baseline-v1
    so a later PF5 checker can read the file without this driver importing it.
    """
    directory = BASELINE_DIR if dest is None else dest
    directory.mkdir(parents=True, exist_ok=True)
    image = row.get("image")
    if image in (None, "", "-"):
        image = None
    workers = int(row["worker_nums"])
    doc = {
        "schema": "pf5-baseline-v1",
        "id": row["id"],
        "source": source,
        "date": row["date"],
        "hardware": row["hardware"],
        "worker_nums": workers,
        "decode_auto": row["decode_auto"],
        "success_rps": row["success_rps"],
        "p50_ms": row["p50_ms"],
        "p99_ms": row["p99_ms"],
        "concurrency": workers * 2,
        "duration": "60s",
        "warmup": "5s",
        "warmup_s": row["warmup_s"],
        "wall_s": row["wall_s"],
        "error_rate": row["error_rate"],
        "load_command": row["load_command"],
        "load_url": row["load_url"],
        "image": image,
        "server_uri": row["server_uri"],
        "conf_server": row["conf_server"],
        "engine": row["engine"],
    }
    path = directory / ("%s.json" % row["id"])
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("baseline %s is not json: %s" % (path.name, exc)) from exc
        if existing.get("hardware") != doc["hardware"]:
            raise RuntimeError(
                "baseline hardware mismatch for %s: have %s, current %s (baseline not rewritten)"
                % (row["id"], existing.get("hardware"), doc["hardware"])
            )
        return "kept"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return "created"


def _status_table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["id\tstatus\tdetail"]
    for model_id, status, detail in rows:
        lines.append("%s\t%s\t%s" % (model_id, status, detail))
    return "\n".join(lines)


def check_catalog(catalog: str, skeleton: bool, quiet: bool = False) -> int:
    """Validate a checklist in place. Does not write it.

    skeleton=False requires every section to be a qualified PASS row.
    skeleton=True allows explicit blocked/missing sections and still judges
    any section that claims PASS.
    """
    specs = specs_for(catalog)
    path = report_path_for(catalog)
    before = path.read_bytes() if path.is_file() else None
    label = "--check-skeleton" if skeleton else "--check"
    missing_caps = missing_p99_caps([row["id"] for row in specs])
    if missing_caps:
        if not quiet:
            sys.stderr.write(format_missing_p99_table(missing_caps))
        return _fail("%s %s: missing p99 caps\n" % (catalog, label))
    mismatch = contract_ok(specs)
    if mismatch:
        return _fail("%s %s: %s\n" % (catalog, label, mismatch))
    if not path.is_file():
        return _fail("%s %s: %s is missing\n" % (catalog, label, path))
    sections = parse_report_sections(path.read_text(encoding="utf-8"))
    found = [model_id for model_id, _fields in sections]
    expected = [row["id"] for row in specs]
    if found != expected:
        if not quiet:
            sys.stderr.write(
                "%s %s: section ids\nfound %s\nexpected %s\n" % (catalog, label, found, expected)
            )
        return _fail("%s %s: section ids do not match the catalog\n" % (catalog, label))
    table = spec_map_for(specs)
    summary: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str, str]] = []
    for model_id, fields in sections:
        kind = section_kind(fields)
        if kind == "blocked":
            summary.append((model_id, "blocked", "evidence=missing"))
            if not skeleton:
                failures.append((model_id, "blocked", "not PASS"))
            continue
        if kind == "blank":
            if catalog == "coverage_set" and model_id == "libface":
                ok, why = judge_row({"id": "libface", "blank": True}, table)
                if not ok:
                    failures.append((model_id, "blank", why))
                else:
                    summary.append((model_id, "blank", "libface"))
                continue
            failures.append((model_id, "blank", "not PASS"))
            summary.append((model_id, "blank", "not PASS"))
            continue
        if kind != "pass-claim":
            failures.append((model_id, "invalid", "not blocked and not PASS"))
            summary.append((model_id, "invalid", "not blocked and not PASS"))
            continue
        try:
            row = row_from_fields(model_id, fields)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append((model_id, "invalid", str(exc)))
            summary.append((model_id, "invalid", str(exc)))
            continue
        ok, why = judge_row(row, table)
        if not ok:
            failures.append((model_id, "invalid", why))
            summary.append((model_id, "invalid", why))
            continue
        summary.append((model_id, "PASS", "qualified"))
    after = path.read_bytes() if path.is_file() else None
    if before != after:
        return _fail("%s %s: wrote the checklist\n" % (catalog, label))
    passed = sum(1 for _model_id, status, _detail in summary if status == "PASS")
    blocked = sum(1 for _model_id, status, _detail in summary if status == "blocked")
    if failures:
        if not quiet:
            sys.stderr.write(
                "%s %s: %d PASS, %d blocked/missing, %d not accepted\n"
                % (catalog, label, passed, blocked, len(failures))
            )
            sys.stderr.write(
                "blocked/missing is not PASS. No HTTP RPS numbers were invented.\n"
            )
            sys.stderr.write(_status_table(failures) + "\n")
        return 1
    print(
        "%s %s passed (%d PASS, %d blocked/missing, %d sections)"
        % (catalog, label, passed, blocked, len(summary))
    )
    return 0


def render_blocked_section(model_id: str) -> str:
    reason = (
        "no live GPU HTTP RPS; run "
        "python3 scripts/coverage_http_report.py --catalog product_static --id %s "
        "on a machine with nvidia-smi" % model_id
    )
    return (
        "## %s\nstatus=blocked | evidence=missing | checked=no | pf1=not-pass | reason=%s\n"
        % (model_id, _fmt("reason", reason).split("=", 1)[1])
    )


def render_product_skeleton() -> str:
    specs = product_static_specs()
    lines = [
        "# product static HTTP RPS",
        "",
        "Catalog: `conf/product_static_bs1.json` (%d product TensorRT ids, JSON order)."
        % len(specs),
        "",
        "This file is a skeleton. It is not PF1 or TP1 evidence. No success RPS,",
        "p50, or p99 is recorded here. Each section stays `status=blocked` and",
        "`evidence=missing` until a GPU host runs the driver and `judge_row` accepts",
        "the row. A blocked section that also contains `success_rps`, `p50_ms`,",
        "`p99_ms`, or `throughput_qps` is rejected.",
        "",
        "Policy:",
        "",
        "- `--catalog product_static --check` exits non-zero until all %d sections are qualified PASS rows."
        % len(specs),
        "- `--catalog product_static --check-skeleton` exits 0 when the headings match",
        "  JSON order and every section is either explicit blocked/missing or a qualified PASS row.",
        "- Coverage-set nine stay in `docs/perf/coverage_set_http_rps.md` (ops / TP2 subset).",
        "  Those measured rows are not copied here, so this file does not claim PASS for them.",
        "",
        "Operator procedure on a GPU host (solo residency; do not require the other ids",
        "to be resident). The driver does not write a PASS row unless `judge_row` accepts it:",
        "",
        "1. `mortredctl prepare --id <id>`",
        "2. One-id pack. If `conf/packs/<id>.toml` exists and lists only that id, it is",
        "   the temp pack seed (per-id packs from the pack work). Otherwise the driver",
        "   writes `[pack.<id>] worker_nums = 1`. Git packs are not modified.",
        "3. `mortredctl calibrate --pack <temp> --write-pack --duration 60s`",
        "   (the calibrator posts with `http_infer_rps.py --raw`).",
        "4. Deploy that pack alone (`MORTRED_PACK` is the temp pack).",
        "5. `python3 scripts/server/http_infer_rps.py --raw --model <id>`",
        "   `--gateway http://127.0.0.1:8080 -c <worker_nums×2> -d 60s --warmup 5s`",
        "",
        "60s meets the ≥30s bar. Concurrency is the deployed pack `worker_nums` times 2.",
        "`conf/server` stays `worker_nums=1`.",
        "",
        "```bash",
        "python3 scripts/coverage_http_report.py --catalog product_static --id yolov8n",
        "python3 scripts/coverage_http_report.py --catalog product_static --from yolov8n",
        "python3 scripts/coverage_http_report.py --catalog product_static --check-skeleton",
        "python3 scripts/coverage_http_report.py --catalog product_static --check",
        "```",
        "",
        "A PASS row also writes `docs/perf/baselines/<id>.json` (`schema` `pf5-baseline-v1`)",
        "the first time, and keeps that file when the hardware string matches. No baseline",
        "file is written for a blocked section.",
        "",
        "Seed p99 caps for every product id live in `scripts/calibrate_pack.py` `P99_CAP_MS`.",
        "A missing cap aborts calibrate with a table. The latency gate is not skipped.",
        "",
    ]
    for spec in specs:
        lines.append(render_blocked_section(spec["id"]).rstrip("\n"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_self_test() -> int:
    before = REPORT.read_bytes() if REPORT.is_file() else None
    product_before = PRODUCT_REPORT.read_bytes() if PRODUCT_REPORT.is_file() else None
    baseline_before = (
        sorted(path.name for path in BASELINE_DIR.glob("*")) if BASELINE_DIR.is_dir() else None
    )
    mismatch = contract_ok()
    if mismatch:
        return _fail("check: %s\n" % mismatch)
    try:
        if read_throughput("yolov8s") != YOLO_THROUGHPUT:
            return _fail("check: yolov8s throughput file\n")
    except (OSError, RuntimeError) as exc:
        return _fail("check: %s\n" % exc)

    uri = MODEL_BY_ID["mobilenetv2"]["server_uri"]
    after = (
        'mortred_queue_wait_duration_ms_bucket{model="%s",le="1"} 8\n'
        'mortred_queue_wait_duration_ms_count{model="%s"} 8\n'
        'mortred_inference_duration_ms_bucket{model="%s",le="1"} 8\n'
        'mortred_inference_duration_ms_count{model="%s"} 8\n'
    ) % (uri, uri, uri, uri)
    directory = after.replace(uri, "mobilenetv2")
    if hist_count_delta("", after, "mortred_queue_wait_duration_ms", uri) != 8.0:
        return _fail("check: empty before delta\n")
    if hist_count_delta("", directory, "mortred_queue_wait_duration_ms", uri) is not None:
        return _fail("check: directory id histogram counted\n")
    if _gauge(directory, "mortred_workers_available", uri) is not None:
        return _fail("check: directory id worker gauge counted\n")

    rows = [_sample_row(spec["id"]) for spec in MODELS]
    rows[2] = {"id": "libface", "blank": True}
    ok, why = judge_document(rows)
    if not ok:
        return _fail("check: blank libface document: %s\n" % why)
    full = [_sample_row(spec["id"]) for spec in MODELS]
    ok, why = judge_document(full)
    if not ok:
        return _fail("check: full document: %s\n" % why)

    misplaced = list(full)
    misplaced[0] = {"id": "mobilenetv2", "blank": True}
    ok, _why = judge_document(misplaced)
    if ok:
        return _fail("check: non-libface blank passed\n")

    cases = []
    preload = _sample_row("mobilenetv2", process_workers=0, workers_available_after=0, workers_busy_after=0)
    cases.append(_expect_fail(preload, "preload zero"))
    cases.append(
        _expect_fail(
            _sample_row("mobilenetv2", queue_model="mobilenetv2", infer_model="mobilenetv2"),
            "directory id",
        )
    )
    cases.append(_expect_fail(_sample_row("mobilenetv2", infer_count_delta=None), "missing histogram"))
    folded = _sample_row(
        "mobilenetv2", requests=100, errors=2, http_429=2, error_rate=0.02, success_rps=10.0
    )
    cases.append(_expect_fail(folded, "429 folded"))
    cases.append(_expect_fail(_sample_row("mobilenetv2", wall_s=59.0), "short wall"))
    cases.append(_expect_fail(_sample_row("mobilenetv2", warmup_s=0.0), "warmup"))
    short_c = _sample_row("mobilenetv2")
    short_c["load_command"] = short_c["load_command"].replace("-c 8", "-c 4")
    cases.append(_expect_fail(short_c, "concurrency"))
    cases.append(_expect_fail(_sample_row("yolov8s", throughput_qps=400.0), "yolov8s denominator"))
    cases.append(_expect_fail(_sample_row("mobilenetv2", decode_auto="auto"), "decode_auto"))
    cases.append(
        _expect_fail(
            _sample_row("pphuman_mobile", image="demo_data/model_test_input/scene_segmentation/human_image.jpg"),
            "pphuman image",
        )
    )
    cases.append(
        _expect_fail(
            _sample_row("dbnet", image="demo_data/model_test_input/ocr/railway_ticket.png"),
            "dbnet image",
        )
    )
    cases.append(
        _expect_fail(
            _sample_row("superpoint", image="demo_data/model_test_input/feature_point/test.png"),
            "superpoint image",
        )
    )
    cases.append(
        _expect_fail(
            _sample_row(
                "metric3d_512",
                image="demo_data/model_test_input/mono_depth_estimation/0000000005.png",
            ),
            "metric3d image",
        )
    )
    equal = _sample_row(
        "mobilenetv2", requests=10000, errors=100, http_429=0, error_rate=0.01, success_rps=20.0
    )
    ok, why = judge_row(equal)
    if not ok:
        return _fail("check: 1%% error rate should pass (%s)\n" % why)
    gpu_row = _sample_row(
        "pphuman_mobile",
        decode_auto="gpu",
        mortred_decode_auto="gpu",
        jpeg_jpeggpu=10.0,
        jpeg_cpu_reduced=1.0,
        jpeg_cpu_full=0.0,
        jpeg_landing="gpu",
    )
    ok, why = judge_row(gpu_row)
    if not ok:
        return _fail("check: gpu landing should pass (%s)\n" % why)
    cases.append(
        _expect_fail(
            _sample_row(
                "pphuman_mobile",
                decode_auto="gpu",
                mortred_decode_auto="gpu",
                jpeg_jpeggpu=1.0,
                jpeg_cpu_reduced=1.0,
                jpeg_cpu_full=1.0,
                jpeg_landing="gpu",
            ),
            "gpu not strictly greater",
        )
    )
    for item in cases:
        if item:
            return _fail("self-test: %s\n" % item)

    product_specs = product_static_specs()
    if len(product_specs) != 39:
        return _fail("self-test: product catalog length %s\n" % len(product_specs))
    missing_caps = missing_p99_caps([row["id"] for row in product_specs])
    if missing_caps:
        sys.stderr.write(format_missing_p99_table(missing_caps))
        return _fail("self-test: product ids missing p99 caps\n")
    product_table = spec_map_for(product_specs)
    yolov8n = product_table["yolov8n"]
    synthetic = _sample_row("yolov8n", spec=yolov8n)
    ok, why = judge_row(synthetic, product_table)
    if not ok:
        return _fail("self-test: synthetic yolov8n should pass (%s)\n" % why)
    short_product = _sample_row("yolov8n", spec=yolov8n)
    short_product["load_command"] = short_product["load_command"].replace("-c 8", "-c 4")
    ok, why = judge_row(short_product, product_table)
    if ok or why != "load concurrency":
        return _fail("self-test: product concurrency should fail (%s)\n" % why)
    # judge_row with the coverage map must reject an id outside the nine.
    ok, why = judge_row(synthetic, MODEL_BY_ID)
    if ok or why != "unknown id":
        return _fail("self-test: yolov8n must be unknown to the coverage map (%s)\n" % why)
    blocked_only = {
        "id": "yolov8n",
        "status": "blocked",
        "evidence": "missing",
        "checked": "no",
    }
    if section_kind(blocked_only) != "blocked":
        return _fail("self-test: blocked section kind\n")
    fake = dict(blocked_only)
    fake["success_rps"] = "1"
    if section_kind(fake) == "blocked":
        return _fail("self-test: blocked row with success_rps must not count as blocked\n")
    skeleton_text = PRODUCT_REPORT.read_text(encoding="utf-8") if PRODUCT_REPORT.is_file() else ""
    for model_id, fields in parse_report_sections(skeleton_text):
        if any(key in fields for key in ("success_rps", "p50_ms", "p99_ms", "throughput_qps")):
            return _fail("self-test: product skeleton %s contains an RPS field\n" % model_id)
    skeleton_code = check_catalog("product_static", skeleton=True)
    if skeleton_code != 0:
        return _fail("self-test: product --check-skeleton failed\n")
    evidence_code = check_catalog("product_static", skeleton=False, quiet=True)
    if evidence_code == 0:
        return _fail("self-test: product --check must fail while rows are blocked\n")
    coverage_code = check_catalog("coverage_set", skeleton=False)
    if coverage_code != 0:
        return _fail("self-test: coverage --check failed\n")

    with tempfile.TemporaryDirectory(prefix="mortred-pf5-") as tmp:
        dest = Path(tmp)
        status = record_pf5_baseline(synthetic, "docs/perf/product_static_http_rps.md", dest)
        if status != "created":
            return _fail("self-test: pf5 baseline was not created\n")
        written = json.loads((dest / "yolov8n.json").read_text(encoding="utf-8"))
        if written.get("schema") != "pf5-baseline-v1" or written.get("success_rps") != synthetic["success_rps"]:
            return _fail("self-test: pf5 baseline shape\n")
        if written.get("concurrency") != synthetic["worker_nums"] * 2:
            return _fail("self-test: pf5 baseline concurrency\n")
        changed = dict(synthetic)
        changed["success_rps"] = 1.0
        changed["ratio"] = 1.0 / float(changed["throughput_qps"])
        if record_pf5_baseline(changed, "docs/perf/product_static_http_rps.md", dest) != "kept":
            return _fail("self-test: pf5 baseline was not kept\n")
        again = json.loads((dest / "yolov8n.json").read_text(encoding="utf-8"))
        if again.get("success_rps") != synthetic["success_rps"]:
            return _fail("self-test: pf5 baseline numbers were rewritten\n")
        try:
            record_pf5_baseline(
                dict(synthetic, hardware="Different GPU, 1 MiB"),
                "docs/perf/product_static_http_rps.md",
                dest,
            )
        except RuntimeError as exc:
            if "hardware mismatch" not in str(exc):
                return _fail("self-test: pf5 hardware mismatch message: %s\n" % exc)
        else:
            return _fail("self-test: pf5 hardware mismatch must refuse\n")
        after_hw = json.loads((dest / "yolov8n.json").read_text(encoding="utf-8"))
        if after_hw.get("hardware") != synthetic["hardware"]:
            return _fail("self-test: pf5 hardware mismatch rewrote the baseline\n")

    after_bytes = REPORT.read_bytes() if REPORT.is_file() else None
    if before != after_bytes:
        return _fail("self-test: wrote the coverage checklist\n")
    product_after = PRODUCT_REPORT.read_bytes() if PRODUCT_REPORT.is_file() else None
    if product_before != product_after:
        return _fail("self-test: wrote the product checklist\n")
    baseline_after = (
        sorted(path.name for path in BASELINE_DIR.glob("*")) if BASELINE_DIR.is_dir() else None
    )
    if baseline_before != baseline_after:
        return _fail("self-test: wrote docs/perf/baselines\n")
    print("coverage_http_report.py --self-test passed")
    print(
        "fixtures: judge_row concurrency, error_rate fold, jpeg landing, "
        "blank libface, yolov8s throughput, product blocked is not PASS, "
        "missing p99 cap, pf5 baseline create/keep/refuse"
    )
    return 0


def accepting(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        sock.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def bindable(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def port_block_reason(port: int) -> str | None:
    if accepting(port):
        return "%d accepting" % port
    if not bindable(port):
        return "%d not bindable" % port
    return None


def load_tokens() -> dict[str, str] | None:
    found = {}
    for key in TOKEN_KEYS:
        value = os.environ.get(key, "")
        if value:
            found[key] = value
    if all(key in found for key in TOKEN_KEYS):
        return found
    path = ROOT / "conf" / "local" / "trust.env"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key in TOKEN_KEYS and key not in found and value:
                found[key] = value
    if any(key not in found for key in TOKEN_KEYS):
        return None
    return found


def base_env(tokens: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["MORTRED_PROJECT_ROOT"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["LD_LIBRARY_PATH"] = "%s:%s:%s" % (
        ROOT / "_lib",
        ROOT / "3rd_party" / "libs",
        env.get("LD_LIBRARY_PATH", ""),
    )
    for key in (
        "MORTRED_DEMO_IMAGE",
        "MORTRED_PACK",
        "MORTRED_MODEL_CONFIG_FILE",
        "MORTRED_WORKER_NUMS",
        "MORTRED_DECODE_AUTO",
        "MORTRED_LISTEN_PORT",
        "MORTRED_AUTOSTART",
    ):
        env.pop(key, None)
    if tokens:
        for key, value in tokens.items():
            env[key] = value
    return env


def http_get(port: int, path: str, token: str, timeout_s: float = 5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        conn.request(
            "GET",
            path,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "text/plain, application/json",
                "Connection": "keep-alive",
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        status = resp.status
        close_client_first(conn, drain_s=1.0)
        conn = None
        return status, body
    except (OSError, http.client.HTTPException, TimeoutError):
        return None
    finally:
        if conn is not None:
            close_client_first(conn, drain_s=0.0)


def fetch_status(token: str):
    got = http_get(8787, "/api/v1/status", token)
    if got is None or got[0] != 200:
        return None
    try:
        doc = json.loads(got[1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def one_server(doc: dict, model_id: str):
    servers = doc.get("servers")
    if not isinstance(servers, list):
        return None
    found = [item for item in servers if isinstance(item, dict) and item.get("id") == model_id]
    if len(found) != 1:
        return None
    return found[0]


def read_environ(pid: int) -> dict[str, str]:
    raw = Path("/proc/%d/environ" % pid).read_bytes()
    out = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_supervisor(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.kill(proc.pid, signal.SIGINT)
    except OSError:
        return
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.kill(proc.pid, signal.SIGKILL)
    except OSError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def wait_ports_clear(ports: list[int], timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(port_block_reason(port) is None for port in ports):
            return True
        time.sleep(0.2)
    return all(port_block_reason(port) is None for port in ports)


def run_logged(argv: list[str], env: dict[str, str], log_path: Path, timeout: float | None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(("\n+ %s\n" % " ".join(argv)).encode())
        handle.flush()
        proc = subprocess.run(
            argv,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return proc.returncode


def log_tail(path: Path, n: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def ensure_realesrgan_jpeg(path: Path) -> None:
    if path.is_file():
        data = path.read_bytes()
        if jpeg_sof0_size(data) == (256, 256) and b"Exif" not in data:
            return
    import cv2
    import numpy as np

    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[:, :, 0] = 40
    frame[:, :, 1] = 90
    frame[:, :, 2] = 140
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError("failed to write %s" % path)
    data = path.read_bytes()
    if jpeg_sof0_size(data) != (256, 256) or b"Exif" in data:
        raise RuntimeError("realesrgan jpeg is not a 256x256 baseline scan")


def hardware_line() -> str:
    gpu = gpu_inventory()
    if not isinstance(gpu, dict) or not str(gpu.get("name") or "").strip():
        raise RuntimeError("nvidia-smi did not return a GPU name")
    total = gpu.get("memory_total_mib")
    if not isinstance(total, (int, float)) or float(total) <= 0:
        raise RuntimeError("nvidia-smi did not return GPU memory")
    shown = int(total) if float(total) == int(total) else total
    return "%s, %s MiB" % (str(gpu["name"]).strip(), shown)


def append_text(text: str, report: Path = REPORT) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def reset_report(hardware: str, today: str, report: Path = REPORT) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# coverage set HTTP RPS\n\n"
        "worktree: %s\n"
        "hardware: %s\n"
        "date: %s\n\n"
        % (ROOT, hardware, today)
    )
    report.write_text(text, encoding="utf-8")


def _integral(value: float):
    if float(value) == int(value):
        return int(value)
    return value


def run_id(
    spec: dict,
    tokens: dict[str, str],
    hardware: str,
    today: str,
    ctl: Path,
    baseline_source: str,
):
    model_id = spec["id"]
    ports = [8080, 8787, spec["port"]]
    blocked = [port_block_reason(port) for port in ports]
    blocked = [item for item in blocked if item]
    if blocked:
        return "fail", "port not free: %s" % ", ".join(blocked)

    log_path = ROOT / "logs" / ("p2-11-%s.log" % model_id)
    env = base_env()
    image_rel = load_image_rel(spec)
    if image_rel:
        env["MORTRED_DEMO_IMAGE"] = image_rel

    sys.stderr.write("== %s prepare ==\n" % model_id)
    sys.stderr.flush()
    try:
        code = run_logged([str(ctl), "prepare", "--id", model_id], env, log_path, 1800)
    except subprocess.TimeoutExpired:
        return "fail", "prepare timed out"
    if code != 0:
        return "no-rps", "prepare exited %s\n%s" % (code, log_tail(log_path))

    tmpdir = tempfile.mkdtemp(prefix="mortred-p2-11-%s-" % model_id)
    pack = Path(tmpdir) / "pack.toml"
    try:
        pack_text = solo_pack_text(model_id)
    except RuntimeError as exc:
        return "fail", str(exc)
    if str(ROOT.resolve()) in str(pack.resolve()):
        return "fail", "temporary pack path"
    if "model_config" in pack_text:
        return "fail", "solo pack must not set model_config"
    pack.write_text(pack_text, encoding="utf-8")
    calibrate = "mortredctl calibrate --pack %s --write-pack --duration 60s" % pack
    sys.stderr.write("== %s calibrate ==\n" % model_id)
    sys.stderr.flush()
    try:
        code = run_logged(
            [str(ctl), "calibrate", "--pack", str(pack), "--write-pack", "--duration", "60s"],
            env,
            log_path,
            None,
        )
    except subprocess.TimeoutExpired:
        return "fail", "calibrate timed out"
    if code != 0:
        return "no-rps", "calibrate exited %s\n%s" % (code, log_tail(log_path))
    section = pack_table(pack, model_id)
    workers = section.get("worker_nums")
    decode_auto = section.get("decode_auto")
    if not isinstance(workers, int) or isinstance(workers, bool) or decode_auto not in ("cpu", "gpu"):
        return "no-rps", "pack did not write worker_nums and decode_auto (%s)" % section
    if "model_config" in section:
        return "fail", "pack wrote model_config"

    load_cmd = [
        "python3",
        "scripts/server/http_infer_rps.py",
        "--raw",
        "--model",
        model_id,
        "--gateway",
        GATEWAY,
        "-c",
        str(workers * 2),
        "-d",
        "60s",
        "--warmup",
        "5s",
    ]
    if image_rel:
        load_cmd.extend(["--image", image_rel])
    out_path = Path(tmpdir) / "load.json"
    load_cmd.extend(["--out", str(out_path), "--quiet"])

    sup_log = ROOT / "logs" / ("p2-11-supervisor-%s.log" % model_id)
    sup_log.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(sup_log, "wb")
    proc = None
    try:
        sys.stderr.write("== %s deploy ==\n" % model_id)
        sys.stderr.flush()
        proc = subprocess.Popen(
            [str(ROOT / "_bin" / "mortred-supervisor.out")],
            cwd=str(ROOT),
            env=base_env(tokens) | {"MORTRED_PACK": str(pack)},
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 90.0
        item = None
        last_state = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return "no-rps", "supervisor exited before ready\n%s" % log_tail(sup_log)
            doc = fetch_status(tokens["MORTRED_API_TOKEN"])
            if doc is not None:
                current = one_server(doc, model_id)
                if current is not None:
                    last_state = str(current.get("state"))
                    if current.get("state") == "failed":
                        return "no-rps", "model failed: %s" % current.get("error")
                    gateway = doc.get("gateway")
                    gateway_ready = (
                        isinstance(gateway, dict)
                        and gateway.get("id") == "__gateway"
                        and gateway.get("state") == "running"
                        and gateway.get("ready") is True
                    )
                    pid = current.get("pid")
                    if (
                        gateway_ready
                        and current.get("ready") is True
                        and current.get("state") == "running"
                        and isinstance(pid, int)
                        and not isinstance(pid, bool)
                        and pid > 0
                    ):
                        item = current
                        break
            time.sleep(0.3)
        if item is None:
            return "no-rps", "model not ready (%s)" % last_state
        pid = int(item["pid"])
        try:
            child_env = read_environ(pid)
        except OSError as exc:
            return "fail", "cannot read process environ: %s" % exc
        if child_env.get("MORTRED_WORKER_NUMS") != str(workers):
            return "fail", "MORTRED_WORKER_NUMS=%s" % child_env.get("MORTRED_WORKER_NUMS")
        if child_env.get("MORTRED_DECODE_AUTO") != decode_auto:
            return "fail", "MORTRED_DECODE_AUTO=%s" % child_env.get("MORTRED_DECODE_AUTO")
        auth = child_env.get("MORTRED_AUTH_TOKEN") or ""
        if not auth:
            return "fail", "model process has no auth token"
        before = http_get(spec["port"], "/metrics", auth)
        if before is None or before[0] != 200:
            return "fail", "pre-load metrics scrape failed"
        before_text = before[1].decode("utf-8", "replace")
        sys.stderr.write("== %s load ==\n" % model_id)
        sys.stderr.flush()
        try:
            load_code = run_logged(load_cmd, base_env(tokens), log_path, 240)
        except subprocess.TimeoutExpired:
            return "fail", "official load timed out"
        if not out_path.is_file():
            return "no-rps", "load report missing (exit %s)\n%s" % (load_code, log_tail(log_path))
        try:
            report = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return "no-rps", "load report unreadable: %s" % exc
        ok_count = int(report.get("ok") or 0)
        rps = report.get("rps")
        if ok_count <= 0 or not isinstance(rps, (int, float)) or float(rps) <= 0:
            return "no-rps", "no successful RPS (exit %s, ok=%s)" % (load_code, ok_count)
        if not pid_alive(pid):
            return "fail", "model process exited before the post-load scrape"
        doc = fetch_status(tokens["MORTRED_API_TOKEN"])
        current = one_server(doc, model_id) if isinstance(doc, dict) else None
        if current is None or current.get("pid") != pid:
            return "fail", "model pid changed before the post-load scrape"
        after = http_get(spec["port"], "/metrics", auth)
        if after is None or after[0] != 200:
            return "fail", "post-load metrics scrape failed"
        after_text = after[1].decode("utf-8", "replace")
        gateway = http_get(8080, "/metrics", tokens["MORTRED_METRICS_TOKEN"])
        if gateway is None or gateway[0] != 200:
            return "fail", "gateway metrics scrape failed"
        gateway_text = gateway[1].decode("utf-8", "replace")
        available = _gauge(after_text, "mortred_workers_available", spec["server_uri"])
        busy = _gauge(after_text, "mortred_workers_busy", spec["server_uri"])
        if available is None or busy is None:
            return "fail", "post-load worker gauges missing for %s" % spec["server_uri"]
        queue_delta = hist_count_delta(
            before_text, after_text, "mortred_queue_wait_duration_ms", spec["server_uri"]
        )
        infer_delta = hist_count_delta(
            before_text, after_text, "mortred_inference_duration_ms", spec["server_uri"]
        )
        counts = jpeg_delta(before_text, after_text)
        p50s = {}
        for stage in STAGES:
            series = stage_series(gateway_text, stage)
            p50 = histogram_p50(series)
            if p50 is None:
                return "fail", "gateway stage %s has no p50" % stage
            p50s[stage] = p50
        stage_sum = sum(p50s.values())
        server_p50 = report.get("server_time_ms_p50")
        if not isinstance(server_p50, (int, float)) or isinstance(server_p50, bool):
            return "fail", "server_time_ms p50 missing"
        gauge = dominant_gauge(gateway_text)
        dominant = dominant_name(p50s) if stage_sum > 0.5 * float(server_p50) else None
        requests = int(report.get("requests") or 0)
        errors = int(report.get("errors") or 0)
        http_429 = int((report.get("by_http") or {}).get("429") or 0)
        throughput = read_throughput(model_id)
        latency = report.get("latency_ms") or {}
        row = {
            "id": model_id,
            "blank": False,
            "worktree": str(ROOT),
            "hardware": hardware,
            "date": today,
            "server_uri": spec["server_uri"],
            "conf_server": spec["conf_server"],
            "conf_server_worker_nums": 1,
            "worker_nums": workers,
            "decode_auto": decode_auto,
            "prepare": "mortredctl prepare --id %s" % model_id,
            "calibrate": calibrate,
            "calibrate_raw": "worker sweep and jpeg decode vote use --raw",
            "mortred_worker_nums": workers,
            "mortred_decode_auto": decode_auto,
            "process_workers": _integral(available + busy),
            "process_workers_source": "after_load",
            "process_workers_note": (
                "官方压测产生完成响应且进程仍在时，"
                "mortred_workers_available{model=\"%s\"} 与 "
                "mortred_workers_busy{model=\"%s\"} 之和；不是压测前刮取"
                % (spec["server_uri"], spec["server_uri"])
            ),
            "workers_available_after": available,
            "workers_busy_after": busy,
            "load_command": " ".join(load_cmd),
            "load_url": str(report.get("url") or ""),
            "wall_s": report.get("wall_s"),
            "warmup_s": report.get("warmup_s"),
            "success_rps": rps,
            "p50_ms": latency.get("p50"),
            "p99_ms": latency.get("p99"),
            "http_429": http_429,
            "error_rate": (errors - http_429) / requests if requests else None,
            "requests": requests,
            "errors": errors,
            "throughput_qps": throughput,
            "ratio": float(rps) / throughput,
            "queue_count_delta": queue_delta,
            "queue_model": spec["server_uri"],
            "infer_count_delta": infer_delta,
            "infer_model": spec["server_uri"],
            "jpeg_jpeggpu": counts["jpeggpu"],
            "jpeg_cpu_reduced": counts["cpu-reduced"],
            "jpeg_cpu_full": counts["cpu-full"],
            "jpeg_landing": jpeg_landing(decode_auto, counts),
            "stage_p50_ms": p50s,
            "stage_p50_sum": stage_sum,
            "server_time_ms_p50": float(server_p50),
            "dominant_stage": dominant,
            "dominant_gauge": gauge if gauge else "absent",
            "image": image_rel or "-",
            "decoded_width": 256 if model_id == "realesrgan" else None,
            "decoded_height": 256 if model_id == "realesrgan" else None,
            "engine": "static shape batch=1",
        }
        passed, why = judge_row(row, {model_id: spec})
        if not passed:
            return "fail", why
        rendered = render_row(row)
        for value in tokens.values():
            if value and value in rendered:
                return "fail", "refusing to write a token"
        if auth in rendered:
            return "fail", "refusing to write a token"
        try:
            record_pf5_baseline(row, baseline_source)
        except RuntimeError as exc:
            return "fail", str(exc)
        return "row", rendered
    finally:
        stop_supervisor(proc)
        try:
            log_fp.close()
        except OSError:
            pass
        if not wait_ports_clear(ports):
            sys.stderr.write("%s: ports still busy after stop: %s\n" % (model_id, ports))


def require_sof0(rel: str, width: int, height: int, nbytes: int, label: str) -> str | None:
    path = ROOT / rel
    try:
        data = path.read_bytes()
    except OSError:
        return "%s missing\n" % label
    if jpeg_sof0_size(data) != (width, height) or len(data) != nbytes:
        return "%s is not the SOF0 %dx%d %d-byte image\n" % (label, width, height, nbytes)
    return None


def report_section_ids(report: Path = REPORT) -> list[str]:
    if not report.is_file():
        return []
    return re.findall(r"(?m)^## ([A-Za-z0-9_]+)\s*$", report.read_text(encoding="utf-8"))


def upsert_section(text: str, model_id: str, body: str) -> str:
    """Replace the ## model_id section, or append it."""
    section = body if body.endswith("\n") else body + "\n"
    if not section.endswith("\n\n"):
        section = section.rstrip("\n") + "\n\n"
    matches = list(re.finditer(r"(?m)^## ([A-Za-z0-9_]+)\s*$", text))
    for index, match in enumerate(matches):
        if match.group(1) != model_id:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[: match.start()] + section + text[end:]
    if text and not text.endswith("\n"):
        text += "\n"
    return text + section


def live_context(specs: list[dict]):
    mismatch = contract_ok(specs)
    if mismatch is None:
        missing_caps = missing_p99_caps([spec["id"] for spec in specs])
        if missing_caps:
            sys.stderr.write(format_missing_p99_table(missing_caps))
            return _fail("missing p99 caps\n"), None
    if mismatch:
        return _fail(mismatch + "\n"), None
    ctl = ROOT / "_bin" / "mortredctl.out"
    supervisor = ROOT / "_bin" / "mortred-supervisor.out"
    if not ctl.is_file() or not supervisor.is_file():
        return _fail("mortredctl or supervisor binary missing\n"), None
    tokens = load_tokens()
    if tokens is None:
        return _fail("missing management, inference, or metrics token\n"), None
    ids = {spec["id"] for spec in specs}
    if "pphuman_mobile" in ids:
        err = require_sof0(FACE_IMAGE, 380, 381, 45770, "face_w_mask.jpg")
        if err:
            return _fail(err), None
    if "dbnet" in ids:
        err = require_sof0(DBNET_IMAGE, 746, 530, 109127, "coco_text_test.jpg")
        if err:
            return _fail(err), None
    if "superpoint" in ids:
        err = require_sof0(SUPERPOINT_IMAGE, 640, 480, 55892, "test.jpg")
        if err:
            return _fail(err), None
    if "metric3d_512" in ids:
        err = require_sof0(METRIC3D_IMAGE, 1242, 375, 147444, "0000000005.jpg")
        if err:
            return _fail(err), None
    try:
        if "realesrgan" in ids:
            ensure_realesrgan_jpeg(ROOT / REALESRGAN_IMAGE)
        hardware = hardware_line()
        for spec in specs:
            read_throughput(spec["id"])
    except (OSError, RuntimeError) as exc:
        return _fail("%s\n" % exc), None
    return 0, {
        "ctl": ctl,
        "tokens": tokens,
        "hardware": hardware,
        "today": date.today().isoformat(),
    }


def append_runs(
    specs: list[dict],
    tokens: dict[str, str],
    hardware: str,
    today: str,
    ctl: Path,
    report: Path = REPORT,
    baseline_source: str = "docs/perf/coverage_set_http_rps.md",
) -> int:
    for spec in specs:
        kind, payload = run_id(spec, tokens, hardware, today, ctl, baseline_source)
        if kind == "row":
            append_text(payload if isinstance(payload, str) else "", report)
            sys.stderr.write("== %s wrote ==\n" % spec["id"])
            sys.stderr.flush()
            continue
        if kind == "no-rps" and spec["id"] == "libface" and report == REPORT:
            append_text(render_row({"id": "libface", "blank": True}), report)
            sys.stderr.write("== libface blank: %s ==\n" % payload)
            sys.stderr.flush()
            continue
        sys.stderr.write("%s failed: %s\n" % (spec["id"], payload))
        return 1
    print("coverage_http_report.py wrote %s" % report)
    return 0


def run_live(catalog: str = "coverage_set") -> int:
    if catalog != "coverage_set":
        return _fail(
            "product_static live runs need --id or --from so the skeleton is not wiped\n"
        )
    specs = list(specs_for(catalog))
    code, ctx = live_context(specs)
    if ctx is None:
        return code
    report = report_path_for(catalog)
    reset_report(ctx["hardware"], ctx["today"], report)
    return append_runs(
        specs,
        ctx["tokens"],
        ctx["hardware"],
        ctx["today"],
        ctx["ctl"],
        report,
        "docs/perf/coverage_set_http_rps.md",
    )


def _source_for(catalog: str) -> str:
    if catalog == "product_static":
        return "docs/perf/product_static_http_rps.md"
    return "docs/perf/coverage_set_http_rps.md"


def run_from(start_id: str, catalog: str = "coverage_set") -> int:
    specs_all = list(specs_for(catalog))
    ids = [spec["id"] for spec in specs_all]
    if start_id not in ids:
        return _fail("unknown id %s\n" % start_id)
    index = ids.index(start_id)
    report = report_path_for(catalog)
    found = report_section_ids(report)
    expected = ids[:index]
    if catalog == "coverage_set":
        if found != expected:
            return _fail(
                "checklist has %s; --from %s keeps existing rows and needs exactly %s\n"
                % (found, start_id, expected)
            )
    else:
        if found[:index] != expected or found[index:] != ids[index:]:
            return _fail(
                "product checklist sections are %s; --from %s needs the catalog order\n"
                % (found, start_id)
            )
    if not report.is_file():
        return _fail("checklist missing; --from does not create or clear it\n")
    original = report.read_bytes()
    specs = specs_all[index:]
    code, ctx = live_context(specs)
    if ctx is None:
        return code
    if report.read_bytes() != original:
        return _fail("refusing to alter rows already written\n")
    if catalog == "coverage_set":
        return append_runs(
            specs,
            ctx["tokens"],
            ctx["hardware"],
            ctx["today"],
            ctx["ctl"],
            report,
            _source_for(catalog),
        )
    for spec in specs:
        kind, payload = run_id(
            spec, ctx["tokens"], ctx["hardware"], ctx["today"], ctx["ctl"], _source_for(catalog)
        )
        if kind != "row" or not isinstance(payload, str):
            sys.stderr.write("%s failed: %s\n" % (spec["id"], payload))
            return 1
        text = upsert_section(report.read_text(encoding="utf-8"), spec["id"], payload)
        report.write_text(text, encoding="utf-8")
        sys.stderr.write("== %s wrote ==\n" % spec["id"])
    print("coverage_http_report.py wrote %s" % report)
    return 0


def run_one(model_id: str, catalog: str) -> int:
    specs_all = list(specs_for(catalog))
    ids = [spec["id"] for spec in specs_all]
    if model_id not in ids:
        return _fail("unknown id %s\n" % model_id)
    report = report_path_for(catalog)
    found = report_section_ids(report)
    if model_id not in found:
        return _fail(
            "checklist has no ## %s; --id replaces one section and does not create the file\n"
            % model_id
        )
    spec = specs_all[ids.index(model_id)]
    original = report.read_bytes()
    code, ctx = live_context([spec])
    if ctx is None:
        return code
    if report.read_bytes() != original:
        return _fail("refusing to alter rows already written\n")
    kind, payload = run_id(
        spec, ctx["tokens"], ctx["hardware"], ctx["today"], ctx["ctl"], _source_for(catalog)
    )
    if report.read_bytes() != original:
        return _fail("refusing to alter rows already written\n")
    if kind != "row" or not isinstance(payload, str):
        sys.stderr.write("%s failed: %s\n" % (model_id, payload))
        return 1
    report.write_text(upsert_section(report.read_text(encoding="utf-8"), model_id, payload), encoding="utf-8")
    print("coverage_http_report.py wrote %s" % report)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="coverage_http_report.py")
    parser.add_argument(
        "--catalog",
        choices=CATALOG_NAMES,
        default="coverage_set",
        help="coverage_set (default) or product_static (conf/product_static_bs1.json)",
    )
    parser.add_argument("--check", action="store_true", help="require qualified PASS rows")
    parser.add_argument(
        "--check-skeleton",
        action="store_true",
        help="require catalog headings; blocked/missing sections are allowed",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--from", dest="from_id", help="live run starting at this id")
    parser.add_argument("--id", dest="only_id", help="live run for one id")
    args = parser.parse_args(argv)
    modes = [args.check, args.check_skeleton, args.self_test, bool(args.from_id), bool(args.only_id)]
    if sum(bool(item) for item in modes) > 1:
        return _fail(
            "usage: python3 scripts/coverage_http_report.py "
            "[--catalog coverage_set|product_static] "
            "[--check | --check-skeleton | --self-test | --from <id> | --id <id>]\n"
        )
    if args.self_test:
        return run_self_test()
    if args.check:
        return check_catalog(args.catalog, skeleton=False)
    if args.check_skeleton:
        return check_catalog(args.catalog, skeleton=True)
    if args.only_id:
        return run_one(args.only_id, args.catalog)
    if args.from_id:
        return run_from(args.from_id, args.catalog)
    return run_live(args.catalog)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
