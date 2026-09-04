#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mortred model server demo client.

Reads the authoritative server configuration directly from conf/server/*.toml
(host / port / server_uri), so the URLs this client targets can never drift
from the configs the servers actually start with. A per-model demo image map
(defaults under demo_data/model_test_input) provides the request payload; a
custom image can be supplied with --image.

Stdlib only (no locust, no requests). Concurrent HTTP RPS lives in http_infer_rps.py.

Usage (run from anywhere; the repository root is located relative to this file):

  python3 scripts/server/test_server.py --list

  python3 scripts/server/test_server.py --server mobilenetv2 --mode single [--times 3]
  python3 scripts/server/test_server.py --server mobilenetv2 --mode single --dry-run

  python3 scripts/server/test_server.py --server yolov5 --mode load \\
      --concurrency 8 --duration 30s

  python3 scripts/server/test_server.py --server MOBILENETV2 --mode load \\
      --gateway --token "$MORTRED_GATEWAY_AUTH_TOKEN" --concurrency 8 --duration 30s
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

# repository root: <root>/scripts/server/test_server.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

from repo_toml import load_toml  # noqa: E402
from http_infer_rps import LoadConfig, parse_duration, print_report, run_load  # noqa: E402

# per-model demo image (relative to demo_data/model_test_input), keyed by the
# normalized model directory name under conf/server/<category>/<model>/
DEMO_IMAGE_BY_MODEL = {
    "attentiveganderain": "enhancement/derain/test_1.png",
    "enlightengan": "enhancement/low_light/lol_test_1.png",
    "realesrgan": "enhancement/real_esr/test.JPG",
    "superpoint": "feature_point/test.png",
    "modnet": "matting/matting_test.jpg",
    "ppmatting": "matting/matting_test.jpg",
    "depthanything": "mono_depth_estimation/0000000005.png",
    "metric3d": "mono_depth_estimation/0000000005.png",
    "dbnet": "ocr/railway_ticket.png",
    "centerfacedet": "object_detection/face_wo_mask.jpg",
    "libfacedet": "object_detection/face_wo_mask.jpg",
    "nanodet": "object_detection/bus.jpg",
    "yolov5": "object_detection/bus.jpg",
    "yolov6": "object_detection/bus.jpg",
    "yolov7": "object_detection/bus.jpg",
    "yolov8": "object_detection/bus.jpg",
    "bisenetv2": "scene_segmentation/cityscapes_test.png",
    "hrnet": "scene_segmentation/cityscapes_test.png",
    "pphumanseg": "scene_segmentation/human_image.jpg",
    "densenet": "classification/ILSVRC2012_val_00000003.JPEG",
    "mobilenetv2": "classification/ILSVRC2012_val_00000003.JPEG",
    "resnet": "classification/ILSVRC2012_val_00000003.JPEG",
}

# category-level fallback demo image
DEMO_IMAGE_BY_CATEGORY = {
    "classification": "classification/ILSVRC2012_val_00000003.JPEG",
    "object_detection": "object_detection/bus.jpg",
    "scene_segmentation": "scene_segmentation/cityscapes_test.png",
    "enhancement": "enhancement/derain/test_1.png",
    "feature_point": "feature_point/test.png",
    "matting": "matting/matting_test.jpg",
    "mono_depth_estimation": "mono_depth_estimation/0000000005.png",
    "ocr": "ocr/railway_ticket.png",
}


def normalize(name: str) -> str:
    """lowercase + keep alphanumerics only, e.g. 'ATTENTIVE_GAN_DERAIN_SERVER' -> 'attentiveganderainserver'"""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def build_catalog(root: Path) -> list[dict]:
    """Discover model servers from conf/server/*.toml (single source of truth)."""
    entries: list[dict] = []
    seen: set[tuple] = set()
    conf_root = root / "conf" / "server"
    if not conf_root.exists():
        return entries
    for cfg in sorted(conf_root.rglob("*.toml")):
        try:
            table = load_toml(cfg)
        except (ValueError, OSError):
            continue
        rel = cfg.relative_to(conf_root)
        parts = rel.parts
        category = parts[0] if len(parts) > 1 else "other"
        model_dir = parts[-2] if len(parts) > 1 else ""
        for section, kv in table.items():
            if not isinstance(kv, dict):
                continue
            uri = kv.get("server_uri")
            if not isinstance(uri, str) or not uri:
                continue
            try:
                port = int(kv.get("port", 0))
            except (TypeError, ValueError):
                port = 0
            host = kv.get("host") or "127.0.0.1"
            model_id = kv.get("model")
            if not isinstance(model_id, str) or not model_id:
                model_id = section[:-7] if section.endswith("_SERVER") else section
            key = (section, host, port, uri)
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "section": section,
                "norm_section": normalize(section),
                "category": category,
                "model_dir": model_dir,
                "norm_model": normalize(model_dir),
                "model_id": model_id,
                "host": host,
                "port": port,
                "uri": uri,
                "url": "http://%s:%d%s" % (host, port, uri),
                "worker_nums": kv.get("worker_nums", ""),
            })
    return entries


def resolve_server(arg: str, entries: list[dict]) -> tuple[dict | None, str | None]:
    """Match --server against discovered servers; exact section name wins first."""
    norm = normalize(arg)
    if not norm:
        return None, "empty server name"
    exact = [e for e in entries if e["norm_section"] == norm]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        names = ", ".join(sorted(e["section"] for e in exact))
        return None, "ambiguous server name %r, candidates: %s" % (arg, names)
    by_id = [e for e in entries if normalize(e["model_id"]) == norm]
    if len(by_id) == 1:
        return by_id[0], None
    sub = [e for e in entries if norm in e["norm_section"]]
    if len(sub) == 1:
        return sub[0], None
    if len(sub) > 1:
        names = ", ".join(sorted(e["section"] for e in sub))
        return None, "ambiguous server name %r, candidates: %s" % (arg, names)
    return None, "no server matches %r; run with --list to see available servers" % arg


def resolve_demo_image(entry: dict, root: Path) -> Path | None:
    rel = DEMO_IMAGE_BY_MODEL.get(entry["norm_model"]) or DEMO_IMAGE_BY_CATEGORY.get(entry["category"])
    if not rel:
        return None
    path = root / "demo_data" / "model_test_input" / rel
    return path if path.exists() else None


def build_payload(image_path: Path) -> dict:
    image_data = image_path.read_bytes()
    task_id = hashlib.md5((str(image_path) + str(time.time())).encode()).hexdigest()
    return {
        "images": [base64.b64encode(image_data).decode()],
        "req_id": task_id,
    }


def apply_gateway(entry: dict, gateway: str) -> dict:
    """Route through mortred-gateway catalog path, not the model's loopback port."""
    out = dict(entry)
    hostport = gateway.strip()
    if "://" in hostport:
        parsed = urllib.parse.urlparse(hostport)
        hostport = parsed.netloc or hostport
    out["url"] = "http://%s/v1/models/%s/infer" % (hostport, entry["model_id"])
    return out


def _post_once(url: str, payload: dict, token: str, timeout_s: float) -> tuple[int, str]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    if host in ("localhost", "::1"):
        host = "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = path + "?" + parsed.query
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(len(body)),
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    if parsed.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port, timeout=timeout_s)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", errors="replace")
        return resp.status, text
    finally:
        conn.close()


def run_single(entry: dict, image_path: Path, times: int, dry_run: bool,
               token: str = "") -> int:
    url = entry["url"]
    print("server      : %s" % entry["section"])
    print("url         : %s" % url)
    print("input image : %s" % image_path)
    print("loop times  : %d" % times)
    if dry_run:
        print("[dry-run] no request sent")
        return 0
    payload = build_payload(image_path)
    for i in range(times):
        try:
            status, text = _post_once(url, payload, token, 30.0)
            print("[%d/%d] http=%d %s" % (i + 1, times, status, text[:200]))
            if status >= 400:
                return 1
        except OSError as exc:
            print("request failed: %s" % exc)
            return 1
    return 0


def run_load_mode(entry: dict, image_path: Path, args: argparse.Namespace) -> int:
    url = entry["url"]
    print("server      : %s" % entry["section"])
    print("url         : %s" % url)
    print("input image : %s" % image_path)
    if args.dry_run:
        print("[dry-run] concurrency=%d duration=%s requests=%d" % (
            args.concurrency, args.duration or "-", args.requests))
        return 0
    duration_s = parse_duration(args.duration) if args.duration else 0.0
    warmup_s = parse_duration(args.warmup) if args.warmup else 0.0
    if duration_s <= 0 and args.requests <= 0:
        print("load mode needs --duration and/or --requests", file=sys.stderr)
        return 1
    token = args.token or os.environ.get("MORTRED_GATEWAY_AUTH_TOKEN", "")
    cfg = LoadConfig(
        url=url,
        image_path=image_path,
        concurrency=args.concurrency,
        duration_s=duration_s,
        requests=args.requests,
        warmup_s=warmup_s,
        qps=args.qps,
        timeout_s=args.timeout,
        token=token,
        follow_retry_after=args.follow_retry_after,
        progress=not args.quiet,
    )
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mortred model server demo client (config-driven from conf/server).")
    parser.add_argument("--list", action="store_true", help="list all discoverable model servers")
    parser.add_argument("--server", type=str, default="", help="model server name, e.g. mobilenetv2 / yolov5")
    parser.add_argument("--mode", choices=["single", "load"], default="single",
                        help="single: print sequential responses; load: closed-loop concurrent client")
    parser.add_argument("--times", type=int, default=3, help="loop times for single mode (default 3)")
    parser.add_argument("--image", type=str, default="", help="override the demo input image path")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without sending any request")
    parser.add_argument("-c", "--concurrency", type=int, default=8,
                        help="load-mode worker threads (keep-alive connections)")
    parser.add_argument("-d", "--duration", default="",
                        help="load-mode measure window, e.g. 30s / 2m")
    parser.add_argument("-n", "--requests", type=int, default=0,
                        help="load-mode stop after this many measured requests")
    parser.add_argument("--warmup", default="0s", help="load-mode discard samples for this long")
    parser.add_argument("--qps", type=float, default=0.0, help="optional shared rate cap (0 = closed loop)")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request socket timeout seconds")
    parser.add_argument("--follow-retry-after", action="store_true",
                        help="sleep Retry-After on HTTP 429")
    parser.add_argument("--out", default="", help="write load JSON report to this path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gateway", nargs="?", const="127.0.0.1:8080", default="",
                        help="route via mortred-gateway POST /v1/models/{id}/infer "
                             "(optional HOST:PORT, default 127.0.0.1:8080)")
    parser.add_argument("--token", default="",
                        help="Authorization Bearer (default MORTRED_GATEWAY_AUTH_TOKEN)")
    args = parser.parse_args()

    entries = build_catalog(ROOT)
    if args.list:
        if not entries:
            print("no server configs found under conf/server")
            return 1
        print("discovered %d model servers:" % len(entries))
        for e in sorted(entries, key=lambda x: x["section"]):
            workers = "worker_nums=%s" % e["worker_nums"] if e["worker_nums"] else ""
            print("  %-32s %s  model=%s %s" % (e["section"], e["url"], e["model_id"], workers))
        return 0

    if not args.server:
        parser.error("--server is required unless --list is used")
    if not entries:
        print("no server configs found under conf/server; build/configure the project first")
        return 1

    entry, err = resolve_server(args.server, entries)
    if err:
        print(err)
        return 1
    if args.gateway:
        entry = apply_gateway(entry, args.gateway)
        print("via gateway : %s" % args.gateway)

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
    else:
        image_path = resolve_demo_image(entry, ROOT)
        if image_path is None:
            print("no demo image available for %s; pass --image PATH" % entry["section"])
            return 1
    if not image_path.exists():
        print("input image not exist: %s" % image_path)
        return 1

    if args.mode == "single":
        token = args.token or os.environ.get("MORTRED_GATEWAY_AUTH_TOKEN", "")
        return run_single(entry, image_path, args.times, args.dry_run, token=token)
    return run_load_mode(entry, image_path, args)


if __name__ == "__main__":
    sys.exit(main())
