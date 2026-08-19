#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mortred model server demo client.

Reads the authoritative server configuration directly from conf/server/*.toml
(host / port / server_uri), so the URLs this client targets can never drift
from the configs the servers actually start with. A per-model demo image map
(defaults under demo_data/model_test_input) provides the request payload; a
custom image can be supplied with --image.

Usage (run from anywhere; the repository root is located relative to this file):

  # list all discoverable model servers
  python server/test_server.py --list

  # single-mode smoke test: post a demo image N times
  python server/test_server.py --server mobilenetv2 --mode single [--times 1000]
  python server/test_server.py --server mobilenetv2 --mode single --dry-run

  # locust load test (requires: pip install locust requests)
  python server/test_server.py --server yolov5 --mode locust \
      [--users 20] [--spawn-rate 10] [--time 10m]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# repository root: <root>/scripts/server/test_server.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
# make scripts/repo_toml.py importable regardless of cwd / PYTHONPATH
sys.path.insert(0, str(ROOT / "scripts"))

from repo_toml import load_toml  # noqa: E402

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
            host = kv.get("host") or "localhost"
            entries.append({
                "section": section,
                "norm_section": normalize(section),
                "category": category,
                "model_dir": model_dir,
                "norm_model": normalize(model_dir),
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
    with open(image_path, "rb") as f:
        image_data = f.read()
    task_id = str(image_path) + str(time.time())
    task_id = hashlib.md5(task_id.encode()).hexdigest()
    return {
        "img_data": base64.b64encode(image_data).decode(),
        "req_id": task_id,
    }


def run_single(entry: dict, image_path: Path, times: int, dry_run: bool) -> int:
    url = entry["url"]
    print("server      : %s" % entry["section"])
    print("url         : %s" % url)
    print("input image : %s" % image_path)
    print("loop times  : %d" % times)
    if dry_run:
        print("[dry-run] no request sent")
        return 0
    try:
        import requests
    except ImportError:
        print("requests is required for single mode: pip install requests")
        return 1
    payload = build_payload(image_path)
    for i in range(times):
        try:
            resp = requests.post(url=url, data=json.dumps(payload), timeout=30)
            print("[%d/%d] http=%d %s" % (i + 1, times, resp.status_code, resp.text[:200]))
        except Exception as exc:  # noqa: BLE001 - demo client, surface the error
            print("request failed: %s" % exc)
            return 1
    return 0


def run_locust(entry: dict, image_path: Path, users: int, spawn_rate: int, duration: str, dry_run: bool) -> int:
    url = entry["url"]
    print("server      : %s" % entry["section"])
    print("url         : %s" % url)
    print("input image : %s" % image_path)
    print("locust      : -u %d -r %d -t %s" % (users, spawn_rate, duration))
    if dry_run:
        print("[dry-run] no load test started")
        return 0
    try:
        import locust  # noqa: F401
    except ImportError:
        print("locust is required for locust mode: pip install locust")
        return 1
    script = ROOT / "scripts" / "server" / "locust_performance.py"
    env = dict(os.environ)
    env["LOCUST_URL"] = url
    env["LOCUST_IMG"] = str(image_path)
    cmd = [
        sys.executable, "-m", "locust", "-f", str(script),
        "--host=" + url, "--headless",
        "-u", str(users), "-r", str(spawn_rate), "-t", duration,
    ]
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mortred model server demo client (config-driven from conf/server).")
    parser.add_argument("--list", action="store_true", help="list all discoverable model servers")
    parser.add_argument("--server", type=str, default="", help="model server name, e.g. mobilenetv2 / yolov5")
    parser.add_argument("--mode", choices=["single", "locust"], default="single",
                        help="single: loop a demo request; locust: headless load test")
    parser.add_argument("--times", type=int, default=1000, help="loop times for single mode (default 1000)")
    parser.add_argument("--image", type=str, default="", help="override the demo input image path")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without sending any request")
    parser.add_argument("-u", "--users", type=int, default=20, help="locust number of users")
    parser.add_argument("-r", "--spawn-rate", type=int, default=10, help="locust spawn rate")
    parser.add_argument("-t", "--time", type=str, default="10m", help="locust test duration")
    args = parser.parse_args()

    entries = build_catalog(ROOT)
    if args.list:
        if not entries:
            print("no server configs found under conf/server")
            return 1
        print("discovered %d model servers:" % len(entries))
        for e in sorted(entries, key=lambda x: x["section"]):
            workers = "worker_nums=%s" % e["worker_nums"] if e["worker_nums"] else ""
            print("  %-32s %s %s" % (e["section"], e["url"], workers))
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
        return run_single(entry, image_path, args.times, args.dry_run)
    return run_locust(entry, image_path, args.users, args.spawn_rate, args.time, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
