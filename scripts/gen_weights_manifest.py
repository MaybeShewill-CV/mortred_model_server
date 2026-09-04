#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_weights_manifest.py - generate/update conf/weights_manifest.json (weight
version-lock manifest).

Walks the local weights/ directory and computes sha256 and size for every file;
by default queries the HF API (MaybeShewill-CV/mortred_model_server) for the
real repo layout, tagging each file with `hf_path` (path inside HF) and `on_hf`
(whether it exists in the HF repo). Files that already exist with the same size
reuse their old sha256 (no rehash, fast regeneration).

Usage (run from the repo root):
  python3 scripts/gen_weights_manifest.py                  # query HF API + reuse hashes
  python3 scripts/gen_weights_manifest.py --no-hf-api      # offline: no query, on_hf = true
  python3 scripts/gen_weights_manifest.py --force-hash     # force rehash of everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
OUT = ROOT / "conf" / "weights_manifest.json"
HF_API = "https://huggingface.co/api/models/MaybeShewill-CV/mortred_model_server"

# Curated cpu-profile weight set (the cpu deployment profile serves exactly
# these models; matches the *_cpu_config.toml variants under conf/). Extend
# this list when adding a model to the cpu catalog - it drives both the
# manifest "profiles" tag and fetch_weights.py --profile cpu.
CPU_WEIGHTS = {
    "weights/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn",
    "weights/classification/resnet/resnet-50.mnn",
    "weights/object_detection/yolov8/yolov8s.onnx",
    "weights/scene_segmentation/hrnet/hrnetw48_ccd.onnx",
    # Hosted cpu-profile golden set (conf/ci_hosted_golden.json). Keep these
    # tagged cpu so fetch --profile cpu and CI stay aligned after regenerate.
    "weights/object_detection/nanodet/nanodet_plus_m_1x5.mnn",
    "weights/ocr/db_text_detector/db_model_large.mnn",
    "weights/feature_point/superpoint/superpoint_120x160.mnn",
    "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.mnn",
}


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch_hf_files() -> set[str]:
    """Set of file paths in the HF repo (before stripping the weights/ prefix,
    i.e. repo-internal paths)."""
    import requests

    r = requests.get(HF_API, timeout=30)
    r.raise_for_status()
    return {s["rfilename"] for s in r.json().get("siblings", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default="MaybeShewill-CV/mortred_model_server")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--no-hf-api", action="store_true",
                        help="offline mode: skip HF API queries, set on_hf to true")
    parser.add_argument("--force-hash", action="store_true", help="force rehash of all sha256")
    args = parser.parse_args()

    if not WEIGHTS_DIR.exists():
        sys.exit(f"[ERROR] {WEIGHTS_DIR} not found (nothing to manifest)")

    # read the old manifest to reuse sha256 values
    old_by_path: dict[str, dict] = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
            old_by_path = {f["path"]: f for f in old.get("files", [])}
        except (json.JSONDecodeError, OSError):
            pass

    hf_files: set[str] | None = None
    if not args.no_hf_api:
        try:
            hf_files = fetch_hf_files()
            print(f"[info] HF repo file count: {len(hf_files)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] HF API query failed, on_hf set to true: {exc}")

    files = []
    for p in sorted(WEIGHTS_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()  # e.g. weights/xxx/yyy.onnx
        size = p.stat().st_size
        prev = old_by_path.get(rel, {})
        if not args.force_hash and prev.get("size") == size and prev.get("sha256"):
            sha = prev["sha256"]
        else:
            sha = sha256_of(p)
        hf_path = rel[len("weights/"):] if rel.startswith("weights/") else rel
        entry = {
            "path": rel,
            "size": size,
            "sha256": sha,
            "hf_path": hf_path,
            "on_hf": (hf_files is None) or (hf_path in hf_files),
            # deployment profiles this weight belongs to; curated cpu set is
            # both, everything else is gpu-only (drives fetch --profile)
            "profiles": ["cpu", "gpu"] if rel in CPU_WEIGHTS else ["gpu"],
        }
        files.append(entry)
        print(f"  {rel}  {size}  {sha[:12]}…  hf={'Y' if entry['on_hf'] else 'N'}")

    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "description": (
            "weights baseline manifest (sha256 locked); hf_path/on_hf mark HF "
            "repo reachability. Generated by scripts/gen_weights_manifest.py"
        ),
        "files": files,
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_hf = sum(1 for f in files if f["on_hf"])
    print(f"\n== wrote {OUT.relative_to(ROOT)}: {len(files)} files "
          f"(on_hf={n_hf}, off_hf={len(files) - n_hf})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
