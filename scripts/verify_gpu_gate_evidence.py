#!/usr/bin/env python3
"""Verify a GPU/gate evidence manifest: schema fields, version alignment, artifact digests."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--allow-missing-artifacts", action="store_true",
                    help="For example manifests / forks without downloaded artifacts")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if man.get("schema_version") != "1":
        print("FAIL schema_version", file=sys.stderr); return 1
    if not man.get("product_version"):
        print("FAIL product_version", file=sys.stderr); return 1
    if not man.get("reproduce"):
        print("FAIL reproduce commands required", file=sys.stderr); return 1
    # align with single version source when available
    try:
        from check_version_single_source import cmake_version
        cv = cmake_version(root)
        if man["product_version"] != cv:
            print(f"FAIL manifest product_version {man['product_version']} != cmake {cv}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"WARN could not import version check: {e}")
    for art in man.get("artifacts") or []:
        digest = art.get("sha256", "")
        if len(digest) != 64:
            print(f"FAIL bad sha256 for {art.get('name')}", file=sys.stderr); return 1
        path = root / art["path"]
        if not path.is_file():
            if args.allow_missing_artifacts:
                print(f"SKIP missing artifact {path}")
                continue
            print(f"FAIL missing artifact {path}", file=sys.stderr); return 1
        got = sha256_file(path)
        if got != digest:
            print(f"FAIL digest mismatch {art['name']}: {got} != {digest}", file=sys.stderr); return 1
        print(f"OK {art['name']} sha256 matched")
    print("OK evidence manifest verified")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
