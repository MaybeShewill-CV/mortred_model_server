#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_weights.py - automatically download/verify model weights from Hugging Face.

Replaces the manual Baidu-netdisk download flow: the manifest
conf/weights_manifest.json declares the repo, revision and every file
(path/size/sha256); this script downloads per the manifest into weights/,
skipping files that already exist with a matching sha256 (resumable).

Usage (run from the repo root):
  python3 scripts/fetch_weights.py                    # download all missing weights
  python3 scripts/fetch_weights.py --only yolov8      # only files whose path contains yolov8
  python3 scripts/fetch_weights.py --check            # only verify existing files, no download
  python3 scripts/fetch_weights.py --dry-run          # print what would be downloaded
  python3 scripts/fetch_weights.py --manifest FILE    # custom manifest (default conf/weights_manifest.json)

Deps: requests (or huggingface_hub, either is enough); to regenerate the
manifest use python3 scripts/gen_weights_manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "conf" / "weights_manifest.json"
WEIGHTS_DIR = ROOT / "weights"


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[ERROR] manifest not found: {path} (run python3 scripts/gen_weights_manifest.py)")
    # utf-8-sig: tolerate a BOM that Windows editors may write
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_url(repo: str, revision: str, hf_path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{hf_path}"


def hf_path_of(item: dict) -> str:
    """Path inside the HF repo: explicit manifest hf_path wins; otherwise strip
    the weights/ prefix (HF layout = local layout minus the weights/ prefix)."""
    rel = item["path"]
    if item.get("hf_path"):
        return item["hf_path"]
    return rel[len("weights/"):] if rel.startswith("weights/") else rel


def download_file(url: str, dst: Path) -> None:
    # prefer huggingface_hub (resumable/multithreaded), fall back to requests
    try:
        import huggingface_hub  # noqa: F401
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=url.split("/resolve/")[0].removeprefix("https://huggingface.co/"),
            filename=url.split("/resolve/", 1)[1].split("/", 1)[1],
            revision=url.split("/resolve/", 1)[1].split("/", 1)[0],
            local_dir=str(dst.parent),
        )
        if Path(local) != dst:
            import shutil
            shutil.move(local, dst)
        return
    except ImportError:
        pass
    import requests

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def is_safe_weights_rel(rel: str) -> bool:
    """Path safety: reject segments like .. / ., and the resolved path must stay
    under weights/ (prevents path traversal)."""
    if any(part in ("..", ".") for part in Path(rel).parts):
        return False
    try:
        resolved = (WEIGHTS_DIR / rel).resolve()
    except OSError:
        return False
    return resolved.is_relative_to(WEIGHTS_DIR.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify existing files, no download")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    parser.add_argument("--only", type=str, default="", help="only process files whose path contains this substring")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    repo = manifest.get("repo", "")
    revision = manifest.get("revision", "main")
    files = manifest.get("files", [])
    if not repo:
        sys.exit("[ERROR] manifest has no repo field")
    if not files:
        sys.exit("[ERROR] manifest has no files")

    selected = [f for f in files if not args.only or args.only.lower() in f["path"].lower()]
    ok, missing, nohf, failed = 0, [], [], []

    for item in selected:
        rel = item["path"]
        # path safety: only relative paths under weights/ that do not escape after resolve
        if not rel.startswith("weights/") or not is_safe_weights_rel(rel):
            failed.append((rel, "unsafe path (must stay under weights/)"))
            continue
        dst = ROOT / rel
        expect = item.get("sha256", "")
        if dst.exists():
            if args.dry_run:
                # dry-run only reports files to download and does not hash
                # existing files (avoids hashing large models)
                ok += 1
                continue
            actual = sha256_of(dst)
            if expect and actual != expect:
                if args.check:
                    failed.append((rel, f"sha256 mismatch (got {actual[:12]}…, want {expect[:12]}…)"))
                else:
                    dst.unlink()  # corrupted file: delete and re-download
                    missing.append(rel)
            else:
                ok += 1
                print(f"[ok]   {rel} (sha256 match)")
        else:
            if item.get("on_hf") is False:
                # not on HF: cannot download, just note the archive source
                nohf.append(rel)
                print(f"[nohf] {rel} (not in the HF repo; get it from the archive source)")
            else:
                missing.append(rel)

    if args.check:
        print(f"\n== check done: {ok} ok, {len(missing)} missing, {len(nohf)} off-hf, {len(failed)} failed")
        for rel, why in failed:
            print(f"  [FAIL] {rel}: {why}")
        for rel in missing:
            print(f"  [MISS] {rel}")
        for rel in nohf:
            print(f"  [NOHF] {rel} (archive source only)")
        return 1 if (missing or failed or nohf) else 0

    if args.dry_run:
        for rel in missing:
            print(f"[plan] {rel}")
        print(f"\n== dry-run: {len(missing)} to download (off-hf not downloadable: {len(nohf)})")
        return 0

    for rel in missing:
        item = next((f for f in files if f["path"] == rel), {})
        url = resolve_url(repo, revision, hf_path_of(item))
        print(f"[get]  {rel}  <- {hf_path_of(item)}")
        try:
            download_file(url, ROOT / rel)
            actual = sha256_of(ROOT / rel)
            expect = next((f["sha256"] for f in files if f["path"] == rel), "")
            if expect and actual != expect:
                failed.append((rel, f"sha256 mismatch after download (got {actual[:12]}…)"))
            else:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((rel, str(exc)))

    print(f"\n== done: {ok} ok, {len(missing)} downloaded/attempted, {len(failed)} failed")
    for rel, why in failed:
        print(f"  [FAIL] {rel}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
