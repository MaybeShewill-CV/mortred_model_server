#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_weights.py - 从 Hugging Face 自动下载/校验模型权重。

替代"从百度网盘手动下载"的流程：权重清单 conf/weights_manifest.json 声明
repo、revision 与每个文件（路径/size/sha256），本脚本按清单下载到 weights/，
已存在且 sha256 匹配的文件自动跳过（可断点续传）。

用法（在仓库根目录执行）:
  python3 scripts/fetch_weights.py                    # 下载全部缺失权重
  python3 scripts/fetch_weights.py --only yolov8      # 只下载路径含 yolov8 的文件
  python3 scripts/fetch_weights.py --check            # 只校验已存在文件，不下载
  python3 scripts/fetch_weights.py --dry-run          # 打印将要下载的文件
  python3 scripts/fetch_weights.py --manifest FILE    # 指定清单（默认 conf/weights_manifest.json）

依赖: requests（或 huggingface_hub，二选一即可）；生成清单用
  python3 scripts/gen_weights_manifest.py
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
    # utf-8-sig：容忍 Windows 编辑器可能写入的 BOM
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_url(repo: str, revision: str, rel: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{rel}"


def download_file(url: str, dst: Path) -> None:
    # 优先 huggingface_hub（带断点续传/多线程），退回 requests
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
    """路径安全：拒绝含 .. / . 段，且 resolve 后必须仍在 weights/ 内（防路径穿越）。"""
    if any(part in ("..", ".") for part in Path(rel).parts):
        return False
    try:
        resolved = (WEIGHTS_DIR / rel).resolve()
    except OSError:
        return False
    return resolved.is_relative_to(WEIGHTS_DIR.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验已存在文件，不下载")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要下载的文件")
    parser.add_argument("--only", type=str, default="", help="只处理路径包含该子串的文件")
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
    ok, missing, failed = 0, [], []

    for item in selected:
        rel = item["path"]
        # 路径安全：只允许 weights/ 内、且 resolve 后不越界的相对路径
        if not rel.startswith("weights/") or not is_safe_weights_rel(rel):
            failed.append((rel, "unsafe path (must stay under weights/)"))
            continue
        dst = ROOT / rel
        expect = item.get("sha256", "")
        if dst.exists():
            actual = sha256_of(dst)
            if expect and actual != expect:
                if args.check:
                    failed.append((rel, f"sha256 mismatch (got {actual[:12]}…, want {expect[:12]}…)"))
                else:
                    dst.unlink()  # 损坏文件：删除重新下载
                    missing.append(rel)
            else:
                ok += 1
                print(f"[ok]   {rel} (sha256 match)")
        else:
            missing.append(rel)

    if args.check:
        print(f"\n== check done: {ok} ok, {len(missing)} missing, {len(failed)} failed")
        for rel, why in failed:
            print(f"  [FAIL] {rel}: {why}")
        for rel in missing:
            print(f"  [MISS] {rel}")
        return 1 if (missing or failed) else 0

    if args.dry_run:
        for rel in missing:
            print(f"[plan] {rel}")
        print(f"\n== dry-run: {len(missing)} to download")
        return 0

    for rel in missing:
        url = resolve_url(repo, revision, rel)
        print(f"[get]  {rel}")
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
