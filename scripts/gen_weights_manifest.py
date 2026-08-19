#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_weights_manifest.py - 生成 conf/weights_manifest.json（权重版本锁定清单）。

遍历本地 weights/ 目录，为每个文件计算 sha256 与 size，写入清单供
scripts/fetch_weights.py 下载/校验使用。清单即"权重基线"的版本记录。

用法（在仓库根目录执行）:
  python3 scripts/gen_weights_manifest.py [--repo MaybeShewill-CV/mortred_model_server]
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


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default="MaybeShewill-CV/mortred_model_server")
    parser.add_argument("--revision", type=str, default="main")
    args = parser.parse_args()

    if not WEIGHTS_DIR.exists():
        sys.exit(f"[ERROR] {WEIGHTS_DIR} not found (nothing to manifest)")
    files = []
    for p in sorted(WEIGHTS_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()  # 如 weights/xxx/yyy.onnx
        files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_of(p)})
        print(f"  {rel}  {p.stat().st_size}  {files[-1]['sha256'][:12]}…")

    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "description": "weights 基线清单（sha256 锁定）；由 scripts/gen_weights_manifest.py 生成",
        "files": files,
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n== wrote {OUT.relative_to(ROOT)} with {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
