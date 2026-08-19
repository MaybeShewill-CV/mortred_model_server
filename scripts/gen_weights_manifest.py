#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_weights_manifest.py - 生成/更新 conf/weights_manifest.json（权重版本锁定清单）。

遍历本地 weights/ 目录，为每个文件计算 sha256 与 size；默认查询 HF API
（MaybeShewill-CV/mortred_model_server）获取仓库真实文件布局，为每个文件标注
`hf_path`（HF 内路径）与 `on_hf`（是否存在于 HF 仓库）。已存在且 size 相同的
文件复用原 sha256（免重算，支持快速再生成）。

用法（在仓库根目录执行）:
  python3 scripts/gen_weights_manifest.py                  # 查询 HF API + 复用哈希
  python3 scripts/gen_weights_manifest.py --no-hf-api      # 离线：不查询，on_hf 置 true
  python3 scripts/gen_weights_manifest.py --force-hash     # 强制重算全部 sha256
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
    """返回 HF 仓库文件路径集合（去 weights/ 前缀前的形态，即仓库内路径）。"""
    import requests

    r = requests.get(HF_API, timeout=30)
    r.raise_for_status()
    return {s["rfilename"] for s in r.json().get("siblings", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default="MaybeShewill-CV/mortred_model_server")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--no-hf-api", action="store_true",
                        help="离线模式：不查询 HF API，on_hf 一律置 true")
    parser.add_argument("--force-hash", action="store_true", help="强制重算全部 sha256")
    args = parser.parse_args()

    if not WEIGHTS_DIR.exists():
        sys.exit(f"[ERROR] {WEIGHTS_DIR} not found (nothing to manifest)")

    # 读取旧清单以便复用 sha256
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
            print(f"[info] HF 仓库文件数: {len(hf_files)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 查询 HF API 失败，on_hf 一律置 true: {exc}")

    files = []
    for p in sorted(WEIGHTS_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()  # 如 weights/xxx/yyy.onnx
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
        }
        files.append(entry)
        print(f"  {rel}  {size}  {sha[:12]}…  hf={'Y' if entry['on_hf'] else 'N'}")

    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "description": (
            "weights 基线清单（sha256 锁定）；hf_path/on_hf 标注 HF 仓库可达性。"
            "由 scripts/gen_weights_manifest.py 生成"
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
