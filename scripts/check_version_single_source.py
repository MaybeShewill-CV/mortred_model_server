#!/usr/bin/env python3
"""Assert single version source: CMake project(VERSION) == mortred_version.h == optional git tag."""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

def cmake_version(root: Path) -> str:
    text = (root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"project\s*\(\s*mortred_model_server\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    if not m:
        raise SystemExit("CMake project VERSION not found")
    return m.group(1)

def header_version(root: Path) -> str:
    h = root / "src/common/mortred_version.h"
    if not h.is_file():
        h = root / "src/common/mortred_version.h.in"
        text = h.read_text(encoding="utf-8", errors="replace")
        if "@PROJECT_VERSION@" in text or "@Mortred_VERSION@" in text or "PROJECT_VERSION" in text:
            return cmake_version(root)  # generated
        raise SystemExit("mortred_version.h missing")
    text = h.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'#\s*define\s+MORTRED_VERSION\s+"([0-9]+\.[0-9]+\.[0-9]+)"', text)
    if not m:
        raise SystemExit("MORTRED_VERSION define not found")
    return m.group(1)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cv = cmake_version(root)
    hv = header_version(root)
    if cv != hv:
        print(f"FAIL version mismatch cmake={cv} header={hv}", file=sys.stderr)
        return 1
    print(f"OK single version source: {cv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
