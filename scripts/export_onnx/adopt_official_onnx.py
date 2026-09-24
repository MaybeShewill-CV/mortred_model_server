#!/usr/bin/env python3
"""Adopt an already-exported official ONNX into stem.static_bs1.onnx + stem.dyn.onnx."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import rename_io, write_pair_from_static  # noqa: E402


def parse_map(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        old, new = item.split("=", 1)
        out[old] = new
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--static-dst", required=True)
    parser.add_argument("--dyn-dst", required=True)
    parser.add_argument("--rename-input", action="append", default=[])
    parser.add_argument("--rename-output", action="append", default=[])
    args = parser.parse_args()
    src = Path(args.src)
    static_dst = Path(args.static_dst)
    dyn_dst = Path(args.dyn_dst)
    write_pair_from_static(src, static_dst, dyn_dst)
    ins = parse_map(args.rename_input)
    outs = parse_map(args.rename_output)
    if ins or outs:
        rename_io(static_dst, ins, outs)
        rename_io(dyn_dst, ins, outs)
        from common import io_summary

        print(f"[renamed] static {io_summary(static_dst)}")
        print(f"[renamed] dyn    {io_summary(dyn_dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
