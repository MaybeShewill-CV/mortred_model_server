#!/usr/bin/env python3
"""Set ONNX input/output spatial dims (H,W) while keeping batch as requested."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))


def set_nchw_hw(item, h: int | None, w: int | None) -> None:
    dims = item.type.tensor_type.shape.dim
    if len(dims) != 4:
        return
    if h is not None:
        dims[2].ClearField("dim_param")
        dims[2].dim_value = h
    if w is not None:
        dims[3].ClearField("dim_param")
        dims[3].dim_value = w


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    model = onnx.load(args.src, load_external_data=False)
    for item in list(model.graph.input) + list(model.graph.output):
        set_nchw_hw(item, args.height, args.width)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.dst)
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
