#!/usr/bin/env python3
"""Copy a DINOv2 ONNX and mark block tensors as graph outputs.

TensorRT 10.3 folds this ViT into a fused function whose 768-d embedding
does not match ONNX Runtime (cosine about 0.19 against a 0.999 threshold).
Marking each transformer block's attention Q/K multiplies and its Add
nodes as graph outputs makes the embedding named "output" match. The
stored ONNX is left unchanged; convert_trt_engines.sh feeds this copy to
trtexec only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, ValueInfoProto


def materialize(src: Path, dst: Path) -> int:
    model = onnx.load(src)
    added = 0
    for node in model.graph.node:
        name = node.name
        keep = (
            name.endswith("/attn/Mul")
            or name.endswith("/attn/Mul_1")
            or name.endswith("/Add")
            or name.endswith("/Add_1")
        )
        if not keep or "/blocks/blocks." not in name:
            continue
        info = ValueInfoProto()
        info.name = node.output[0]
        info.type.tensor_type.elem_type = TensorProto.FLOAT
        model.graph.output.append(info)
        added += 1
    if added == 0:
        raise SystemExit("no DINOv2 block tensors matched: %s" % src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, dst)
    return added


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: materialize_dinov2_trt_outputs.py SRC.onnx DST.onnx")
    added = materialize(Path(sys.argv[1]), Path(sys.argv[2]))
    print("materialized_outputs %d" % added)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
