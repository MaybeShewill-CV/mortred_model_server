#!/usr/bin/env python3
"""Rewrite the leading batch dimension of ONNX io tensors to dynamic.

TRT optimization profiles can only widen DYNAMIC dimensions: an ONNX exported
with a static batch of 1 rejects kOPT=[8,...] at build time. This rewrites
dim0 of every graph input/output whose leading dim is the literal 1 into a
dim_param ("batch"), in place, with a .static_batch backup next to the file.

Usage:
  python3 scripts/patch_onnx_dynamic_batch.py weights/x/yolov8s.onnx [more.onnx ...]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import onnx


def patch(path: Path) -> bool:
    model = onnx.load(str(path), load_external_data=False)
    changed = []
    for item in list(model.graph.input) + list(model.graph.output):
        dims = item.type.tensor_type.shape.dim
        if dims and dims[0].HasField("dim_value") and dims[0].dim_value == 1:
            dims[0].dim_param = "batch"
            changed.append(item.name)
    if not changed:
        print(f"[keep ] {path.name}: no static leading 1 found")
        return False
    backup = path.with_suffix(path.suffix + ".static_batch")
    if not backup.exists():
        shutil.copy(path, backup)
    onnx.save(model, str(path))
    print(f"[fix ] {path.name}: dim0 -> dynamic for {changed} (backup: {backup.name})")
    return True


def patch_deep(path: Path) -> int:
    """Rewrite Reshape shape constants that bake the batch as a literal 1.

    Static ultralytics exports carry head Reshape targets like [1,144,-1] /
    [1,4,16,8400]: TensorRT resolves the conflict between a dynamic input and
    these constants by specializing the input back to batch=1 (profile
    clamped to [1,1,1]). ONNX Reshape semantics: 0 = copy the input dim, so
    dim0 1 -> 0 lets batch flow from input to output.
    """
    from onnx import numpy_helper

    model = onnx.load(str(path), load_external_data=False)
    constants = {}
    for init in model.graph.initializer:
        if init.dims:
            constants[init.name] = init
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = attr

    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) != 2:
            continue
        target = constants.get(node.input[1])
        if target is None:
            continue
        arr = (numpy_helper.to_array(target.t) if hasattr(target, "t")
               else numpy_helper.to_array(target)).copy()
        if arr.ndim == 1 and len(arr) > 0 and arr[0] == 1:
            arr[0] = 0
            new_tensor = numpy_helper.from_array(arr, node.input[1])
            if hasattr(target, "t"):  # Constant node AttributeProto
                target.t.CopyFrom(new_tensor)
            else:  # graph initializer
                target.CopyFrom(new_tensor)
            fixed += 1
            print(f"[deep] {path.name}: {node.name} reshape target dim0 1 -> 0")
    if fixed:
        backup = path.with_suffix(path.suffix + ".static_batch")
        if not backup.exists():
            shutil.copy(path, backup)
        onnx.save(model, str(path))
    return fixed


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] == "--check":
        for arg in sys.argv[2:]:
            model = onnx.load(arg, load_external_data=False)
            for item in list(model.graph.input) + list(model.graph.output):
                dims = item.type.tensor_type.shape.dim
                shape = [d.dim_param if d.HasField("dim_param") else d.dim_value for d in dims]
                print(f"{Path(arg).name}: {item.name} {shape}")
        return 0
    if sys.argv[1] == "--deep":
        for arg in sys.argv[2:]:
            total = patch_deep(Path(arg))
            if total == 0:
                # still ensure the io declarations are dynamic
                patch(Path(arg))
            else:
                patch(Path(arg))
        return 0
    for arg in sys.argv[1:]:
        patch(Path(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
