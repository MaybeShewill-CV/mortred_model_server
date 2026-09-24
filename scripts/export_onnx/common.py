#!/usr/bin/env python3
"""Helpers for P2 ONNX pairs: stem.static_bs1.onnx + stem.dyn.onnx (opset 17)."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _rename_in_graph(graph: onnx.GraphProto, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    for item in list(graph.input) + list(graph.output) + list(graph.value_info):
        if item.name in mapping:
            item.name = mapping[item.name]
    for init in graph.initializer:
        if init.name in mapping:
            init.name = mapping[init.name]
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in mapping:
                node.input[i] = mapping[name]
        for i, name in enumerate(node.output):
            if name in mapping:
                node.output[i] = mapping[name]


def rename_io(path: Path, inputs: dict[str, str] | None = None, outputs: dict[str, str] | None = None) -> None:
    model = onnx.load(str(path), load_external_data=False)
    mapping = {}
    mapping.update(inputs or {})
    mapping.update(outputs or {})
    _rename_in_graph(model.graph, mapping)
    onnx.save(model, str(path))


def fold_weight_inputs(model: onnx.ModelProto) -> int:
    """Drop graph.input entries that are already initializers (NanoDet / CenterFace)."""
    init_names = {init.name for init in model.graph.initializer}
    keep = [inp for inp in model.graph.input if inp.name not in init_names]
    dropped = len(model.graph.input) - len(keep)
    if dropped:
        del model.graph.input[:]
        model.graph.input.extend(keep)
    return dropped


def _is_batch_dim(dim) -> bool:
    if dim.HasField("dim_param") and dim.dim_param in ("batch", "N", "n"):
        return True
    return dim.HasField("dim_value") and dim.dim_value == 1


def set_batch_dim_model(model: onnx.ModelProto, *, dynamic: bool) -> None:
    """Rewrite leading dim only when it is already a batch (1 or 'batch'). Spatial dims stay."""
    init_names = {init.name for init in model.graph.initializer}
    for item in list(model.graph.input) + list(model.graph.output):
        if item.name in init_names:
            continue
        dims = item.type.tensor_type.shape.dim
        if not dims or not _is_batch_dim(dims[0]):
            continue
        if dynamic:
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "batch"
        else:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1


def set_batch_dim(path: Path, *, dynamic: bool) -> None:
    model = onnx.load(str(path), load_external_data=False)
    fold_weight_inputs(model)
    normalize_dummy_batch(model)
    set_batch_dim_model(model, dynamic=dynamic)
    onnx.save(model, str(path))


def io_summary(path: Path) -> str:
    model = onnx.load(str(path), load_external_data=False)

    def fmt(items) -> str:
        parts = []
        for item in items:
            dims = item.type.tensor_type.shape.dim
            shape = [d.dim_param if d.HasField("dim_param") else d.dim_value for d in dims]
            parts.append(f"{item.name}{shape}")
        return ", ".join(parts)

    return f"in=[{fmt(model.graph.input)}] out=[{fmt(model.graph.output)}]"


def normalize_dummy_batch(model: onnx.ModelProto) -> None:
    """Force a dummy export batch (CenterFace N=10) down to 1 on matching IO."""
    if not model.graph.input:
        return
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if not dims or not dims[0].HasField("dim_value"):
        return
    dummy = dims[0].dim_value
    if dummy == 1:
        return
    for item in list(model.graph.input) + list(model.graph.output):
        item_dims = item.type.tensor_type.shape.dim
        if item_dims and item_dims[0].HasField("dim_value") and item_dims[0].dim_value == dummy:
            item_dims[0].dim_value = 1


def set_rank4_spatial_dynamic(model: onnx.ModelProto, *, h_name: str, w_name: str, out_h: str, out_w: str) -> None:
    for item in model.graph.input:
        dims = item.type.tensor_type.shape.dim
        if len(dims) != 4:
            continue
        dims[2].ClearField("dim_value")
        dims[2].dim_param = h_name
        dims[3].ClearField("dim_value")
        dims[3].dim_param = w_name
    for item in model.graph.output:
        dims = item.type.tensor_type.shape.dim
        if len(dims) != 4:
            continue
        dims[2].ClearField("dim_value")
        dims[2].dim_param = out_h
        dims[3].ClearField("dim_value")
        dims[3].dim_param = out_w


def apply_reshape_batch_passthrough(path: Path) -> int:
    """Rewrite Reshape targets that bake batch as literal 1 into 0 (copy input dim)."""
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
        arr = (numpy_helper.to_array(target.t) if hasattr(target, "t") else numpy_helper.to_array(target)).copy()
        if arr.ndim == 1 and len(arr) > 0 and arr[0] == 1:
            arr[0] = 0
            new_tensor = numpy_helper.from_array(arr, node.input[1])
            if hasattr(target, "t"):
                target.t.CopyFrom(new_tensor)
            else:
                target.CopyFrom(new_tensor)
            fixed += 1
    if fixed:
        onnx.save(model, str(path))
    return fixed


def write_pair_from_static(src: Path, static_dst: Path, dyn_dst: Path) -> None:
    static_dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != static_dst.resolve():
        shutil.copy2(src, static_dst)
    set_batch_dim(static_dst, dynamic=False)
    shutil.copy2(static_dst, dyn_dst)
    set_batch_dim(dyn_dst, dynamic=True)
    print(f"[pair] {static_dst.name}: {io_summary(static_dst)}")
    print(f"[pair] {dyn_dst.name}:    {io_summary(dyn_dst)}")
    print(f"       sha static={sha256_of(static_dst)[:16]} dyn={sha256_of(dyn_dst)[:16]}")
