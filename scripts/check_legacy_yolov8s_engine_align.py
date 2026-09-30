#!/usr/bin/env python3
"""Assert frozen legacy engine/onnx paths match product_static_bs1.

Covers yolov8s plus the six depth/metric3d/hrnet pairs. Also asserts the
matching onnx_sources runtime_path rows equal the product engine. Does not
probe GPU or require weight files to exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "conf" / "trt_engines.json"
PRODUCT = ROOT / "conf" / "product_static_bs1.json"
ONNX_SOURCES = ROOT / "conf" / "onnx_sources.json"
PRODUCT_LABEL = "product_static_bs1"
BARE_SUFFIX = "yolov8s.engine"

# Frozen legacy trt_engines model -> product_static_bs1 id
LEGACY_TO_PRODUCT: dict[str, str] = {
    "yolov8s": "yolov8s",
    "depth_anything_vits14": "depth_vits14",
    "depth_anything_vitb14": "depth_vitb14",
    "depth_anything_vitl14": "depth_vitl14",
    "metric3d_512x1088": "metric3d_512",
    "metric3d_1088x1920": "metric3d_1088",
    "hrnetw48_ccd_fp32": "hrnet",
}

# Frozen onnx_sources (id, variant|None) -> product_static_bs1 id
ONNX_RUNTIME_TO_PRODUCT: dict[tuple[str, str | None], str] = {
    ("HRNET", None): "hrnet",
    ("DEPTH_ANYTHING", "vits14"): "depth_vits14",
    ("DEPTH_ANYTHING", "vitb14"): "depth_vitb14",
    ("DEPTH_ANYTHING", "vitl14"): "depth_vitl14",
    ("METRIC3D", "512x1088"): "metric3d_512",
    ("METRIC3D", "1088x1920"): "metric3d_1088",
}


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def main() -> int:
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
    product = json.loads(PRODUCT.read_text(encoding="utf-8"))
    onnx_sources = json.loads(ONNX_SOURCES.read_text(encoding="utf-8"))

    legacy_by_model = {
        str(row.get("model") or ""): row for row in legacy.get("engines", [])
    }
    product_by_id = {
        str(row.get("id") or ""): row for row in product.get("models", [])
    }

    for legacy_model, product_id in LEGACY_TO_PRODUCT.items():
        legacy_row = legacy_by_model.get(legacy_model)
        product_row = product_by_id.get(product_id)
        if legacy_row is None or product_row is None:
            return _fail(
                f"{legacy_model} missing from legacy manifest or "
                f"{PRODUCT_LABEL}.json (product id {product_id})"
            )
        legacy_engine = legacy_row.get("engine")
        product_engine = product_row.get("engine")
        if legacy_engine != product_engine:
            return _fail(
                f"{legacy_model} engine mismatch: legacy={legacy_engine} "
                f"product={product_engine} ({PRODUCT_LABEL})"
            )
        legacy_onnx = legacy_row.get("onnx")
        product_onnx = product_row.get("onnx")
        if legacy_onnx != product_onnx:
            return _fail(
                f"{legacy_model} onnx mismatch: legacy={legacy_onnx} "
                f"product={product_onnx} ({PRODUCT_LABEL})"
            )

    for row in legacy.get("engines", []):
        engine = str(row.get("engine") or "")
        if engine.endswith(BARE_SUFFIX):
            return _fail(
                f"legacy engine still ends with {BARE_SUFFIX}: {engine} "
                f"({PRODUCT_LABEL})"
            )

    onnx_index: dict[tuple[str, str | None], dict] = {}
    for row in onnx_sources.get("models", []):
        key = (str(row.get("id") or ""), row.get("variant"))
        onnx_index[key] = row

    for key, product_id in ONNX_RUNTIME_TO_PRODUCT.items():
        src_row = onnx_index.get(key)
        product_row = product_by_id.get(product_id)
        if src_row is None or product_row is None:
            return _fail(
                f"onnx_sources {key} or product id {product_id} missing "
                f"({PRODUCT_LABEL})"
            )
        runtime = src_row.get("runtime_path")
        product_engine = product_row.get("engine")
        if runtime != product_engine:
            return _fail(
                f"onnx_sources {key[0]}"
                + (f"/{key[1]}" if key[1] else "")
                + f" runtime_path mismatch: legacy={runtime} "
                f"product={product_engine} ({PRODUCT_LABEL})"
            )

    print(
        "[ok] legacy engine/onnx pairs "
        f"({', '.join(LEGACY_TO_PRODUCT)}) match {PRODUCT_LABEL}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
