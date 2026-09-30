#!/usr/bin/env python3
"""Assert the legacy yolov8s engine path matches the product static_bs1 path.

Does not probe GPU or require weight files to exist.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "conf" / "trt_engines.json"
PRODUCT = ROOT / "conf" / "product_static_bs1.json"
BARE_SUFFIX = "yolov8s.engine"


def main() -> int:
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
    product = json.loads(PRODUCT.read_text(encoding="utf-8"))
    legacy_row = next((row for row in legacy.get("engines", []) if row.get("model") == "yolov8s"), None)
    product_row = next((row for row in product.get("models", []) if row.get("id") == "yolov8s"), None)
    if legacy_row is None or product_row is None:
        print("[FAIL] yolov8s missing from legacy manifest or product_static_bs1.json", file=sys.stderr)
        return 1
    if legacy_row.get("engine") != product_row.get("engine"):
        print(f"[FAIL] engine mismatch: {legacy_row.get('engine')} != {product_row.get('engine')}", file=sys.stderr)
        return 1
    if legacy_row.get("onnx") != product_row.get("onnx"):
        print(f"[FAIL] onnx mismatch: {legacy_row.get('onnx')} != {product_row.get('onnx')}", file=sys.stderr)
        return 1
    for row in legacy.get("engines", []):
        engine = str(row.get("engine") or "")
        if engine.endswith(BARE_SUFFIX):
            print(f"[FAIL] legacy engine still ends with {BARE_SUFFIX}: {engine}", file=sys.stderr)
            return 1
    print("[ok] legacy yolov8s engine matches product static_bs1 path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
