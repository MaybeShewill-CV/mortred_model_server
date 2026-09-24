#!/usr/bin/env python3
"""Dump ONNX IO for a file (used while adopting official graphs)."""
from __future__ import annotations

import sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import io_summary  # noqa: E402


def main() -> int:
    for arg in sys.argv[1:]:
        path = Path(arg)
        print(f"{path}: {io_summary(path)}")
        model = onnx.load(str(path), load_external_data=False)
        print(f"  ir={model.ir_version} opset={[o.version for o in model.opset_import]}")
        print(f"  producer={model.producer_name} {model.producer_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
