#!/usr/bin/env python3
"""Generate batch optimization profiles for TRT engines without one.

Rollout companion of the yolov8 batch pilot: for every engine entry in
conf/trt_engines.json whose "profile" is null, introspects the ONNX source
(input names + static dims) and derives a batch profile
    min = [1,  rest...],  opt = [--opt, rest...],  max = [--max, rest...]
keeping every non-batch dimension at its static value.

Models whose inputs carry dynamic NON-batch dims (lightglue-style) or missing
ONNX sources are reported as needing a manual profile - never guessed.

Usage:
  python scripts/gen_trt_batch_profiles.py                 # dry run (default)
  python scripts/gen_trt_batch_profiles.py --apply         # write JSON + manifest
  python scripts/gen_trt_batch_profiles.py --only yolov5 --opt 8 --max 16

Requires the `onnx` package (pip install onnx) for introspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "conf" / "trt_engines.json"
PROFILE_DIR = ROOT / "conf" / "trt_profiles"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8-sig"))


def onnx_static_inputs(onnx_path: Path) -> list[tuple[str, list[int]]] | None:
    """Graph inputs (excluding initializers) as (name, dims); None = unusable."""
    try:
        import onnx  # noqa: PLC0415 - optional dependency, import lazily
    except ImportError:
        print("[error] the `onnx` package is required (pip install onnx)", file=sys.stderr)
        raise SystemExit(2) from None
    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}
    inputs: list[tuple[str, list[int]]] = []
    for item in model.graph.input:
        if item.name in initializer_names:
            continue
        dims: list[int] = []
        for d in item.type.tensor_type.shape.dim:
            if d.HasField("dim_param") or (d.HasField("dim_value") and d.dim_value <= 0):
                dims.append(-1)
            else:
                dims.append(d.dim_value)
        inputs.append((item.name, dims))
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--opt", type=int, default=8, help="opt batch (tactic anchor, default 8)")
    parser.add_argument("--max", type=int, default=16, help="max batch (default 16)")
    parser.add_argument("--only", default="", help="only entries whose model/path contains this")
    parser.add_argument("--apply", action="store_true",
                        help="write profile JSONs and update the manifest (default: dry run)")
    args = parser.parse_args()
    if args.opt < 1 or args.max < args.opt:
        parser.error("require 1 <= opt <= max")

    manifest = load_manifest()
    planned: list[tuple[str, list[dict]]] = []
    manual: list[str] = []
    for entry in manifest["engines"]:
        model, profile = entry["model"], entry.get("profile")
        if args.only and args.only not in model and args.only not in entry.get("engine", ""):
            continue
        if profile:
            print(f"[keep ] {model}: profile already set ({profile})")
            continue
        onnx_path = ROOT / entry["onnx"]
        if not onnx_path.exists():
            manual.append(f"{model}: onnx missing ({entry['onnx']})")
            continue
        inputs = onnx_static_inputs(onnx_path)
        if not inputs:
            manual.append(f"{model}: no graph inputs found")
            continue
        profile_items: list[dict] = []
        usable = True
        for name, dims in inputs:
            if any(d <= 0 for d in dims[1:]):
                manual.append(f"{model}: input '{name}' has dynamic non-batch dims "
                              f"{dims} (hand-written profile required)")
                usable = False
                break
            rest = dims[1:] if len(dims) > 1 else []
            profile_items.append({
                "name": name,
                "min": [1, *rest],
                "opt": [args.opt, *rest],
                "max": [args.max, *rest],
            })
        if usable and profile_items:
            planned.append((model, profile_items))

    for model, items in planned:
        out = PROFILE_DIR / f"{model}.json"
        preview = "; ".join(
            f"{i['name']}: min={i['min']} opt={i['opt']} max={i['max']}" for i in items)
        if args.apply:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            print(f"[gen ] {model}: {out.relative_to(ROOT)}  ({preview})")
        else:
            print(f"[plan] {model}: -> {out.relative_to(ROOT)}  ({preview})")
    for reason in manual:
        print(f"[man ] {reason}")
    if not args.apply and planned:
        print("\n(dry run; re-run with --apply to write the profiles and update the manifest)")
        return 0
    if args.apply and planned:
        by_model = dict(planned)
        for entry in manifest["engines"]:
            if entry["model"] in by_model:
                entry["profile"] = f"conf/trt_profiles/{entry['model']}.json"
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[gen ] manifest updated: {MANIFEST.relative_to(ROOT)}")
        print("next: ./scripts/convert_trt_engines.sh --force   # rebuild engines on the target GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
