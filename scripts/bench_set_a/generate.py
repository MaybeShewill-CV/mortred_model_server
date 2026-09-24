#!/usr/bin/env python3
"""Write Set A convert manifest, TRT profiles, TensorRT overlays, and machine packs."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import MODELS, unique_converts  # noqa: E402


def onnx_inputs(path: Path) -> list[dict]:
    try:
        import onnx  # type: ignore
    except ImportError:
        return []
    model = onnx.load(str(path), load_external_data=False)
    rows = []
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(str(d.dim_param))
            else:
                dims.append(-1)
        rows.append({"name": inp.name, "dims": dims})
    return rows


def is_spatial_dynamic(dims: list) -> bool:
    if len(dims) < 4:
        return False
    spatial = dims[-3:] if not isinstance(dims[1], int) or dims[1] in (1, 3) else dims[1:3]
    # NCHW: N,C,H,W or NHWC: N,H,W,C
    if len(dims) == 4:
        nums = [d if isinstance(d, int) else -1 for d in dims]
        return any(v <= 0 for v in nums[1:])
    return any(not isinstance(d, int) or d <= 0 for d in dims)


def profile_for(row: dict, inputs: list[dict] | None = None) -> str | None:
    if row["spatial"] != "dynamic":
        return None
    cid = row["convert_id"]
    # Hardcoded specs: do not load ONNX. Rewrite on every generate so profile
    # fixes (second input, min/max) actually land.
    if cid == "libface":
        # toml [H,W]=[320,240] pads /32 -> [320,256]; 640 row is [480,640].
        specs = [
            {
                "name": "input",
                "min": [1, 3, 240, 256],
                "opt": [1, 3, 480, 640],
                "max": [1, 3, 640, 640],
            }
        ]
    elif cid == "centerface":
        specs = [
            {"name": "input", "min": [1, 3, 64, 64], "opt": [1, 3, 640, 640], "max": [1, 3, 768, 768]}
        ]
    elif cid == "enlightengan":
        # Demo lol_test_1 aligns /16 to 400x608; 512x512 max rejected width 608.
        specs = [
            {"name": "input_src", "min": [1, 3, 64, 64], "opt": [1, 3, 256, 256], "max": [1, 3, 640, 640]},
            {"name": "input_gray", "min": [1, 1, 64, 64], "opt": [1, 1, 256, 256], "max": [1, 1, 640, 640]},
        ]
    elif cid == "realesrgan":
        # NHWC RGB; max covers catalog demo [1,480,640,3] on the 8 GB card.
        specs = [
            {"name": "input", "min": [1, 64, 64, 3], "opt": [1, 256, 256, 3], "max": [1, 640, 640, 3]}
        ]
    else:
        return None
    rel = "conf/trt_profiles/set_a_%s.json" % cid
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")
    return rel


def resolve_engine(row: dict) -> str:
    """Prefer an on-disk fp16 engine; fall back to fp32 so generate does not point overlays at a missing .fp16."""
    rel16 = row["engine"].replace("\\", "/")
    p16 = ROOT / rel16
    rel32 = rel16.replace(".fp16", ".fp32")
    p32 = ROOT / rel32
    if p16.is_file() and p16.stat().st_size > 0:
        return rel16
    if p32.is_file() and p32.stat().st_size > 0:
        return rel32
    return rel16


def write_overlay(row: dict, engine_rel: str) -> str:
    src = ROOT / row["product"]
    text = src.read_text(encoding="utf-8")
    text = re.sub(r'type\s*=\s*"(mnn|onnx|tensorrt)"', 'type = "tensorrt"', text, count=1)
    text = re.sub(
        r'model_file_path\s*=\s*"[^"]+"',
        'model_file_path = "../%s"' % engine_rel.replace("\\", "/"),
        text,
        count=1,
    )
    text = re.sub(r'device\s*=\s*"(cpu|gpu)"', 'device = "gpu"', text, count=1)
    extra = row.get("extra_params") or {}
    if extra:
        for key, val in extra.items():
            if isinstance(val, list):
                rendered = "[%s]" % ", ".join(str(x) for x in val)
            elif isinstance(val, str):
                rendered = '"%s"' % val
            else:
                rendered = str(val)
            pat = re.compile(r"^(\s*)%s\s*=\s*.+$" % re.escape(key), re.M)
            if pat.search(text):
                text = pat.sub(r"\1%s = %s" % (key, rendered), text, count=1)
            else:
                text += "\n%s = %s\n" % (key, rendered)
    header = (
        "# Bench overlay: TensorRT static_bs1 engine. Not a product config.\n"
        "# id=%s catalog=%s\n\n" % (row["id"], row["catalog"])
    )
    rel = "conf/bench/set_a/%s.toml" % row["id"]
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + text, encoding="utf-8")
    return rel


def write_server_overlay(row: dict, overlay_rel: str) -> str:
    src = ROOT / row["server"]
    text = src.read_text(encoding="utf-8")
    bench_model = "../%s" % overlay_rel.replace("\\", "/")
    text = re.sub(
        r'model_config_file_path\s*=\s*"[^"]+"',
        'model_config_file_path = "%s"' % bench_model,
        text,
        count=1,
    )
    # Set A HTTP/calibrate is batch=1 vs trtexec. Product classification
    # servers ship max_batch_size=8; that packing is rejected by static_bs1
    # engines and serializes work onto one worker.
    if re.search(r"^\s*max_batch_size\s*=", text, re.M):
        text = re.sub(
            r"^(\s*max_batch_size\s*=\s*)\S+",
            r"\g<1>1",
            text,
            count=1,
            flags=re.M,
        )
    header = (
        "# Bench server overlay: model_config_file_path is the set-A overlay.\n"
        "# Not a product config. id=%s catalog=%s\n\n" % (row["id"], row["catalog"])
    )
    rel = "conf/bench/set_a/%s.server.toml" % row["id"]
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + text, encoding="utf-8")
    return rel


def write_pack(row: dict, overlay_rel: str, server_rel: str) -> str:
    rel = "logs/bench/set_a/packs/%s.toml" % row["id"]
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        text = dest.read_text(encoding="utf-8")
        if re.search(r"^model_config\s*=", text, re.M):
            text = re.sub(
                r'^model_config\s*=\s*"[^"]+"',
                'model_config = "%s"' % overlay_rel,
                text,
                count=1,
                flags=re.M,
            )
        else:
            text = text.rstrip() + '\nmodel_config = "%s"\n' % overlay_rel
        if re.search(r"^server_config\s*=", text, re.M):
            text = re.sub(
                r'^server_config\s*=\s*"[^"]+"',
                'server_config = "%s"' % server_rel,
                text,
                count=1,
                flags=re.M,
            )
        else:
            text = text.rstrip() + '\nserver_config = "%s"\n' % server_rel
        if not text.endswith("\n"):
            text += "\n"
        dest.write_text(text, encoding="utf-8")
        return rel
    body = (
        '[pack.%s]\nworker_nums = 1\nmodel_config = "%s"\nserver_config = "%s"\n'
        % (row["catalog"], overlay_rel, server_rel)
    )
    dest.write_text(body, encoding="utf-8")
    return rel


def main() -> int:
    converts = []
    for row in unique_converts():
        onnx_path = ROOT / row["onnx"]
        inputs = []
        profile = None
        if row["spatial"] == "dynamic":
            profile = profile_for(row)
            if profile is None:
                existing = ROOT / ("conf/trt_profiles/set_a_%s.json" % row["convert_id"])
                if existing.is_file():
                    profile = str(existing.relative_to(ROOT)).replace("\\", "/")
                elif onnx_path.is_file():
                    inputs = onnx_inputs(onnx_path)
                    profile = profile_for(row, inputs)
        converts.append(
            {
                "model": row["convert_id"],
                "onnx": row["onnx"],
                "engine": row["engine"],
                "fp": 1,
                "profile": profile,
                "inputs": inputs,
            }
        )
    man = {
        "_description": "Set A static_bs1 TensorRT convert manifest. Do not merge into conf/trt_engines.json.",
        "engines": [
            {
                "model": c["model"],
                "onnx": c["onnx"],
                "engine": c["engine"],
                "fp": c["fp"],
                "profile": c["profile"],
            }
            for c in converts
        ],
    }
    dest = ROOT / "conf" / "trt_engines.set_a.json"
    dest.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    jobs = []
    for row in MODELS:
        overlay = write_overlay(row, resolve_engine(row)) if row["http"] else ""
        server = write_server_overlay(row, overlay) if row["http"] else ""
        pack = write_pack(row, overlay, server) if row["http"] else ""
        jobs.append(
            {
                **row,
                "overlay": overlay,
                "server_overlay": server,
                "pack": pack,
            }
        )
    manifest = {
        "gpu_note": "RTX 2070 SUPER 8 GiB; TRTEXEC_WORKSPACE=6G; workers 1,2,4,8,16 stop on OOM",
        "jobs": jobs,
        "converts": converts,
    }
    out = ROOT / "conf" / "bench" / "set_a" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", dest)
    print("wrote", out)
    print("converts", len(converts), "http_jobs", sum(1 for j in jobs if j["http"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
