#!/usr/bin/env python3
"""Export PP-Matting ONNX from PaddleSeg inference dirs.

paddle2onnx cannot convert the dynamic AdaptiveAvgPool ASPP (1x1/3x3/5x5) until
feed spatial dims are frozen. Product C++ also requires static H/W from the
session (area>0), names img / tmp_75, NCHW 512 for the HTTP HRNet-W18 model.

Do not reverse the .mnn. Product toml stays type=mnn.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import io_summary, sha256_of, write_pair_from_static  # noqa: E402

ROOT = Path("/mnt/g/Codex/mortred_model_server")
CACHE = Path("/tmp/mortred_p2")
OUT_DIR = ROOT / "weights" / "matting" / "ppmatting"

JOBS = [
    {
        "stem": "ppmatting_hrnet_w18_human_512",
        "src_candidates": [
            ROOT / "weights/matting/ppmatting/ppmatting-hrnet_w18-human_512/ppmatting-hrnet_w18-human_512",
            CACHE / "paddle/ppmatting512/ppmatting-hrnet_w18-human_512",
        ],
        "height": 512,
        "width": 512,
        "fetch": "tmp_75",
    },
    {
        "stem": "ppmatting_hrnet_w18_human_1024",
        "src_candidates": [
            ROOT / "weights/matting/ppmatting/ppmatting-hrnet_w18-human_1024/ppmatting-hrnet_w18-human_1024",
        ],
        "height": 512,
        "width": 512,
        "fetch": "tmp_75",
    },
    {
        "stem": "pp_humanmatting-resnet34_vd",
        "src_candidates": [
            ROOT / "weights/matting/ppmatting/pp-humanmatting-resnet34_vd/pp-humanmatting-resnet34_vd",
        ],
        "height": 2048,
        "width": 2048,
        "fetch": "clip_1.tmp_0",
    },
    {
        "stem": "ppmattingv2_stdc1_human_512",
        "src_candidates": [
            ROOT / "weights/matting/ppmatting/ppmattingv2-stdc1-human_512/ppmattingv2-stdc1-human_512",
        ],
        "height": 512,
        "width": 512,
        "fetch": "sigmoid_5.tmp_0",
    },
]


def _resolve(candidates: list[Path]) -> Path:
    for path in candidates:
        if (path / "model.pdmodel").is_file() and (path / "model.pdiparams").is_file():
            return path
    raise FileNotFoundError("missing paddle inference dir: " + ", ".join(str(p) for p in candidates))


def freeze_and_export(src: Path, height: int, width: int, onnx_path: Path) -> None:
    import paddle
    import paddle2onnx

    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, feeds, fetches = paddle.static.load_inference_model(
        str(src), exe, model_filename="model.pdmodel", params_filename="model.pdiparams"
    )
    block = prog.global_block()
    feed_name = feeds[0] if isinstance(feeds[0], str) else feeds[0].name
    img = block.var(feed_name)
    print(f"[ppmatting] {src} feed {feed_name}{list(img.shape)} fetches {[v.name for v in fetches]}")
    img.desc.set_shape([1, 3, height, width])
    for op in block.ops:
        try:
            op.desc.infer_shape(block.desc)
        except Exception:
            pass
    fetch_name = fetches[0].name
    if fetch_name in block.vars:
        print(f"[ppmatting] after freeze {feed_name}{list(img.shape)} {fetch_name}{list(block.var(fetch_name).shape)}")
    prog.desc.flush()
    work = CACHE / "ppmatting_export" / onnx_path.stem
    work.mkdir(parents=True, exist_ok=True)
    pdmodel = work / "model.pdmodel"
    pdiparams = work / "model.pdiparams"
    pdmodel.write_bytes(prog.desc.serialize_to_string())
    shutil.copy2(src / "model.pdiparams", pdiparams)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx.export(str(pdmodel), str(pdiparams), str(onnx_path), 11, True, True, True, True, True)
    print(f"[ppmatting] onnx {onnx_path} {io_summary(onnx_path)}")


def unsqueeze_scalar_concat(onnx_path: Path) -> int:
    """TRT 10 rejects Concat(axis=0) of rank-0 Squeeze slices (p2o.Concat.35)."""
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path), load_external_data=True)
    graph = model.graph
    produced = {out: node for node in graph.node for out in node.output}
    inserted = 0
    new_nodes: list = []
    for node in graph.node:
        if node.op_type != "Concat":
            new_nodes.append(node)
            continue
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        if axis != 0:
            new_nodes.append(node)
            continue
        new_inputs = []
        for inp in node.input:
            prod = produced.get(inp)
            squeezed = False
            cur = prod
            for _ in range(4):
                if cur is None:
                    break
                if cur.op_type == "Squeeze":
                    squeezed = True
                    break
                if cur.op_type in ("Cast", "Add") and cur.input:
                    cur = produced.get(cur.input[0])
                    continue
                break
            if not squeezed:
                new_inputs.append(inp)
                continue
            unsq_name = inp + "_unsq0"
            unsq = helper.make_node(
                "Unsqueeze",
                [inp],
                [unsq_name],
                name="fix.Unsqueeze." + inp.replace(".", "_"),
                axes=[0],
            )
            new_nodes.append(unsq)
            produced[unsq_name] = unsq
            new_inputs.append(unsq_name)
            inserted += 1
        if new_inputs != list(node.input):
            del node.input[:]
            node.input.extend(new_inputs)
        new_nodes.append(node)
    if inserted:
        del graph.node[:]
        graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))
    print(f"[ppmatting] unsqueeze_scalar_concat {onnx_path.name} inserted={inserted}")
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="stem substring filter")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        if args.only and args.only not in job["stem"]:
            continue
        src = _resolve(job["src_candidates"])
        raw = CACHE / f"{job['stem']}.export.onnx"
        freeze_and_export(src, job["height"], job["width"], raw)
        static_dst = OUT_DIR / f"{job['stem']}.static_bs1.onnx"
        dyn_dst = OUT_DIR / f"{job['stem']}.dyn.onnx"
        write_pair_from_static(raw, static_dst, dyn_dst)
        if "resnet34" in job["stem"]:
            unsqueeze_scalar_concat(static_dst)
            unsqueeze_scalar_concat(dyn_dst)
        print(f"[ppmatting] static {io_summary(static_dst)}")
        print(f"[ppmatting] dyn    {io_summary(dyn_dst)}")
        print(f"            sha static={sha256_of(static_dst)[:16]} dyn={sha256_of(dyn_dst)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
