#!/usr/bin/env python3
"""Export PP-HumanSeg ONNX from local PaddleSeg dygraph pdparams.

Product HTTP is v2-mobile 192: session x / softmax_0.tmp_0, static NCHW [1,3,192,192],
2-class softmax. Sibling lite 192 and v1-server 512 are exported too.

Do not reverse the .mnn. Product toml stays type=mnn.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import io_summary, sha256_of, write_pair_from_static  # noqa: E402

ROOT = Path("/mnt/g/Codex/mortred_model_server")
CACHE = Path("/tmp/mortred_p2")
OUT_DIR = ROOT / "weights" / "scene_segmentation" / "pphuman_seg"
WEIGHTS = OUT_DIR

JOBS = [
    {
        "stem": "pp_humanseg_192x192_mobile",
        "pdparams": WEIGHTS
        / "human_pp_humansegv2_mobile_192x192_pretrained"
        / "human_pp_humansegv2_mobile_192x192_pretrained"
        / "model.pdparams",
        "kind": "v2_mobile",
        "height": 192,
        "width": 192,
    },
    {
        "stem": "pp_humanseg_192x192_lite",
        "pdparams": WEIGHTS
        / "human_pp_humansegv2_lite_192x192_pretrained"
        / "human_pp_humansegv2_lite_192x192_pretrained"
        / "model.pdparams",
        "kind": "v2_lite",
        "height": 192,
        "width": 192,
    },
    {
        "stem": "pp_humanseg_512x512_server",
        "pdparams": WEIGHTS
        / "human_pp_humansegv1_server_512x512_pretrained"
        / "human_pp_humansegv1_server_512x512_pretrained"
        / "model.pdparams",
        "kind": "v1_server",
        "height": 512,
        "width": 512,
    },
]


class SoftmaxSeg(nn.Layer):
    """Single-tensor softmax wrap. Product C++ reads 2-channel NCHW softmax."""

    def __init__(self, model: nn.Layer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return F.softmax(out, axis=1)


def build_model(kind: str) -> nn.Layer:
    from paddleseg.models import DeepLabV3P, MobileSeg, PPLiteSeg
    from paddleseg.models.backbones import MobileNetV3_large_x1_0, ResNet50_vd, STDC1

    if kind == "v2_mobile":
        return PPLiteSeg(
            num_classes=2,
            backbone=STDC1(pretrained=None),
            backbone_indices=[1, 2, 3, 4],
            cm_out_ch=128,
            arm_out_chs=[4, 16, 32, 64],
            seg_head_inter_chs=[4, 16, 32, 64],
            pretrained=None,
        )
    if kind == "v2_lite":
        return MobileSeg(
            num_classes=2,
            backbone=MobileNetV3_large_x1_0(pretrained=None),
            backbone_indices=[0, 1, 2, 3],
            cm_bin_sizes=[1, 2, 4],
            cm_out_ch=128,
            arm_out_chs=[32, 64, 96, 128],
            seg_head_inter_chs=[16, 32, 32, 32],
            use_last_fuse=True,
            pretrained=None,
        )
    if kind == "v1_server":
        return DeepLabV3P(
            num_classes=2,
            backbone=ResNet50_vd(output_stride=8, multi_grid=[1, 2, 4], pretrained=None),
            backbone_indices=[0, 3],
            aspp_ratios=[1, 12, 24, 36],
            aspp_out_channels=256,
            align_corners=False,
            pretrained=None,
        )
    raise ValueError(kind)


def load_weights(model: nn.Layer, path: Path) -> None:
    state = paddle.load(str(path))
    result = model.set_dict(state)
    print(f"[pphuman] load {path.name} result={result}")
    if isinstance(result, tuple):
        miss, unexp = result
        print("[pphuman] missing", miss)
        print("[pphuman] unexpected", unexp)
        if miss:
            scale_only = all(k.endswith("._scale") for k in miss)
            if not scale_only:
                raise SystemExit(f"state_dict missing {len(miss)} keys")
            print("[pphuman] allowing missing UAFM _scale buffers (default 1)")


def mutate_and_onnx(infer_dir: Path, height: int, width: int, onnx_path: Path) -> None:
    import paddle2onnx

    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, feeds, fetches = paddle.static.load_inference_model(
        str(infer_dir / "model"),
        exe,
    )
    block = prog.global_block()
    feed_name = feeds[0] if isinstance(feeds[0], str) else feeds[0].name
    var = block.var(feed_name)
    print(f"[pphuman] infer feed {feed_name}{list(var.shape)} fetches {[v.name for v in fetches]}")
    var.desc.set_shape([1, 3, height, width])
    for op in block.ops:
        try:
            op.desc.infer_shape(block.desc)
        except Exception:
            pass
    fetch_name = fetches[0].name
    if fetch_name in block.vars:
        print(f"[pphuman] frozen {feed_name}{list(var.shape)} {fetch_name}{list(block.var(fetch_name).shape)}")
    work = CACHE / "pphuman_export" / onnx_path.stem
    work.mkdir(parents=True, exist_ok=True)
    pdmodel = work / "model.pdmodel"
    pdiparams = work / "model.pdiparams"
    prog.desc.flush()
    pdmodel.write_bytes(prog.desc.serialize_to_string())
    src_params = infer_dir / "model.pdiparams"
    if not src_params.exists():
        # paddle.jit.save writes model.pdiparams next to prefix
        src_params = Path(str(infer_dir / "model") + ".pdiparams")
    shutil.copy2(src_params, pdiparams)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx.export(str(pdmodel), str(pdiparams), str(onnx_path), 11, True, True, True, True, True)
    from onnx import load, save

    model = load(str(onnx_path), load_external_data=False)
    if model.graph.input and model.graph.input[0].name != "x":
        old = model.graph.input[0].name
        model.graph.input[0].name = "x"
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == old:
                    node.input[i] = "x"
    if model.graph.output:
        old_out = model.graph.output[0].name
        if old_out != "softmax_0.tmp_0":
            model.graph.output[0].name = "softmax_0.tmp_0"
            for node in model.graph.node:
                for i, name in enumerate(node.output):
                    if name == old_out:
                        node.output[i] = "softmax_0.tmp_0"
    save(model, str(onnx_path))
    model = load(str(onnx_path), load_external_data=False)
    for item in list(model.graph.input) + list(model.graph.output):
        dims = item.type.tensor_type.shape.dim
        if len(dims) != 4:
            continue
        dims[2].ClearField("dim_param")
        dims[2].dim_value = height
        dims[3].ClearField("dim_param")
        dims[3].dim_value = width
    save(model, str(onnx_path))
    print(f"[pphuman] onnx {onnx_path} {io_summary(onnx_path)}")


def export_one(job: dict) -> None:
    paddle.disable_static()
    pdparams = Path(job["pdparams"])
    if not pdparams.is_file():
        raise FileNotFoundError(pdparams)
    net = build_model(job["kind"])
    load_weights(net, pdparams)
    wrapped = SoftmaxSeg(net)
    wrapped.eval()
    spec = paddle.static.InputSpec(shape=[1, 3, job["height"], job["width"]], dtype="float32", name="x")
    infer_dir = CACHE / "pphuman_infer" / job["stem"]
    if infer_dir.exists():
        shutil.rmtree(infer_dir)
    infer_dir.mkdir(parents=True, exist_ok=True)
    static_net = paddle.jit.to_static(wrapped, input_spec=[spec], full_graph=True)
    paddle.jit.save(static_net, str(infer_dir / "model"))
    raw = CACHE / f"{job['stem']}.export.onnx"
    mutate_and_onnx(infer_dir, job["height"], job["width"], raw)
    static_dst = OUT_DIR / f"{job['stem']}.static_bs1.onnx"
    dyn_dst = OUT_DIR / f"{job['stem']}.dyn.onnx"
    write_pair_from_static(raw, static_dst, dyn_dst)
    print(f"[pphuman] static {io_summary(static_dst)}")
    print(f"[pphuman] dyn    {io_summary(dyn_dst)}")
    print(f"          sha static={sha256_of(static_dst)[:16]} dyn={sha256_of(dyn_dst)[:16]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        if args.only and args.only not in job["stem"]:
            continue
        print("====", job["stem"], job["kind"])
        export_one(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
