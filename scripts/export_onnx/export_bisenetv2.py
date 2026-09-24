#!/usr/bin/env python3
"""Export product-matching BiseNetV2 ONNX from MaybeShewill TF cityscapes.ckpt.

Product MNN IO (via MNN python): input_tensor, final_output [512,1024,19].
Official freeze uses NHWC input [1,512,1024,3] and squeezed softmax as final_output.
CoinCheung PyTorch pth is a different training run (NCHW, different names) and is not used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TF_REPO = Path("/tmp/mortred_p2/bisenetv2_tf_src")
CKPT = Path("/mnt/g/Codex/mortred_model_server/weights/scene_segmentation/bisenetv2/cityscapes.ckpt")
OUT_DIR = Path("/mnt/g/Codex/mortred_model_server/weights/scene_segmentation/bisenetv2")
CACHE = Path("/tmp/mortred_p2")
HEIGHT, WIDTH = 512, 1024

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _patch_tf1_imports() -> None:
    for rel in ("bisenet_model/cnn_basenet.py", "bisenet_model/bisenet_v2.py"):
        path = TF_REPO / rel
        text = path.read_text(encoding="utf-8")
        orig = text
        text = text.replace("in_channel / split", "int(in_channel // split)")
        if "import tensorflow.compat.v1 as tf" not in text:
            text = text.replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf")
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print("[bisenetv2] patched", rel)


def _prepare_tf() -> None:
    os.chdir(str(TF_REPO))
    sys.path.insert(0, str(TF_REPO))
    _patch_tf1_imports()
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()


def _restore(sess, ckpt: str):
    import tensorflow.compat.v1 as tf

    try:
        with tf.variable_scope("moving_avg"):
            ema = tf.train.ExponentialMovingAverage(0.9995)
            mapping = ema.variables_to_restore()
        saver = tf.train.Saver(mapping)
        saver.restore(sess, ckpt)
        return "ema"
    except Exception as exc:
        print("[bisenetv2] EMA restore failed:", exc)
        keep = [v for v in tf.global_variables() if "Momentum" not in v.name]
        saver = tf.train.Saver(keep)
        saver.restore(sess, ckpt)
        return "raw"


def freeze_pb() -> Path:
    import tensorflow.compat.v1 as tf
    from bisenet_model import bisenet_v2
    from local_utils.config_utils import parse_config_utils

    CACHE.mkdir(parents=True, exist_ok=True)
    frozen_pb = CACHE / "bisenetv2_cityscapes_frozen.pb"

    graph = tf.Graph()
    with graph.as_default():
        input_tensor = tf.placeholder(
            dtype=tf.float32, shape=[1, HEIGHT, WIDTH, 3], name="input_tensor"
        )
        net = bisenet_v2.BiseNetV2(phase="test", cfg=parse_config_utils.cityscapes_cfg_v2)
        _ = net.inference(input_tensor=input_tensor, name="BiseNetV2", reuse=False)
        prob = graph.get_tensor_by_name("BiseNetV2/prob:0")
        tf.squeeze(prob, axis=0, name="final_output")

        sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
        how = _restore(sess, str(CKPT))
        print("[bisenetv2] restored", how, "from", CKPT)
        gd = tf.graph_util.remove_training_nodes(sess.graph.as_graph_def())
        frozen = tf.graph_util.convert_variables_to_constants(sess, gd, ["final_output"])
        tf.train.write_graph(frozen, str(CACHE), frozen_pb.name, as_text=False)
        sess.close()
    print("[bisenetv2] frozen pb", frozen_pb, "bytes", frozen_pb.stat().st_size)
    return frozen_pb


def convert_onnx(frozen_pb: Path) -> Path:
    import subprocess

    raw = CACHE / "bisenetv2_cityscapes.tf2onnx.onnx"
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--input",
        str(frozen_pb),
        "--output",
        str(raw),
        "--inputs",
        "input_tensor:0",
        "--outputs",
        "final_output:0",
        "--opset",
        "13",
    ]
    print("[bisenetv2]", " ".join(cmd))
    subprocess.check_call(cmd)
    return raw


def write_pairs(raw: Path) -> None:
    from common import rename_io, write_pair_from_static

    # tf2onnx often keeps ":0" suffixes; product names do not.
    rename_io(raw, inputs={"input_tensor:0": "input_tensor"}, outputs={"final_output:0": "final_output"})
    static_dst = OUT_DIR / "bisenetv2_cityscapes.static_bs1.onnx"
    dyn_dst = OUT_DIR / "bisenetv2_cityscapes.dyn.onnx"
    write_pair_from_static(raw, static_dst, dyn_dst)


def main() -> int:
    if not TF_REPO.exists():
        raise SystemExit(f"missing TF source at {TF_REPO}")
    if not (CKPT.parent / (CKPT.name + ".index")).exists():
        raise SystemExit(f"missing checkpoint {CKPT}")
    _prepare_tf()
    frozen = freeze_pb()
    raw = convert_onnx(frozen)
    write_pairs(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
