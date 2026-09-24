#!/usr/bin/env python3
"""Export product-matching AttentiveGAN derain ONNX from derain_gan.ckpt-100000.

Product MNN: input_tensor [1,3,240,360] internal / final_output [240,360,3] HWC.
Official freeze is NHWC [1,240,360,3] + squeezed tanh skip_3 as final_output.
The local attentive_gan_derain.pb is a text training GraphDef (tfrecords paths), not a frozen net.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TF_REPO = Path("/tmp/mortred_p2/attentive_gan_tf_src")
CKPT = Path("/mnt/g/Codex/mortred_model_server/weights/enhancement/attentive_gan_derain/derain_gan.ckpt-100000")
OUT_DIR = Path("/mnt/g/Codex/mortred_model_server/weights/enhancement/attentive_gan_derain")
CACHE = Path("/tmp/mortred_p2")
HEIGHT, WIDTH = 240, 360

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _patch_sources() -> None:
    cnn = TF_REPO / "attentive_gan_model" / "cnn_basenet.py"
    text = cnn.read_text(encoding="utf-8")
    orig = text
    text = text.replace("import tensorflow.contrib.layers as tf_layer\n", "")
    text = text.replace("tf.contrib.layers.variance_scaling_initializer()", "tf.variance_scaling_initializer()")
    text = text.replace("in_channel / split", "int(in_channel // split)")
    if "import tensorflow.compat.v1 as tf" not in text:
        text = text.replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf")
    if text != orig:
        cnn.write_text(text, encoding="utf-8")
        print("[derain] patched cnn_basenet.py")

    for rel in (
        "attentive_gan_model/derain_drop_net.py",
        "attentive_gan_model/attentive_gan_net.py",
        "attentive_gan_model/discriminative_net.py",
        "attentive_gan_model/vgg16.py",
    ):
        path = TF_REPO / rel
        t = path.read_text(encoding="utf-8")
        if "import tensorflow.compat.v1 as tf" not in t:
            path.write_text(t.replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf"), encoding="utf-8")
            print("[derain] patched", rel)

    cfg = TF_REPO / "config" / "global_config.py"
    # Avoid easydict; inference does not read TRAIN.* from this module at graph-build time.
    if "easydict" in cfg.read_text(encoding="utf-8"):
        cfg.write_text(
            "class _C:\n    pass\n\n"
            "cfg = _C()\n"
            "cfg.TRAIN = _C()\n"
            "cfg.TEST = _C()\n",
            encoding="utf-8",
        )
        print("[derain] stubbed global_config.py")


def _prepare_tf() -> None:
    os.chdir(str(TF_REPO))
    sys.path.insert(0, str(TF_REPO))
    _patch_sources()
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()


def freeze_pb() -> Path:
    import tensorflow.compat.v1 as tf
    from attentive_gan_model import derain_drop_net

    CACHE.mkdir(parents=True, exist_ok=True)
    frozen_pb = CACHE / "attentive_gan_derain_frozen.pb"

    graph = tf.Graph()
    with graph.as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, HEIGHT, WIDTH, 3], name="input_tensor")
        net = derain_drop_net.DeRainNet(phase=tf.constant("test", tf.string))
        output, _attention = net.inference(input_tensor=input_tensor, name="derain_net")
        tf.squeeze(output, axis=0, name="final_output")

        sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
        keep = [v for v in tf.global_variables() if "Momentum" not in v.name]
        saver = tf.train.Saver(keep)
        saver.restore(sess, str(CKPT))
        print("[derain] restored", CKPT, "n_vars", len(keep))
        gd = tf.graph_util.remove_training_nodes(sess.graph.as_graph_def())
        frozen = tf.graph_util.convert_variables_to_constants(sess, gd, ["final_output"])
        tf.train.write_graph(frozen, str(CACHE), frozen_pb.name, as_text=False)
        sess.close()
    print("[derain] frozen pb", frozen_pb, "bytes", frozen_pb.stat().st_size)
    return frozen_pb


def convert_onnx(frozen_pb: Path) -> Path:
    import subprocess

    raw = CACHE / "attentive_gan_derain.tf2onnx.onnx"
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
    print("[derain]", " ".join(cmd))
    subprocess.check_call(cmd)
    return raw


def write_pairs(raw: Path) -> None:
    from common import rename_io, write_pair_from_static

    rename_io(raw, inputs={"input_tensor:0": "input_tensor"}, outputs={"final_output:0": "final_output"})
    write_pair_from_static(
        raw,
        OUT_DIR / "attentive_gan_derain.static_bs1.onnx",
        OUT_DIR / "attentive_gan_derain.dyn.onnx",
    )


def main() -> int:
    if not TF_REPO.exists():
        raise SystemExit(f"missing TF source at {TF_REPO}")
    if not Path(str(CKPT) + ".index").exists():
        raise SystemExit(f"missing checkpoint {CKPT}")
    _prepare_tf()
    frozen = freeze_pb()
    raw = convert_onnx(frozen)
    write_pairs(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
