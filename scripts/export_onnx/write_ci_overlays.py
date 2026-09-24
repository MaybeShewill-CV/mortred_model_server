#!/usr/bin/env python3
"""Write CI-only ONNX overlays from product tomls. Product type stays mnn."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/mnt/g/Codex/mortred_model_server")
CI = ROOT / "conf" / "ci"

HEADER = """# CI-only overlay: fork-hosted ONNX Runtime CPU proof for this product id.
# Product serving stays the original conf/model/**/*.toml (mnn/tensorrt).
# Do not point conf/server at this file.

"""

JOBS = [
    (
        "conf/model/classification/mobilenetv2/mobilenetv2_config.toml",
        "conf/ci/mobilenetv2_onnx_hosted.toml",
        "../weights/classification/mobilenetv2/mobilenetv2.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/classification/resnet/resnet50_config.toml",
        "conf/ci/resnet_onnx_hosted.toml",
        "../weights/classification/resnet/resnet.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/classification/densenet/densenet121_config.toml",
        "conf/ci/densenet_onnx_hosted.toml",
        "../weights/classification/densenet/densenet.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/object_detection/yolov5/yolov5_config.toml",
        "conf/ci/yolov5_onnx_hosted.toml",
        "../weights/object_detection/yolov5/yolov5l.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/object_detection/libfacedetection/320x240_config.toml",
        "conf/ci/libface_onnx_hosted.toml",
        "../weights/object_detection/libfacedetection/face_detection_yunet_2026may.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/object_detection/yolov6/yolov6_config.toml",
        "conf/ci/yolov6_onnx_hosted.toml",
        "../weights/object_detection/yolov6/yolov6s.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/object_detection/nano_det/nanodet_config.toml",
        "conf/ci/nanodet_onnx_hosted.toml",
        "../weights/object_detection/nanodet/nanodet_plus_m_1x5.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/ocr/db_text_detector/dbnet_config.toml",
        "conf/ci/dbnet_onnx_hosted.toml",
        "../weights/ocr/db_text_detector/db_model_large.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/scene_segmentation/pphuman/pphuman_config.toml",
        "conf/ci/pphuman_onnx_hosted.toml",
        "../weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/scene_segmentation/bisenetv2/bisenetv2_config.toml",
        "conf/ci/bisenetv2_onnx_hosted.toml",
        "../weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/matting/modnet/modnet_config.toml",
        "conf/ci/modnet_onnx_hosted.toml",
        "../weights/matting/modnet/modnet_hrnet_w18.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/matting/ppmatting/ppmatting_config.toml",
        "conf/ci/ppmatting_onnx_hosted.toml",
        "../weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/feature_point/superpoint/superpoint_config.toml",
        "conf/ci/superpoint_onnx_hosted.toml",
        "../weights/feature_point/superpoint/superpoint_120x160.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/enhancement/real_esrgan/real_esrgan.toml",
        "conf/ci/realesrgan_onnx_hosted.toml",
        "../weights/enhancement/real_esrgan/realesr-general-x4v3.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/enhancement/attentive_gan_derain/attentive_gan_derain_config.toml",
        "conf/ci/attentive_gan_onnx_hosted.toml",
        "../weights/enhancement/attentive_gan_derain/attentive_gan_derain.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/enhancement/enlighten_gan/enlightengan.toml",
        "conf/ci/enlightengan_onnx_hosted.toml",
        "../weights/enhancement/enlighten_gan/enlightengan.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/feature_embedding/dinov2/dinov2_vits14_config.toml",
        "conf/ci/dinov2_vits14_onnx_hosted.toml",
        "../weights/classification/dinov2/dinov2_vits14_pretrain.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/feature_embedding/dinov2/dinov2_vitb14_config.toml",
        "conf/ci/dinov2_vitb14_onnx_hosted.toml",
        "../weights/classification/dinov2/dinov2_vitb14_pretrain.static_bs1.onnx",
        1,
    ),
    (
        "conf/model/feature_embedding/dinov2/dinov2_vitl14_config.toml",
        "conf/ci/dinov2_vitl14_onnx_hosted.toml",
        "../weights/classification/dinov2/dinov2_vitl14_pretrain.static_bs1.onnx",
        1,
    ),
]


def rewrite_backend(text: str, onnx_path: str, replacements: int) -> str:
    out = []
    remaining = replacements
    for line in text.splitlines(True):
        stripped = line.lstrip()
        if remaining > 0 and stripped.startswith("type = "):
            indent = line[: len(line) - len(stripped)]
            line = f'{indent}type = "onnx"\n'
        elif remaining > 0 and stripped.startswith("model_file_path = "):
            indent = line[: len(line) - len(stripped)]
            line = f'{indent}model_file_path = "{onnx_path}"\n'
            remaining -= 1
        elif remaining >= 0 and stripped.startswith("device = "):
            indent = line[: len(line) - len(stripped)]
            line = f'{indent}device = "cpu"\n'
        out.append(line)
    return "".join(out)


def main() -> int:
    CI.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_rel, onnx_path, n in JOBS:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        body = rewrite_backend(src.read_text(), onnx_path, n)
        dst.write_text(HEADER + body)
        print(f"wrote {dst_rel}")

    clip_src = ROOT / "conf/model/openai_clip/vit_b_32_config.toml"
    clip_text = clip_src.read_text()
    clip_text = clip_text.replace('type = "mnn"', 'type = "onnx"')
    clip_text = clip_text.replace(
        'model_file_path = "../weights/openai_clip/vit-b-32/visual.mnn"',
        'model_file_path = "../weights/openai_clip/vit-b-32/visual.static_bs1.onnx"',
    )
    clip_text = clip_text.replace(
        'model_file_path = "../weights/openai_clip/vit-b-32/textual.mnn"',
        'model_file_path = "../weights/openai_clip/vit-b-32/textual.static_bs1.onnx"',
    )
    clip_text = clip_text.replace('device = "gpu"', 'device = "cpu"')
    (ROOT / "conf/ci/clip_onnx_hosted.toml").write_text(HEADER + clip_text)
    print("wrote conf/ci/clip_onnx_hosted.toml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
