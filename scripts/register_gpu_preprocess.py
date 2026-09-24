#!/usr/bin/env python3
"""Registers set_gpu_preprocess() in model .inl files for the GPU zero-copy pipeline.
Run from project root: python3 scripts/register_gpu_preprocess.py
"""
import re, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = "jinq::models::backend"

# descriptor template
def desc(resize, norm, color, dtype="F32", nhwc="false", pad="114", extra=""):
    lines = [
        f"        .resize = {NS}::GpuPreprocessDescriptor::Resize::{resize},",
    ]
    if norm:
        lines.append(f"        .norm = {{{norm}}},")
    lines.append(f"        .color = {NS}::GpuPreprocessDescriptor::Color::{color},")
    if pad:
        lines.append(f"        .pad_value = {pad},")
    lines.append(f"        .output_dtype = {NS}::DType::{dtype},")
    lines.append(f"        .output_nhwc = {nhwc},")
    if extra:
        lines.append(f"        {extra}")
    return "\n".join(lines)

# (file_path_relative, size_var, descriptor_string, tag)
REGISTRATIONS = [
    # ── Phase 2: direct_resize + (x-0.5)/0.5 + RGB ──
    ("src/models/scene_segmentation/bisenetv2.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.5f,0.5f,0.5f}, .std = {0.5f,0.5f,0.5f}", "RGB", nhwc="true"), "BiSeNetV2"),
    ("src/models/scene_segmentation/hrnet_segmentation.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.5f,0.5f,0.5f}, .std = {0.5f,0.5f,0.5f}", "RGB"), "HRNet"),
    ("src/models/scene_segmentation/pp_humanseg.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.5f,0.5f,0.5f}, .std = {0.5f,0.5f,0.5f}", "RGB"), "PPHumanSeg"),
    ("src/models/matting/modnet_matting.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.5f,0.5f,0.5f}, .std = {0.5f,0.5f,0.5f}", "RGB"), "ModNet"),
    ("src/models/matting/pp_matting.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.5f,0.5f,0.5f}, .std = {0.5f,0.5f,0.5f}", "RGB"), "PP-Matting"),
    # ── Phase 2: direct_resize + mean/std + BGR/RGB ──
    ("src/models/object_detection/nano_detector.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.406f,0.456f,0.485f}, .std = {0.225f,0.224f,0.229f}", "BGR"), "NanoDet"),
    ("src/models/ocr/db_text_detector.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.485f,0.456f,0.406f}, .std = {0.229f,0.224f,0.225f}", "BGR"), "DBNet"),
    ("src/models/feature_embedding/dinov2.inl", "_m_input_tensor_size",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f, .mean = {0.481f,0.457f,0.408f}, .std = {0.268f,0.261f,0.275f}", "RGB"), "DINOv2"),
    # ── Phase 2: custom scale ──
    ("src/models/enhancement/attentive_gan_derain_net.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 127.5f, .mean = {1.0f,1.0f,1.0f}", "BGR", nhwc="true"), "AttentiveGAN"),
    # LibFace: DIRECT_RESIZE_PAD_TO_MULTIPLE is registered in libface_detector.inl
    ("src/models/object_detection/libface_detector.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE_PAD_TO_MULTIPLE", ".scale = 1.0f", "BGR", pad="0",
          extra=".pre_crop_size = _m_input_size_host,\n        .align_multiple = 32,\n        .dynamic_size = true"), "LibFace"),
    # ── Phase 2: center_crop + ImageNet (classification) ──
    ("src/models/classification/mobilenetv2.inl", "_m_input_tensor_size",
     desc("CENTER_CROP", ".scale = 1.0f / 255.0f, .mean = {0.485f,0.456f,0.406f}, .std = {0.229f,0.224f,0.225f}", "RGB", nhwc="true"), "MobileNetV2"),
    ("src/models/classification/resnet.inl", "_m_input_tensor_size",
     desc("CENTER_CROP", ".scale = 1.0f / 255.0f, .mean = {0.485f,0.456f,0.406f}, .std = {0.229f,0.224f,0.225f}", "RGB", nhwc="true"), "ResNet"),
    ("src/models/classification/densenet.inl", "_m_input_tensor_size",
     desc("CENTER_CROP", ".scale = 1.0f / 255.0f, .mean = {0.485f,0.456f,0.406f}, .std = {0.229f,0.224f,0.225f}", "RGB", nhwc="true"), "DenseNet"),
    # ── Phase 3: special geometry ──
    ("src/models/mono_depth_estimation/depth_anything.inl", "_m_input_size_host",
     desc("KEEP_RATIO_PAD_ZERO", ".scale = 1.0f / 255.0f, .mean = {0.485f,0.456f,0.406f}, .std = {0.229f,0.224f,0.225f}", "BGR", pad="0"), "DepthAnything"),
    ("src/models/mono_depth_estimation/metric3d.inl", "_m_input_size_host",
     desc("KEEP_RATIO_PAD_CENTER", ".mean = {123.675f,116.28f,103.53f}, .std = {58.395f,57.12f,57.375f}", "RGB"), "Metric3D"),
    ("src/models/feature_point/superpoint.inl", "_m_input_size_host",
     desc("DIRECT_RESIZE", ".scale = 1.0f / 255.0f", "GRAY"), "SuperPoint"),
    # ── Phase 4: dynamic size + special ──
    ("src/models/object_detection/centerface_detector.inl", "",
     desc("ALIGN_TO_MULTIPLE", "", "RGB", extra=".align_multiple = 32,\n        .dynamic_size = true"), "CenterFace"),
    ("src/models/enhancement/real_esrgan.inl", "",
     desc("NONE", ".scale = 1.0f / 255.0f", "RGB", nhwc="true", extra=".dynamic_size = true"), "Real-ESRGAN"),
    ("src/models/enhancement/enlightengan.inl", "_m_input_size_host",
     desc("ALIGN_TO_MULTIPLE", ".scale = 1.0f / 255.0f, .mean = {0.5f,0.5f,0.5f}, .std = {0.5f,0.5f,0.5f}", "RGB",
          extra=".align_multiple = 16,\n        .dynamic_size = true,\n        .secondary_gray_output = true"), "EnlightenGAN"),
]

def register_model(filepath, size_var, descriptor, tag):
    full = os.path.join(ROOT, filepath)
    if not os.path.exists(full):
        return f"SKIP (not found): {filepath}"

    with open(full, "r") as f:
        content = f.read()

    if "set_gpu_preprocess" in content:
        return f"SKIP (already): {tag}"

    # Find the first "return StatusCode::OK;" in on_init
    # Insert before it
    lines = content.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "return StatusCode::OK;" in line:
            insert_idx = i
            break

    if insert_idx is None:
        return f"SKIP (no return OK): {tag}"

    # Build the registration code
    if size_var:
        hint_line = f"    this->set_image_decode_hint({size_var}, 1.0f, 1);"
    else:
        hint_line = ""  # dynamic size models may not have a size var

    reg_lines = []
    if hint_line:
        reg_lines.append(hint_line)
    reg_lines.append("    this->set_gpu_preprocess({")
    reg_lines.append(descriptor)
    reg_lines.append("    });")
    reg_lines.append("")

    # Insert before the return line
    indent = "    "
    for j, rl in enumerate(reg_lines):
        lines.insert(insert_idx + j, rl)

    with open(full, "w") as f:
        f.write("\n".join(lines))

    return f"OK: {tag}"

if __name__ == "__main__":
    results = []
    for filepath, size_var, descriptor, tag in REGISTRATIONS:
        results.append(register_model(filepath, size_var, descriptor, tag))
    for r in results:
        print(r)
    ok = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    print(f"\n{ok} registered, {skip} skipped, {len(results)} total")
