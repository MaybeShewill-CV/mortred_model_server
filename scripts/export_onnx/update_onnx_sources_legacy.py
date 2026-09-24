#!/usr/bin/env python3
"""Stamp adopted dual-pair paths onto preexisting entries in conf/onnx_sources.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/g/Codex/mortred_model_server")
PATH = ROOT / "conf" / "onnx_sources.json"

NOTE = (
    "Adopted from the existing ONNX: original file kept for product toml / "
    "trt_engines.json / HF names. Dual interchange is {stem}.static_bs1.onnx "
    "(batch=1) and {stem}.dyn.onnx (symbolic batch). Spatial dims unchanged "
    "except CenterFace, whose dummy 32×32 was opened to height/width."
)

FILE_PATCH = {
    "weights/object_detection/yolov7/yolov7.onnx": (
        "weights/object_detection/yolov7/yolov7.static_bs1.onnx",
        "weights/object_detection/yolov7/yolov7.dyn.onnx",
    ),
    "weights/object_detection/yolov8/yolov8s.onnx": (
        "weights/object_detection/yolov8/yolov8s.static_bs1.onnx",
        "weights/object_detection/yolov8/yolov8s.dyn.onnx",
    ),
    "weights/object_detection/centerface_detection/centerface.onnx": (
        "weights/object_detection/centerface_detection/centerface.static_bs1.onnx",
        "weights/object_detection/centerface_detection/centerface.dyn.onnx",
    ),
    "weights/scene_segmentation/hrnet/hrnetw48_ccd.onnx": (
        "weights/scene_segmentation/hrnet/hrnetw48_ccd.static_bs1.onnx",
        "weights/scene_segmentation/hrnet/hrnetw48_ccd.dyn.onnx",
    ),
    "weights/scene_segmentation/msocrnet/msocrnet_fp16.onnx": (
        "weights/scene_segmentation/msocrnet/msocrnet_fp16.static_bs1.onnx",
        "weights/scene_segmentation/msocrnet/msocrnet_fp16.dyn.onnx",
    ),
    "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.onnx": (
        "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.static_bs1.onnx",
        "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.dyn.onnx",
    ),
    "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.onnx": (
        "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.static_bs1.onnx",
        "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.dyn.onnx",
    ),
    "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.onnx": (
        "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.static_bs1.onnx",
        "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.dyn.onnx",
    ),
    "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.onnx": (
        "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.static_bs1.onnx",
        "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.dyn.onnx",
    ),
    "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.onnx": (
        "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.static_bs1.onnx",
        "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.dyn.onnx",
    ),
    "weights/feature_point/lightglue/extractor.onnx": (
        "weights/feature_point/lightglue/extractor.static_bs1.onnx",
        "weights/feature_point/lightglue/extractor.dyn.onnx",
    ),
    "weights/feature_point/lightglue/matcher.onnx": (
        "weights/feature_point/lightglue/matcher.static_bs1.onnx",
        "weights/feature_point/lightglue/matcher.dyn.onnx",
    ),
    "weights/sam/fastsam_s/FastSAM-s.onnx": (
        "weights/sam/fastsam_s/FastSAM-s.static_bs1.onnx",
        "weights/sam/fastsam_s/FastSAM-s.dyn.onnx",
    ),
    "weights/sam/fastsam_x/FastSAM-x.onnx": (
        "weights/sam/fastsam_x/FastSAM-x.static_bs1.onnx",
        "weights/sam/fastsam_x/FastSAM-x.dyn.onnx",
    ),
    "weights/sam/mobile_sam/mobile_sam_encoder.onnx": (
        "weights/sam/mobile_sam/mobile_sam_encoder.static_bs1.onnx",
        "weights/sam/mobile_sam/mobile_sam_encoder.dyn.onnx",
    ),
    "weights/sam/mobile_sam/mobile_sam_decoder.onnx": (
        "weights/sam/mobile_sam/mobile_sam_decoder.static_bs1.onnx",
        "weights/sam/mobile_sam/mobile_sam_decoder.dyn.onnx",
    ),
    "weights/sam/mobile_sam/sm86/mobile_sam_amg_decoder.onnx": (
        "weights/sam/mobile_sam/sm86/mobile_sam_amg_decoder.static_bs1.onnx",
        "weights/sam/mobile_sam/sm86/mobile_sam_amg_decoder.dyn.onnx",
    ),
    "weights/sam/nano_sam/nano_sam_encoder.onnx": (
        "weights/sam/nano_sam/nano_sam_encoder.static_bs1.onnx",
        "weights/sam/nano_sam/nano_sam_encoder.dyn.onnx",
    ),
    "weights/sam/nano_sam/nano_sam_decoder.onnx": (
        "weights/sam/nano_sam/nano_sam_decoder.static_bs1.onnx",
        "weights/sam/nano_sam/nano_sam_decoder.dyn.onnx",
    ),
    "weights/sam/vit_l/sam_vit_l_decoder.onnx": (
        "weights/sam/vit_l/sam_vit_l_decoder.static_bs1.onnx",
        "weights/sam/vit_l/sam_vit_l_decoder.dyn.onnx",
    ),
    "weights/diffusion/ddpm/ddpm_unet_celeba-hq-128x128.onnx": (
        "weights/diffusion/ddpm/ddpm_unet_celeba-hq-128x128.static_bs1.onnx",
        "weights/diffusion/ddpm/ddpm_unet_celeba-hq-128x128.dyn.onnx",
    ),
    "weights/diffusion/ddpm/ddpm_unet_celeba-hq-256x256.onnx": (
        "weights/diffusion/ddpm/ddpm_unet_celeba-hq-256x256.static_bs1.onnx",
        "weights/diffusion/ddpm/ddpm_unet_celeba-hq-256x256.dyn.onnx",
    ),
    "weights/diffusion/ddpm/cls_cond_ddpm_netease_album_cover_128x128.onnx": (
        "weights/diffusion/ddpm/cls_cond_ddpm_netease_album_cover_128x128.static_bs1.onnx",
        "weights/diffusion/ddpm/cls_cond_ddpm_netease_album_cover_128x128.dyn.onnx",
    ),
    "weights/diffusion/ldm/latent_ddpm_celeba-hq.onnx": (
        "weights/diffusion/ldm/latent_ddpm_celeba-hq.static_bs1.onnx",
        "weights/diffusion/ldm/latent_ddpm_celeba-hq.dyn.onnx",
    ),
    "weights/diffusion/ldm/autoencoder_kl_decoder.onnx": (
        "weights/diffusion/ldm/autoencoder_kl_decoder.static_bs1.onnx",
        "weights/diffusion/ldm/autoencoder_kl_decoder.dyn.onnx",
    ),
}


def stamp(obj: dict) -> None:
    legacy = obj.get("onnx_path")
    if not legacy or legacy not in FILE_PATCH:
        return
    static_p, dyn_p = FILE_PATCH[legacy]
    obj["legacy_onnx_path"] = legacy
    obj["onnx_path"] = static_p
    obj["onnx_dyn_path"] = dyn_p
    src = obj.get("onnx_source") or {}
    src["kind"] = src.get("kind") or "preexisting_adopted"
    src["legacy"] = legacy
    src["notes"] = NOTE
    obj["onnx_source"] = src


def main() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data["description"] = (
        "Per product-id ONNX interchange source, IO contract, and license. "
        "Interchange files are dual: {stem}.static_bs1.onnx and {stem}.dyn.onnx. "
        "Original .onnx names stay on disk for product toml / trt_engines.json / already-hosted HF paths. "
        "Product conf/model/**/*.toml stay type=mnn or type=tensorrt except existing "
        "diffusion/MSOCRNET/SAM-decoder overlays. Do not reverse MNN/.model/.engine. "
        "HF upload is deferred until remaining missing/blocked graphs are closed or waived."
    )
    for model in data["models"]:
        stamp(model)
        for graph in model.get("graphs") or []:
            stamp(graph)
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated", PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
