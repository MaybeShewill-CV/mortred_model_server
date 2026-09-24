#!/usr/bin/env python3
"""Turn preexisting single ONNX files into stem.static_bs1.onnx + stem.dyn.onnx.

Original files stay in place (product toml / trt_engines.json / HF names).
If a .onnx.static_batch sidecar exists, it is the source (true batch=1 snapshot).
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    apply_reshape_batch_passthrough,
    io_summary,
    set_rank4_spatial_dynamic,
    sha256_of,
    write_pair_from_static,
)

SKIP_ALREADY_DUAL = True

# Product + hosted interchange graphs that still lack the dual naming.
JOBS: list[dict] = [
    {"rel": "weights/object_detection/centerface_detection/centerface.onnx", "spatial_dynamic": True},
    {"rel": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.onnx", "deep": True},
    {"rel": "weights/object_detection/yolov7/yolov7.onnx", "deep": True},
    {"rel": "weights/object_detection/yolov7/yolov7x.onnx", "deep": True},
    {"rel": "weights/object_detection/yolov8/yolov8s.onnx", "deep": True},
    {"rel": "weights/object_detection/yolov8/yolov8n.onnx", "deep": True},
    {"rel": "weights/object_detection/yolov8/yolov8l.onnx", "deep": True},
    {"rel": "weights/object_detection/yolov8/yolov8x.onnx", "deep": True},
    {"rel": "weights/scene_segmentation/hrnet/hrnetw48_ccd.onnx"},
    {"rel": "weights/scene_segmentation/msocrnet/msocrnet_fp16.onnx"},
    {"rel": "weights/feature_point/lightglue/extractor.onnx"},
    {"rel": "weights/feature_point/lightglue/matcher.onnx"},
    {"rel": "weights/sam/fastsam_s/FastSAM-s.onnx", "deep": True},
    {"rel": "weights/sam/fastsam_x/FastSAM-x.onnx", "deep": True},
    {"rel": "weights/sam/mobile_sam/mobile_sam_encoder.onnx"},
    {"rel": "weights/sam/mobile_sam/mobile_sam_decoder.onnx"},
    {"rel": "weights/sam/mobile_sam/sm86/mobile_sam_amg_decoder.onnx"},
    {"rel": "weights/sam/nano_sam/nano_sam_encoder.onnx"},
    {"rel": "weights/sam/nano_sam/nano_sam_decoder.onnx"},
    {"rel": "weights/sam/vit_l/sam_vit_l_decoder.onnx"},
    {"rel": "weights/diffusion/ddpm/ddpm_unet_celeba-hq-128x128.onnx"},
    {"rel": "weights/diffusion/ddpm/ddpm_unet_celeba-hq-256x256.onnx"},
    {"rel": "weights/diffusion/ddpm/cls_cond_ddpm_netease_album_cover_128x128.onnx"},
    {"rel": "weights/diffusion/ldm/latent_ddpm_celeba-hq.onnx"},
    {"rel": "weights/diffusion/ldm/autoencoder_kl_decoder.onnx"},
    {"rel": "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.onnx"},
    {"rel": "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.onnx"},
    {"rel": "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.onnx"},
    {"rel": "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.onnx"},
    {"rel": "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.onnx"},
]


def resolve_src(onnx_path: Path) -> Path:
    sidecar = Path(str(onnx_path) + ".static_batch")
    if sidecar.exists():
        return sidecar
    return onnx_path


def pair_paths(onnx_path: Path) -> tuple[Path, Path]:
    stem = onnx_path.name[: -len(".onnx")]
    return onnx_path.parent / f"{stem}.static_bs1.onnx", onnx_path.parent / f"{stem}.dyn.onnx"


def apply_spatial_dynamic(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    set_rank4_spatial_dynamic(model, h_name="height", w_name="width", out_h="out_height", out_w="out_width")
    onnx.save(model, str(path))


def run_job(job: dict) -> None:
    src_named = ROOT / job["rel"]
    if not src_named.exists():
        print(f"[skip] missing {job['rel']}")
        return
    static_dst, dyn_dst = pair_paths(src_named)
    if SKIP_ALREADY_DUAL and static_dst.exists() and dyn_dst.exists():
        print(f"[skip] already dual {static_dst.name}")
        return
    src = resolve_src(src_named)
    print(f"\n=== {job['rel']}  src={src.name} ===", flush=True)
    write_pair_from_static(src, static_dst, dyn_dst)
    if job.get("spatial_dynamic"):
        apply_spatial_dynamic(static_dst)
        apply_spatial_dynamic(dyn_dst)
        print(f"[spatial] static {io_summary(static_dst)}")
        print(f"[spatial] dyn    {io_summary(dyn_dst)}")
    if job.get("deep"):
        n = apply_reshape_batch_passthrough(dyn_dst)
        print(f"[deep] {dyn_dst.name}: {n} Reshape dim0 1->0")
        print(f"       sha dyn={sha256_of(dyn_dst)[:16]}")


def main() -> int:
    only = sys.argv[1:]
    jobs = JOBS
    if only:
        jobs = [j for j in JOBS if any(tok in j["rel"] for tok in only)]
    for job in jobs:
        run_job(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
