#!/usr/bin/env python3
"""static_bs1 product catalog: one lowercase id per engine file.

Source of truth is conf/product_static_bs1.json, emitted by --emit.
Gateway paths are /v1/models/<id>/infer. Old uppercase ids are not aliases.
libface is one catalog id at 640x480. libface_320 is not a product size.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "conf" / "product_static_bs1.json"

BUS = "demo_data/model_test_input/object_detection/bus.jpg"
FACE = "demo_data/model_test_input/object_detection/face_wo_mask.jpg"
CLS = "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG"
MATT = "demo_data/model_test_input/matting/matting_test.jpg"
HUMAN = "demo_data/model_test_input/scene_segmentation/human_image.jpg"
CITY = "demo_data/model_test_input/scene_segmentation/cityscapes_test.png"
DERAIN = "demo_data/model_test_input/enhancement/derain/test_1.png"
LOL = "demo_data/model_test_input/enhancement/low_light/lol_test_1.png"
ESR = "demo_data/model_test_input/enhancement/real_esr/test.JPG"
SP = "demo_data/model_test_input/feature_point/test.png"
DEPTH = "demo_data/model_test_input/mono_depth_estimation/0000000005.png"
OCR = "demo_data/model_test_input/ocr/railway_ticket.png"


def _m(
    id: str,
    family: str,
    fp: str,
    port: int,
    server_section: str,
    area: str,
    onnx: str,
    engine: str,
    server_config: str,
    model_config: str,
    demo: str,
    *,
    layout: str = "nchw",
    profile: str | None = None,
    timeout_ms: int = 1000,
    replaces: str | None = None,
) -> dict:
    return {
        "id": id,
        "family": family,
        "fp": fp,
        "port": port,
        "server_section": server_section,
        "server_uri": f"/mortred_ai_server_v1/{area}/{id}",
        "onnx": onnx,
        "engine": engine,
        "profile": profile,
        "layout": layout,
        "demo": demo,
        "server_config": server_config,
        "model_config": model_config,
        "timeout_ms": timeout_ms,
        "replaces": replaces,
    }


def model_rows() -> list[dict]:
    """The 39 product ids. Engine suffix matches fp. libface is a single row."""
    y8 = "weights/object_detection/yolov8"
    return [
        _m("yolov8n", "yolov8", "fp16", 9100, "YOLOV8_DETECTION_SERVER", "obj_detection",
           f"{y8}/yolov8n.static_bs1.onnx", f"{y8}/yolov8n.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov8/yolov8n_server_config.toml",
           "conf/model/object_detection/yolov8/yolov8n.toml", BUS),
        _m("yolov8s", "yolov8", "fp16", 9056, "YOLOV8_DETECTION_SERVER", "obj_detection",
           f"{y8}/yolov8s.static_bs1.onnx", f"{y8}/yolov8s.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov8/yolov8s_server_config.toml",
           "conf/model/object_detection/yolov8/yolov8s.toml", BUS, replaces="YOLOV8"),
        _m("yolov8l", "yolov8", "fp16", 9101, "YOLOV8_DETECTION_SERVER", "obj_detection",
           f"{y8}/yolov8l.static_bs1.onnx", f"{y8}/yolov8l.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov8/yolov8l_server_config.toml",
           "conf/model/object_detection/yolov8/yolov8l.toml", BUS),
        _m("yolov8x", "yolov8", "fp16", 9102, "YOLOV8_DETECTION_SERVER", "obj_detection",
           f"{y8}/yolov8x.static_bs1.onnx", f"{y8}/yolov8x.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov8/yolov8x_server_config.toml",
           "conf/model/object_detection/yolov8/yolov8x.toml", BUS),
        _m("yolov6s", "yolov6", "fp16", 9054, "YOLOV6_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/yolov6/yolov6s.static_bs1.onnx",
           "weights/object_detection/yolov6/yolov6s.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov6/yolov6s_server_config.toml",
           "conf/model/object_detection/yolov6/yolov6s.toml", BUS, replaces="YOLOV6"),
        _m("yolov5l", "yolov5", "fp16", 9053, "YOLOV5_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/yolov5/yolov5l.static_bs1.onnx",
           "weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov5/yolov5l_server_config.toml",
           "conf/model/object_detection/yolov5/yolov5l.toml", BUS, replaces="YOLOV5"),
        _m("yolov7", "yolov7", "fp16", 9055, "YOLOV7_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/yolov7/yolov7.static_bs1.onnx",
           "weights/object_detection/yolov7/yolov7.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov7/yolov7_server_config.toml",
           "conf/model/object_detection/yolov7/yolov7.toml", BUS, replaces="YOLOV7"),
        _m("yolov7x", "yolov7", "fp16", 9103, "YOLOV7_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/yolov7/yolov7x.static_bs1.onnx",
           "weights/object_detection/yolov7/yolov7x.engine.static_bs1.fp16",
           "conf/server/object_detection/yolov7/yolov7x_server_config.toml",
           "conf/model/object_detection/yolov7/yolov7x.toml", BUS),
        _m("nanodet_1x5", "nanodet", "fp16", 9052, "NANODET_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/nanodet/nanodet_plus_m_1x5.static_bs1.onnx",
           "weights/object_detection/nanodet/nanodet_plus_m_1x5.engine.static_bs1.fp16",
           "conf/server/object_detection/nano_det/nanodet_1x5_server_config.toml",
           "conf/model/object_detection/nano_det/nanodet_1x5.toml", BUS, replaces="NANODET"),
        _m("nanodet_416", "nanodet", "fp16", 9104, "NANODET_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/nanodet/nanodet_plus_m_416.static_bs1.onnx",
           "weights/object_detection/nanodet/nanodet_plus_m_416.engine.static_bs1.fp16",
           "conf/server/object_detection/nano_det/nanodet_416_server_config.toml",
           "conf/model/object_detection/nano_det/nanodet_416.toml", BUS),
        _m("libface", "libface", "fp16", 9051, "LIBFACE_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/libfacedetection/face_detection_yunet_2026may.static_bs1.onnx",
           "weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16",
           "conf/server/object_detection/libface_det/libface_server_config.toml",
           "conf/model/object_detection/libfacedetection/libface.toml", FACE,
           profile="conf/trt_profiles/set_a_libface.json", replaces="LIBFACE"),
        _m("centerface", "centerface", "fp16", 9050, "CENTER_FACE_DETECTION_SERVER", "obj_detection",
           "weights/object_detection/centerface_detection/centerface.static_bs1.onnx",
           "weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16",
           "conf/server/object_detection/center_face_det/centerface_server_config.toml",
           "conf/model/object_detection/centerface/centerface.toml", FACE,
           profile="conf/trt_profiles/set_a_centerface.json", replaces="CENTER_FACE"),
        _m("mobilenetv2", "mobilenetv2", "fp32", 9002, "MOBILENETV2_CLASSIFICATION_SERVER", "classification",
           "weights/classification/mobilenetv2/mobilenetv2.static_bs1.onnx",
           "weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32",
           "conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml",
           "conf/model/classification/mobilenetv2/mobilenetv2.toml", CLS,
           layout="nhwc", timeout_ms=500, replaces="MOBILENETV2"),
        _m("resnet", "resnet", "fp32", 9003, "RESNET_CLASSIFICATION_SERVER", "classification",
           "weights/classification/resnet/resnet.static_bs1.onnx",
           "weights/classification/resnet/resnet.engine.static_bs1.fp32",
           "conf/server/classification/resnet/resnet_server_config.toml",
           "conf/model/classification/resnet/resnet.toml", CLS,
           layout="nhwc", timeout_ms=500, replaces="RESNET"),
        _m("densenet", "densenet", "fp32", 9001, "DENSENET_CLASSIFICATION_SERVER", "classification",
           "weights/classification/densenet/densenet.static_bs1.onnx",
           "weights/classification/densenet/densenet.engine.static_bs1.fp32",
           "conf/server/classification/densenet/densenet_server_config.toml",
           "conf/model/classification/densenet/densenet.toml", CLS,
           layout="nhwc", timeout_ms=500, replaces="DENSENET"),
        _m("superpoint", "superpoint", "fp16", 9020, "SUPERPOINT_FP_SERVER", "feature_point",
           "weights/feature_point/superpoint/superpoint_120x160.static_bs1.onnx",
           "weights/feature_point/superpoint/superpoint_120x160.engine.static_bs1.fp16",
           "conf/server/feature_point/superpoint/superpoint_server_config.toml",
           "conf/model/feature_point/superpoint/superpoint.toml", SP, replaces="SUPERPOINT"),
        _m("dbnet", "dbnet", "fp16", 9060, "DBNET_SERVER", "ocr",
           "weights/ocr/db_text_detector/db_model_large.static_bs1.onnx",
           "weights/ocr/db_text_detector/db_model_large.engine.static_bs1.fp16",
           "conf/server/ocr/dbnet/dbnet_server_config.toml",
           "conf/model/ocr/db_text_detector/dbnet.toml", OCR, replaces="DBNET"),
        _m("pphuman_mobile", "pphuman", "fp16", 9072, "PPHUMAN_SEG_SERVER", "scene_segmentation",
           "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.static_bs1.onnx",
           "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.engine.static_bs1.fp16",
           "conf/server/scene_segmentation/pphuman_seg/pphuman_mobile_server_config.toml",
           "conf/model/scene_segmentation/pphuman/pphuman_mobile.toml", HUMAN, replaces="PPHUMAN_SEG"),
        _m("pphuman_lite", "pphuman", "fp16", 9105, "PPHUMAN_SEG_SERVER", "scene_segmentation",
           "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.static_bs1.onnx",
           "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.engine.static_bs1.fp16",
           "conf/server/scene_segmentation/pphuman_seg/pphuman_lite_server_config.toml",
           "conf/model/scene_segmentation/pphuman/pphuman_lite.toml", HUMAN),
        _m("pphuman_server", "pphuman", "fp16", 9106, "PPHUMAN_SEG_SERVER", "scene_segmentation",
           "weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.static_bs1.onnx",
           "weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.engine.static_bs1.fp16",
           "conf/server/scene_segmentation/pphuman_seg/pphuman_server_server_config.toml",
           "conf/model/scene_segmentation/pphuman/pphuman_server.toml", HUMAN),
        _m("bisenetv2", "bisenetv2", "fp32", 9070, "BISENETV2_SERVER", "scene_segmentation",
           "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.static_bs1.onnx",
           "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.engine.static_bs1.fp32",
           "conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml",
           "conf/model/scene_segmentation/bisenetv2/bisenetv2.toml", CITY,
           layout="nhwc", replaces="BISENETV2"),
        _m("hrnet", "hrnet", "fp16", 9071, "HRNET_SERVER", "scene_segmentation",
           "weights/scene_segmentation/hrnet/hrnetw48_ccd.static_bs1.onnx",
           "weights/scene_segmentation/hrnet/hrnetw48_ccd.engine.static_bs1.fp16",
           "conf/server/scene_segmentation/hrnet/hrnet_server_config.toml",
           "conf/model/scene_segmentation/hrnet/hrnet.toml", CITY,
           timeout_ms=5000, replaces="HRNET"),
        _m("msocrnet", "msocrnet", "fp16", 9107, "MSOCRNET_SERVER", "scene_segmentation",
           "weights/scene_segmentation/msocrnet/msocrnet_fp16.static_bs1.onnx",
           "weights/scene_segmentation/msocrnet/msocrnet_fp16.engine.static_bs1.fp16",
           "conf/server/scene_segmentation/msocrnet/msocrnet_server_config.toml",
           "conf/model/scene_segmentation/msocrnet/msocrnet.toml", OCR, timeout_ms=5000),
        _m("modnet", "modnet", "fp16", 9030, "MODNET_SERVER", "matting",
           "weights/matting/modnet/modnet_hrnet_w18.static_bs1.onnx",
           "weights/matting/modnet/modnet_hrnet_w18.engine.static_bs1.fp16",
           "conf/server/matting/modnet/modnet_server_config.toml",
           "conf/model/matting/modnet/modnet.toml", MATT, replaces="MODNET"),
        _m("ppmatting_512", "ppmatting", "fp16", 9031, "PP_MATTING_SERVER", "matting",
           "weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.static_bs1.onnx",
           "weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.engine.static_bs1.fp16",
           "conf/server/matting/ppmatting/ppmatting_512_server_config.toml",
           "conf/model/matting/ppmatting/ppmatting_512.toml", MATT, replaces="PP_MATTING"),
        _m("ppmatting_1024", "ppmatting", "fp16", 9108, "PP_MATTING_SERVER", "matting",
           "weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.static_bs1.onnx",
           "weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.engine.static_bs1.fp16",
           "conf/server/matting/ppmatting/ppmatting_1024_server_config.toml",
           "conf/model/matting/ppmatting/ppmatting_1024.toml", MATT),
        _m("ppmatting_resnet34", "ppmatting", "fp16", 9109, "PP_MATTING_SERVER", "matting",
           "weights/matting/ppmatting/pp_humanmatting-resnet34_vd.static_bs1.onnx",
           "weights/matting/ppmatting/pp_humanmatting-resnet34_vd.engine.static_bs1.fp16",
           "conf/server/matting/ppmatting/ppmatting_resnet34_server_config.toml",
           "conf/model/matting/ppmatting/ppmatting_resnet34.toml", MATT),
        _m("ppmatting_v2", "ppmatting", "fp16", 9110, "PP_MATTING_SERVER", "matting",
           "weights/matting/ppmatting/ppmattingv2_stdc1_human_512.static_bs1.onnx",
           "weights/matting/ppmatting/ppmattingv2_stdc1_human_512.engine.static_bs1.fp16",
           "conf/server/matting/ppmatting/ppmatting_v2_server_config.toml",
           "conf/model/matting/ppmatting/ppmatting_v2.toml", MATT),
        _m("attentive_gan", "attentive_gan", "fp32", 9010, "ATTENTIVE_GAN_DERAIN_SERVER", "enhancement",
           "weights/enhancement/attentive_gan_derain/attentive_gan_derain.static_bs1.onnx",
           "weights/enhancement/attentive_gan_derain/attentive_gan_derain.engine.static_bs1.fp32",
           "conf/server/enhancement/attentive_gan_derain/attentive_gan_server_config.toml",
           "conf/model/enhancement/attentive_gan_derain/attentive_gan.toml", DERAIN,
           layout="nhwc", replaces="ATTENTIVE_GAN_DERAIN"),
        _m("enlightengan", "enlightengan", "fp16", 9011, "ENLIGHTEN_GAN_SERVER", "enhancement",
           "weights/enhancement/enlighten_gan/enlightengan.static_bs1.onnx",
           "weights/enhancement/enlighten_gan/enlightengan.engine.static_bs1.fp16",
           "conf/server/enhancement/enlightengan/enlightengan_server_config.toml",
           "conf/model/enhancement/enlighten_gan/enlightengan.toml", LOL,
           profile="conf/trt_profiles/set_a_enlightengan.json", replaces="ENLIGHTEN_GAN"),
        _m("realesrgan", "realesrgan", "fp32", 9012, "REAL_ESRGAN_SERVER", "enhancement",
           "weights/enhancement/real_esrgan/realesr-general-x4v3.static_bs1.onnx",
           "weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32",
           "conf/server/enhancement/real_esrgan/realesrgan_server_config.toml",
           "conf/model/enhancement/real_esrgan/realesrgan.toml", ESR,
           layout="nhwc", profile="conf/trt_profiles/set_a_realesrgan.json", replaces="REAL_ESRGAN"),
        _m("dinov2_vits14", "dinov2", "fp16", 9111, "DINOV2_FEATURE_EMBEDDING_SERVER", "feature_embedding",
           "weights/classification/dinov2/dinov2_vits14_pretrain.static_bs1.onnx",
           "weights/classification/dinov2/dinov2_vits14_pretrain.engine.static_bs1.fp16",
           "conf/server/feature_embedding/dinov2/dinov2_vits14_server_config.toml",
           "conf/model/feature_embedding/dinov2/dinov2_vits14.toml", CLS, timeout_ms=500),
        _m("dinov2_vitb14", "dinov2", "fp16", 9080, "DINOV2_FEATURE_EMBEDDING_SERVER", "feature_embedding",
           "weights/classification/dinov2/dinov2_vitb14_pretrain.static_bs1.onnx",
           "weights/classification/dinov2/dinov2_vitb14_pretrain.engine.static_bs1.fp16",
           "conf/server/feature_embedding/dinov2/dinov2_vitb14_server_config.toml",
           "conf/model/feature_embedding/dinov2/dinov2_vitb14.toml", CLS,
           timeout_ms=500, replaces="DINOV2"),
        _m("dinov2_vitl14", "dinov2", "fp16", 9112, "DINOV2_FEATURE_EMBEDDING_SERVER", "feature_embedding",
           "weights/classification/dinov2/dinov2_vitl14_pretrain.static_bs1.onnx",
           "weights/classification/dinov2/dinov2_vitl14_pretrain.engine.static_bs1.fp16",
           "conf/server/feature_embedding/dinov2/dinov2_vitl14_server_config.toml",
           "conf/model/feature_embedding/dinov2/dinov2_vitl14.toml", CLS, timeout_ms=500),
        _m("depth_vits14", "depth_anything", "fp16", 9113, "DEPTH_ANYTHING_ESTIMATION_SERVER", "mono_depth_estimation",
           "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.static_bs1.onnx",
           "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.engine.static_bs1.fp16",
           "conf/server/mono_depth_estimation/depth_anything/depth_vits14_server_config.toml",
           "conf/model/mono_depth_estimation/depth_anything/depth_vits14.toml", DEPTH, timeout_ms=5000),
        _m("depth_vitb14", "depth_anything", "fp16", 9040, "DEPTH_ANYTHING_ESTIMATION_SERVER", "mono_depth_estimation",
           "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.static_bs1.onnx",
           "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.engine.static_bs1.fp16",
           "conf/server/mono_depth_estimation/depth_anything/depth_vitb14_server_config.toml",
           "conf/model/mono_depth_estimation/depth_anything/depth_vitb14.toml", DEPTH,
           timeout_ms=5000, replaces="DEPTH_ANYTHING"),
        _m("depth_vitl14", "depth_anything", "fp16", 9114, "DEPTH_ANYTHING_ESTIMATION_SERVER", "mono_depth_estimation",
           "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.static_bs1.onnx",
           "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.engine.static_bs1.fp16",
           "conf/server/mono_depth_estimation/depth_anything/depth_vitl14_server_config.toml",
           "conf/model/mono_depth_estimation/depth_anything/depth_vitl14.toml", DEPTH, timeout_ms=5000),
        _m("metric3d_512", "metric3d", "fp16", 9041, "METRIC3D_ESTIMATION_SERVER", "mono_depth_estimation",
           "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.static_bs1.onnx",
           "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16",
           "conf/server/mono_depth_estimation/metric3d/metric3d_512_server_config.toml",
           "conf/model/mono_depth_estimation/metric3d/metric3d_512.toml", DEPTH,
           timeout_ms=10000, replaces="METRIC3D"),
        _m("metric3d_1088", "metric3d", "fp16", 9115, "METRIC3D_ESTIMATION_SERVER", "mono_depth_estimation",
           "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.static_bs1.onnx",
           "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.engine.static_bs1.fp16",
           "conf/server/mono_depth_estimation/metric3d/metric3d_1088_server_config.toml",
           "conf/model/mono_depth_estimation/metric3d/metric3d_1088.toml", DEPTH, timeout_ms=10000),
    ]


def document() -> dict:
    rows = model_rows()
    return {
        "description": (
            "One lowercase catalog id per static_bs1 fp16/fp32 engine. "
            "Gateway: POST /v1/models/<id>/infer. No uppercase alias. "
            "libface is the only YuNet id (320 and 640 share one engine)."
        ),
        "gateway_infer": "/v1/models/{id}/infer",
        "models": rows,
    }


def validate_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 39:
        errors.append(f"expected 39 models, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate id")
    if any(row["id"] != row["id"].lower() or row["id"] == "" for row in rows):
        errors.append("id must be lowercase")
    if "libface_320" in ids or "libface_640" in ids:
        errors.append("libface_320/libface_640 must not be catalog ids")
    if ids.count("libface") != 1:
        errors.append("libface must appear once")
    ports = [row["port"] for row in rows]
    if len(set(ports)) != len(ports):
        errors.append("duplicate port")
    uris = [row["server_uri"] for row in rows]
    if len(set(uris)) != len(uris):
        errors.append("duplicate server_uri")
    engines = [row["engine"] for row in rows]
    if len(set(engines)) != len(engines):
        errors.append("duplicate engine path")
    for row in rows:
        suffix = ".engine.static_bs1." + row["fp"]
        if not row["engine"].endswith(suffix):
            errors.append(f"{row['id']}: engine suffix must be {suffix}")
        if not row["onnx"].endswith(".static_bs1.onnx"):
            errors.append(f"{row['id']}: onnx must be a static_bs1 file")
        if row["fp"] not in ("fp16", "fp32"):
            errors.append(f"{row['id']}: fp must be fp16 or fp32")
    return errors


def load_table(path: Path = TABLE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pack_ids(pack_path: Path) -> list[str]:
    ids: list[str] = []
    for line in pack_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("[pack.") and text.endswith("]"):
            pid = text[6:-1].strip()
            if pid:
                ids.append(pid)
    return ids


def check_pack(pack_path: Path, rows: list[dict]) -> list[str]:
    allowed = {row["id"] for row in rows}
    errors: list[str] = []
    found = pack_ids(pack_path)
    if not found:
        errors.append(f"pack has no [pack.<id>] tables: {pack_path}")
    for pid in found:
        if pid not in allowed:
            errors.append(
                f"pack id {pid!r} is not a static_bs1 catalog id "
                f"(gateway path is /v1/models/<lowercase>/infer)"
            )
    return errors


def trt_manifest(rows: list[dict]) -> dict:
    engines = []
    for row in rows:
        engines.append({
            "model": row["id"],
            "onnx": row["onnx"],
            "engine": row["engine"],
            "fp": 0 if row["fp"] == "fp32" else 1,
            "profile": row["profile"],
        })
    return {
        "_description": "Product static_bs1 TensorRT manifest. Exact engine paths only.",
        "engines": engines,
    }


def bench_source(row: dict) -> Path | None:
    if row["id"] == "libface":
        return ROOT / "conf/bench/set_a/libface_640.toml"
    if row["id"] == "msocrnet":
        return None
    return ROOT / "conf/bench/set_a" / f"{row['id']}.toml"


def render_model_toml(row: dict) -> str:
    engine = "../" + row["engine"]
    source = bench_source(row)
    if source is None:
        return (
            f"# Product TensorRT static_bs1 config. catalog id={row['id']}\n"
            "[MSOCRNET]\n"
            "[MSOCRNET.backend]\n"
            'type = "tensorrt"\n'
            f'model_file_path = "{engine}"\n'
            'device = "gpu"\n'
            "threads = 4\n"
            f'input_layout = "{row["layout"]}"\n'
            "\n"
            "[MSOCRNET.params]\n"
            "model_input_image_size = [1024, 2048]\n"
            'image_decode_mode = "budget"\n'
            "image_decode_budget_upscale = 1.25\n"
            'image_decode_backend = "auto"\n'
        )
    if not source.is_file():
        raise SystemExit(f"missing bench source for {row['id']}: {source}")
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^# Bench overlay:.*\n", "", text)
    text = re.sub(r"(?m)^# id=.*\n", "", text)
    text = re.sub(
        r'model_file_path\s*=\s*"[^"]+"',
        f'model_file_path = "{engine}"',
        text,
        count=1,
    )
    if re.search(r'input_layout\s*=\s*"', text):
        text = re.sub(
            r'input_layout\s*=\s*"[^"]+"',
            f'input_layout = "{row["layout"]}"',
            text,
            count=1,
        )
    else:
        text = text.replace(
            'device = "gpu"\n',
            f'device = "gpu"\ninput_layout = "{row["layout"]}"\n',
            1,
        )
    text = text.lstrip("\n")
    return f"# Product TensorRT static_bs1 config. catalog id={row['id']}\n" + text


def _set_device_cpu(text: str) -> str:
    return re.sub(r'(device\s*=\s*")gpu(")', r"\1cpu\2", text, count=1)


def preserve_cpu_golden_configs() -> None:
    """CPU goldens cannot load TensorRT. Keep MNN copies before the gpu rewrite."""
    copies = [
        (
            ROOT / "conf/model/classification/mobilenetv2/mobilenetv2.toml",
            ROOT / "conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml",
        ),
        (
            ROOT / "conf/model/classification/resnet/resnet.toml",
            ROOT / "conf/model/classification/resnet/resnet_cpu.toml",
        ),
        (
            ROOT / "conf/model/object_detection/yolov7/yolov7.toml",
            ROOT / "conf/ci/yolov7_cpu_golden.toml",
        ),
        (
            ROOT / "conf/model/object_detection/centerface/centerface.toml",
            ROOT / "conf/ci/centerface_cpu_golden.toml",
        ),
    ]
    for src, dst in copies:
        if not src.is_file():
            raise SystemExit(f"missing source config to preserve: {src}")
        if dst.is_file() and 'type = "mnn"' in dst.read_text(encoding="utf-8"):
            continue
        body = src.read_text(encoding="utf-8")
        if 'type = "mnn"' not in body:
            raise SystemExit(f"{src} is not MNN; refuse to overwrite {dst}")
        if dst.name.endswith("_cpu.toml"):
            body = _set_device_cpu(body)
            if 'device = "cpu"' not in body:
                raise SystemExit(f"failed to set device=cpu in {dst}")
        header = (
            "# CPU golden / cpu-profile config. Not the gpu TensorRT product file.\n"
        )
        if not body.startswith("# CPU golden"):
            body = header + body
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(body, encoding="utf-8")


def point_server_at_model(server_path: Path, model_config: str) -> None:
    text = server_path.read_text(encoding="utf-8")
    rel = "../" + model_config
    new, n = re.subn(
        r'model_config_file_path\s*=\s*"[^"]+"',
        f'model_config_file_path="{rel}"',
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"no model_config_file_path in {server_path}")
    server_path.write_text(new, encoding="utf-8")


def apply_model_configs(rows: list[dict]) -> None:
    preserve_cpu_golden_configs()
    for row in rows:
        dest = ROOT / row["model_config"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(render_model_toml(row), encoding="utf-8")
        point_server_at_model(ROOT / row["server_config"], row["model_config"])
    point_server_at_model(
        ROOT / "conf/server/classification/mobilenetv2/mobilenetv2_server_cpu_config.toml",
        "conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml",
    )
    point_server_at_model(
        ROOT / "conf/server/classification/resnet/resnet_server_cpu_config.toml",
        "conf/model/classification/resnet/resnet_cpu.toml",
    )


def _backend_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    in_backend = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_backend = stripped.endswith(".backend]")
            continue
        if not in_backend or "=" not in stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        fields[key.strip()] = value.strip().strip('"')
    return fields


RETIRED_CATALOG_IDS = {
    "YOLOV8", "YOLOV8S", "YOLOV5", "YOLOV6", "YOLOV7", "NANODET", "LIBFACE",
    "CENTER_FACE", "PPHUMAN_SEG", "PP_MATTING", "DINOV2", "DEPTH_ANYTHING",
    "METRIC3D", "MOBILENETV2", "RESNET", "DENSENET", "SUPERPOINT", "DBNET",
    "BISENETV2", "MODNET", "ATTENTIVE_GAN_DERAIN", "ENLIGHTEN_GAN",
    "REAL_ESRGAN", "HRNET",
}


def check_set_a_manifest(rows: list[dict]) -> list[str]:
    """conf/trt_engines.set_a.json must match the product-table trtexec manifest."""
    path = ROOT / "conf" / "trt_engines.set_a.json"
    if not path.is_file():
        return [f"missing {path.relative_to(ROOT)}"]
    try:
        on_disk = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"conf/trt_engines.set_a.json is not valid JSON: {exc}"]
    want = {
        (e["model"], e["onnx"], e["engine"], e["fp"], e.get("profile"))
        for e in trt_manifest(rows)["engines"]
    }
    got = {
        (
            str(e.get("model") or "").strip(),
            str(e.get("onnx") or "").strip(),
            str(e.get("engine") or "").strip(),
            e.get("fp"),
            e.get("profile"),
        )
        for e in on_disk.get("engines", [])
    }
    errors: list[str] = []
    if want != got:
        only_want = sorted(want - got)
        only_got = sorted(got - want)
        if only_want:
            sample = ", ".join(f"{m}/{eng}" for m, _o, eng, _fp, _p in only_want[:5])
            errors.append(
                "conf/trt_engines.set_a.json missing product manifest rows: "
                + sample
                + ("..." if len(only_want) > 5 else "")
            )
        if only_got:
            sample = ", ".join(f"{m}/{eng}" for m, _o, eng, _fp, _p in only_got[:5])
            errors.append(
                "conf/trt_engines.set_a.json has rows not in product manifest: "
                + sample
                + ("..." if len(only_got) > 5 else "")
            )
    return errors


def validate_product_files(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    nhwc = {row["id"] for row in rows if row["layout"] == "nhwc"}
    for row in rows:
        path = ROOT / row["model_config"]
        if not path.is_file():
            errors.append(f"missing model config {path}")
            continue
        text = path.read_text(encoding="utf-8")
        fields = _backend_fields(text)
        if fields.get("type") != "tensorrt":
            errors.append(f"{row['id']}: backend type is {fields.get('type')!r}")
        engine = "../" + row["engine"]
        if fields.get("model_file_path") != engine:
            errors.append(f"{row['id']}: engine path {fields.get('model_file_path')!r}")
        if fields.get("input_layout") != row["layout"]:
            errors.append(f"{row['id']}: layout {fields.get('input_layout')!r}")
        if row["id"] in nhwc and row["fp"] != "fp32":
            errors.append(f"{row['id']}: nhwc product rows are the fp32 six")
        server = (ROOT / row["server_config"]).read_text(encoding="utf-8")
        if f'model="{row["id"]}"' not in server:
            errors.append(f"{row['id']}: server model id mismatch")
        if f'../{row["model_config"]}' not in server:
            errors.append(f"{row['id']}: server does not point at {row['model_config']}")
        if row["id"] == "libface" and "kps_32" not in text:
            errors.append("libface missing 12-head output_names")
        if row["id"] == "libface" and "model_input_image_size = [480, 640]" not in text:
            errors.append("libface params are not 640x480")
    fp32_ids = {row["id"] for row in rows if row["fp"] == "fp32"}
    if fp32_ids != nhwc:
        errors.append(f"fp32 ids {sorted(fp32_ids)} must be the nhwc set {sorted(nhwc)}")
    joined = "\n".join((ROOT / row["model_config"]).read_text(encoding="utf-8") for row in rows)
    for stem in (
        "hrnetw48_ccd.engine.static_bs1.fp16",
        "depth_anything_vits14.engine.static_bs1.fp16",
        "depth_anything_vitb14.engine.static_bs1.fp16",
        "depth_anything_vitl14.engine.static_bs1.fp16",
        "metric3d_750k_512x1088.engine.static_bs1.fp16",
        "metric3d_750k_1088x1920.engine.static_bs1.fp16",
        "msocrnet_fp16.engine.static_bs1.fp16",
    ):
        if stem not in joined:
            errors.append(f"missing static_bs1 engine reference {stem}")
    if "hrnetw48_ccd_fp32.engine" in joined:
        errors.append("hrnet still references the old fp32 engine")
    cpu_mv = ROOT / "conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml"
    cpu_rn = ROOT / "conf/model/classification/resnet/resnet_cpu.toml"
    for path in (cpu_mv, cpu_rn):
        body = path.read_text(encoding="utf-8") if path.is_file() else ""
        if 'type = "mnn"' not in body or 'device = "cpu"' not in body:
            errors.append(f"{path}: cpu profile config must stay MNN on cpu")
    for cfg in (ROOT / "conf/server").rglob("*.toml"):
        for line in cfg.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("model="):
                value = stripped.split("=", 1)[1].strip().strip('"')
                if value in RETIRED_CATALOG_IDS:
                    errors.append(f"{cfg}: retired catalog id {value}")
    return errors


def server_file_text(row: dict) -> str:
    model_rel = "../" + row["model_config"]
    return (
        f"[{row['server_section']}]\n"
        f"model=\"{row['id']}\"\n"
        f"port={row['port']}\n"
        "host=\"localhost\"\n"
        "max_connections=500\n"
        "peer_resp_timeout=15\n"
        "request_size_limit=64\n"
        "compute_threads=-1\n"
        "handler_threads=50\n"
        "worker_nums=1\n"
        f"model_run_timeout={row['timeout_ms']}\n"
        "max_queue_depth=32\n"
        "max_batch_size=1\n"
        f"server_uri=\"{row['server_uri']}\"\n"
        "server_exe=\"mortred-model-server.out\"\n"
        "\n"
        f"[{row['id']}]\n"
        f"model_config_file_path=\"{model_rel}\"\n"
    )


def _replace_identity(text: str, row: dict) -> str:
    old = row.get("replaces") or ""
    lines = []
    replaced_model = False
    replaced_uri = False
    replaced_table = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if old and stripped == f'model="{old}"':
            newline = "\n" if line.endswith("\n") else ""
            lines.append(f'model="{row["id"]}"{newline}')
            replaced_model = True
            continue
        if stripped.startswith("server_uri="):
            newline = "\n" if line.endswith("\n") else ""
            lines.append(f'server_uri="{row["server_uri"]}"{newline}')
            replaced_uri = True
            continue
        if old and stripped == f"[{old}]":
            newline = "\n" if line.endswith("\n") else ""
            lines.append(f"[{row['id']}]{newline}")
            replaced_table = True
            continue
        lines.append(line)
    if old and not (replaced_model and replaced_uri and replaced_table):
        raise SystemExit(
            f"{row['id']}: identity rewrite incomplete "
            f"(model={replaced_model} uri={replaced_uri} table={replaced_table})"
        )
    body = "".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    return body


def apply_servers(rows: list[dict]) -> None:
    """Rename existing server identity and write sibling server files."""
    extra_cpu = {
        "mobilenetv2": ROOT / "conf/server/classification/mobilenetv2/mobilenetv2_server_cpu_config.toml",
        "resnet": ROOT / "conf/server/classification/resnet/resnet_server_cpu_config.toml",
    }
    for row in rows:
        path = ROOT / row["server_config"]
        if row.get("replaces"):
            if not path.is_file():
                raise SystemExit(f"missing server config to rename: {path}")
            path.write_text(_replace_identity(path.read_text(encoding="utf-8"), row), encoding="utf-8")
            cpu = extra_cpu.get(row["id"])
            if cpu is not None:
                cpu.write_text(_replace_identity(cpu.read_text(encoding="utf-8"), row), encoding="utf-8")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(server_file_text(row), encoding="utf-8")


def convert_filters_for_pack(pack: Path, project_root: Path) -> list[str]:
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import pack_trt

    rows = load_table()["models"]
    known = {row["engine"] for row in rows}
    engines = pack_trt.pack_trt_engines(pack, project_root)
    filters: list[str] = []
    errors: list[str] = []
    for pid, engine in engines:
        try:
            rel = engine.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            rel = engine.as_posix()
        if rel not in known:
            errors.append(f"{pid}: {rel} is not an exact static_bs1 engine path")
        elif rel not in filters:
            filters.append(rel)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        raise SystemExit(1)
    return filters


def _read_product_models() -> list[dict]:
    """Read conf/product_static_bs1.json. Does not emit or rewrite the table."""
    try:
        table = load_table()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {TABLE_PATH}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    models = table.get("models")
    if not isinstance(models, list):
        print("product_static_bs1.json models is not a list", file=sys.stderr)
        raise SystemExit(1)
    return models


def lookup_product_engine(catalog_id: str) -> int:
    """Print the JSON engine path for one id. Read-only."""
    for row in _read_product_models():
        if row.get("id") == catalog_id:
            engine = row.get("engine") or ""
            if not engine:
                print(f"{catalog_id}: empty engine", file=sys.stderr)
                return 1
            print(engine)
            return 0
    print(f"unknown product id: {catalog_id}", file=sys.stderr)
    return 1


def print_missing_product_engines() -> int:
    """Print id<TAB>engine for missing or zero-byte product engines. Read-only."""
    for row in _read_product_models():
        catalog_id = str(row.get("id") or "")
        engine = str(row.get("engine") or "")
        if not catalog_id or not engine:
            continue
        path = ROOT / engine
        try:
            missing = (not path.is_file()) or path.stat().st_size == 0
        except OSError:
            missing = True
        if missing:
            print(f"{catalog_id}\t{engine}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write conf/product_static_bs1.json")
    parser.add_argument("--check", action="store_true", help="validate the committed table")
    parser.add_argument("--apply-servers", action="store_true", help="rewrite conf/server identity from the table")
    parser.add_argument("--apply-models", action="store_true", help="write TRT product model tomls and repoint servers")
    parser.add_argument("--check-pack", type=Path)
    parser.add_argument("--write-manifest", type=Path, help="write the trtexec manifest derived from the table")
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--convert-filters", action="store_true")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--lookup-id",
        default=None,
        help="print the product engine path for this id (read-only)",
    )
    parser.add_argument(
        "--missing-product-engines",
        action="store_true",
        help="print id and engine for missing or empty product engines (read-only)",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    # Read-only queries return before --emit / --apply-models / --apply-servers.
    if args.lookup_id is not None:
        return lookup_product_engine(args.lookup_id)
    if args.missing_product_engines:
        return print_missing_product_engines()

    rows = model_rows()
    errors = validate_rows(rows)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    if args.emit:
        TABLE_PATH.write_text(json.dumps(document(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {TABLE_PATH}")

    if args.apply_servers:
        apply_servers(rows)
        print("updated conf/server identity")

    if args.apply_models:
        apply_model_configs(rows)
        print("wrote product TensorRT model configs")

    if args.check:
        if not TABLE_PATH.is_file():
            print(f"missing {TABLE_PATH}; run --emit", file=sys.stderr)
            return 1
        on_disk = load_table()
        fresh = document()
        if on_disk != fresh:
            print("conf/product_static_bs1.json drifted from scripts/product_static_bs1.py", file=sys.stderr)
            return 1
        disk_errors = (
            validate_rows(on_disk["models"])
            + validate_product_files(on_disk["models"])
            + check_set_a_manifest(on_disk["models"])
        )
        if disk_errors:
            for err in disk_errors:
                print(err, file=sys.stderr)
            return 1
        print(f"product_static_bs1: {len(on_disk['models'])} ids ok")

    if args.check_pack is not None:
        pack_errors = check_pack(args.check_pack, rows)
        if pack_errors:
            for err in pack_errors:
                print(err, file=sys.stderr)
            return 1
        print(f"pack ok: {args.check_pack}")

    if args.write_manifest is not None:
        args.write_manifest.write_text(json.dumps(trt_manifest(rows), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.write_manifest}")

    if args.convert_filters:
        if args.pack is None:
            print("--pack is required with --convert-filters", file=sys.stderr)
            return 1
        pack = args.pack if args.pack.is_absolute() else args.project_root / args.pack
        for token in convert_filters_for_pack(pack, args.project_root):
            print(token)

    if not any((args.emit, args.check, args.apply_servers, args.apply_models, args.check_pack, args.write_manifest, args.convert_filters)):
        parser.error("choose --emit, --check, --apply-servers, --apply-models, --check-pack, --write-manifest, or --convert-filters")
    return 0


def self_test() -> int:
    rows = model_rows()
    errors = validate_rows(rows)
    if errors:
        print("self-test rows invalid: " + "; ".join(errors), file=sys.stderr)
        return 1
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "pack.toml"
        pack.write_text("[pack.YOLOV8]\nworker_nums=1\n", encoding="utf-8")
        bad = check_pack(pack, rows)
        if not any("YOLOV8" in err for err in bad):
            print("self-test: uppercase pack id must fail", file=sys.stderr)
            return 1
        pack.write_text("[pack.yolov8s]\nworker_nums=1\n", encoding="utf-8")
        if check_pack(pack, rows):
            print("self-test: yolov8s pack must pass", file=sys.stderr)
            return 1
        manifest = trt_manifest(rows)
        yolov8s = next(item for item in manifest["engines"] if item["model"] == "yolov8s")
        if not yolov8s["engine"].endswith(".fp16") or yolov8s["fp"] != 1:
            print("self-test: yolov8s manifest precision", file=sys.stderr)
            return 1
        mobilenet = next(item for item in manifest["engines"] if item["model"] == "mobilenetv2")
        if not mobilenet["engine"].endswith(".fp32") or mobilenet["fp"] != 0:
            print("self-test: mobilenetv2 manifest precision", file=sys.stderr)
            return 1
    print("product_static_bs1.py self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
