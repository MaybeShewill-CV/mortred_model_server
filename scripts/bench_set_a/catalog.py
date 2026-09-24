#!/usr/bin/env python3
"""A-set catalog: convert/profile/HTTP order for the Set A TRT benchmark."""
from __future__ import annotations

# Execution order matches the agreed plan. libface converts once; HTTP is two rows.
# msocrnet is convert+profile only (no HTTP catalog).


def _e(onnx: str) -> str:
    return onnx.replace(".static_bs1.onnx", ".engine.static_bs1.fp16")


def M(
    order: int,
    id: str,
    onnx: str,
    catalog: str,
    product: str,
    server: str,
    demo: str,
    *,
    http: bool = True,
    spatial: str = "static",
    convert_id: str | None = None,
    extra_params: dict | None = None,
    layout: str = "nchw",
) -> dict:
    return {
        "order": order,
        "id": id,
        "onnx": onnx,
        "engine": _e(onnx),
        "catalog": catalog,
        "product": product,
        "server": server,
        "demo": demo,
        "http": http,
        "spatial": spatial,
        "convert_id": convert_id or id,
        "extra_params": extra_params or {},
        "layout": layout,
    }


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

YOLO8_P = "conf/model/object_detection/yolov8/yolov8_config.toml"
YOLO8_S = "conf/server/object_detection/yolov8/yolov8_server_config.toml"
YOLO7_P = "conf/model/object_detection/yolov7/yolov7_config.toml"
YOLO7_S = "conf/server/object_detection/yolov7/yolov7_server_config.toml"
NANO_P = "conf/model/object_detection/nano_det/nanodet_config.toml"
NANO_S = "conf/server/object_detection/nano_det/nanodet_server_config.toml"
PPH_P = "conf/model/scene_segmentation/pphuman/pphuman_config.toml"
PPH_S = "conf/server/scene_segmentation/pphuman_seg/pphuman_seg_server_config.toml"
PPM_P = "conf/model/matting/ppmatting/ppmatting_config.toml"
PPM_S = "conf/server/matting/ppmatting/pp_matting_server_config.toml"
DINO_S = "conf/server/feature_embedding/dinov2/dinov2_feature_embedding_server_config.toml"
DA_S = "conf/server/mono_depth_estimation/depth_anything/depth_anything_server_cfg.toml"

MODELS: list[dict] = [
    M(1, "yolov8s", "weights/object_detection/yolov8/yolov8s.static_bs1.onnx", "YOLOV8", YOLO8_P, YOLO8_S, BUS),
    M(2, "yolov8n", "weights/object_detection/yolov8/yolov8n.static_bs1.onnx", "YOLOV8", YOLO8_P, YOLO8_S, BUS),
    M(3, "yolov8l", "weights/object_detection/yolov8/yolov8l.static_bs1.onnx", "YOLOV8", YOLO8_P, YOLO8_S, BUS),
    M(4, "yolov8x", "weights/object_detection/yolov8/yolov8x.static_bs1.onnx", "YOLOV8", YOLO8_P, YOLO8_S, BUS),
    M(5, "yolov6s", "weights/object_detection/yolov6/yolov6s.static_bs1.onnx", "YOLOV6",
      "conf/model/object_detection/yolov6/yolov6_config.toml",
      "conf/server/object_detection/yolov6/yolov6_server_config.toml", BUS),
    M(6, "yolov5l", "weights/object_detection/yolov5/yolov5l.static_bs1.onnx", "YOLOV5",
      "conf/model/object_detection/yolov5/yolov5_config.toml",
      "conf/server/object_detection/yolov5/yolov5_server_config.toml", BUS),
    M(7, "yolov7", "weights/object_detection/yolov7/yolov7.static_bs1.onnx", "YOLOV7", YOLO7_P, YOLO7_S, BUS),
    M(8, "yolov7x", "weights/object_detection/yolov7/yolov7x.static_bs1.onnx", "YOLOV7", YOLO7_P, YOLO7_S, BUS),
    M(9, "nanodet_1x5", "weights/object_detection/nanodet/nanodet_plus_m_1x5.static_bs1.onnx", "NANODET",
      NANO_P, NANO_S, BUS),
    M(10, "nanodet_416", "weights/object_detection/nanodet/nanodet_plus_m_416.static_bs1.onnx", "NANODET",
      NANO_P, NANO_S, BUS, extra_params={"model_input_image_size": [416, 416]}),
    M(11, "mobilenetv2", "weights/classification/mobilenetv2/mobilenetv2.static_bs1.onnx", "MOBILENETV2",
      "conf/model/classification/mobilenetv2/mobilenetv2_config.toml",
      "conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml", CLS, layout="nhwc"),
    M(12, "resnet", "weights/classification/resnet/resnet.static_bs1.onnx", "RESNET",
      "conf/model/classification/resnet/resnet50_config.toml",
      "conf/server/classification/resnet/resnet50_server_config.toml", CLS, layout="nhwc"),
    M(13, "densenet", "weights/classification/densenet/densenet.static_bs1.onnx", "DENSENET",
      "conf/model/classification/densenet/densenet121_config.toml",
      "conf/server/classification/densenet/densenet_server_config.toml", CLS, layout="nhwc"),
    M(14, "superpoint", "weights/feature_point/superpoint/superpoint_120x160.static_bs1.onnx", "SUPERPOINT",
      "conf/model/feature_point/superpoint/superpoint_config.toml",
      "conf/server/feature_point/superpoint/superpoint_server_cfg.toml", SP),
    M(15, "dbnet", "weights/ocr/db_text_detector/db_model_large.static_bs1.onnx", "DBNET",
      "conf/model/ocr/db_text_detector/dbnet_config.toml",
      "conf/server/ocr/dbnet/dbtext_detection_server_config.toml", OCR),
    M(16, "libface_320", "weights/object_detection/libfacedetection/face_detection_yunet_2026may.static_bs1.onnx",
      "LIBFACE", "conf/model/object_detection/libfacedetection/320x240_config.toml",
      "conf/server/object_detection/libface_det/libface_server_config.toml", FACE,
      spatial="dynamic", convert_id="libface"),
    M(17, "libface_640", "weights/object_detection/libfacedetection/face_detection_yunet_2026may.static_bs1.onnx",
      "LIBFACE", "conf/model/object_detection/libfacedetection/640x480_config.toml",
      "conf/server/object_detection/libface_det/libface_server_config.toml", FACE,
      spatial="dynamic", convert_id="libface"),
    M(18, "centerface", "weights/object_detection/centerface_detection/centerface.static_bs1.onnx", "CENTER_FACE",
      "conf/model/object_detection/centerface/centerface_config.toml",
      "conf/server/object_detection/center_face_det/center_face_server_config.toml", FACE,
      spatial="dynamic"),
    M(19, "pphuman_mobile", "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.static_bs1.onnx",
      "PPHUMAN_SEG", PPH_P, PPH_S, HUMAN),
    M(20, "pphuman_lite", "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.static_bs1.onnx",
      "PPHUMAN_SEG", PPH_P, PPH_S, HUMAN),
    M(21, "pphuman_server", "weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.static_bs1.onnx",
      "PPHUMAN_SEG", PPH_P, PPH_S, HUMAN),
    M(22, "bisenetv2", "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.static_bs1.onnx", "BISENETV2",
      "conf/model/scene_segmentation/bisenetv2/bisenetv2_config.toml",
      "conf/server/scene_segmentation/bisenetv2/bisenetv2_server_config.toml", CITY, layout="nhwc"),
    M(23, "modnet", "weights/matting/modnet/modnet_hrnet_w18.static_bs1.onnx", "MODNET",
      "conf/model/matting/modnet/modnet_config.toml",
      "conf/server/matting/modnet/modnet_server_config.toml", MATT),
    M(24, "ppmatting_512", "weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.static_bs1.onnx", "PP_MATTING",
      PPM_P, PPM_S, MATT),
    M(25, "ppmatting_1024", "weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.static_bs1.onnx", "PP_MATTING",
      PPM_P, PPM_S, MATT),
    M(26, "ppmatting_resnet34", "weights/matting/ppmatting/pp_humanmatting-resnet34_vd.static_bs1.onnx", "PP_MATTING",
      PPM_P, PPM_S, MATT),
    M(27, "ppmatting_v2", "weights/matting/ppmatting/ppmattingv2_stdc1_human_512.static_bs1.onnx", "PP_MATTING",
      PPM_P, PPM_S, MATT),
    M(28, "attentive_gan", "weights/enhancement/attentive_gan_derain/attentive_gan_derain.static_bs1.onnx",
      "ATTENTIVE_GAN_DERAIN",
      "conf/model/enhancement/attentive_gan_derain/attentive_gan_derain_config.toml",
      "conf/server/enhancement/attentive_gan_derain/attentive_gan_server_cfg.toml", DERAIN, layout="nhwc"),
    M(29, "enlightengan", "weights/enhancement/enlighten_gan/enlightengan.static_bs1.onnx", "ENLIGHTEN_GAN",
      "conf/model/enhancement/enlighten_gan/enlightengan.toml",
      "conf/server/enhancement/enlightengan/enlighten_gan_server_cfg.toml", LOL, spatial="dynamic"),
    M(30, "realesrgan", "weights/enhancement/real_esrgan/realesr-general-x4v3.static_bs1.onnx", "REAL_ESRGAN",
      "conf/model/enhancement/real_esrgan/real_esrgan.toml",
      "conf/server/enhancement/real_esrgan/real_esrgan_server_cfg.toml", ESR, spatial="dynamic", layout="nhwc"),
    M(31, "dinov2_vits14", "weights/classification/dinov2/dinov2_vits14_pretrain.static_bs1.onnx", "DINOV2",
      "conf/model/feature_embedding/dinov2/dinov2_vits14_config.toml", DINO_S, CLS),
    M(32, "dinov2_vitb14", "weights/classification/dinov2/dinov2_vitb14_pretrain.static_bs1.onnx", "DINOV2",
      "conf/model/feature_embedding/dinov2/dinov2_vitb14_config.toml", DINO_S, CLS),
    M(33, "dinov2_vitl14", "weights/classification/dinov2/dinov2_vitl14_pretrain.static_bs1.onnx", "DINOV2",
      "conf/model/feature_embedding/dinov2/dinov2_vitl14_config.toml", DINO_S, CLS),
    M(34, "depth_vits14", "weights/mono_depth_estimation/depth_anything/depth_anything_vits14.static_bs1.onnx",
      "DEPTH_ANYTHING", "conf/model/mono_depth_estimation/depth_anything/vit_s.toml", DA_S, DEPTH),
    M(35, "depth_vitb14", "weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.static_bs1.onnx",
      "DEPTH_ANYTHING", "conf/model/mono_depth_estimation/depth_anything/vit_b.toml", DA_S, DEPTH),
    M(36, "depth_vitl14", "weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.static_bs1.onnx",
      "DEPTH_ANYTHING", "conf/model/mono_depth_estimation/depth_anything/vit_l.toml", DA_S, DEPTH),
    M(37, "metric3d_512", "weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.static_bs1.onnx",
      "METRIC3D", "conf/model/mono_depth_estimation/metric3d/metric3d_512x1088.toml",
      "conf/server/mono_depth_estimation/metric3d/metric3d_server_cfg.toml", DEPTH),
    M(38, "metric3d_1088", "weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.static_bs1.onnx",
      "METRIC3D", "conf/model/mono_depth_estimation/metric3d/metric3d_1088x1920.toml",
      "conf/server/mono_depth_estimation/metric3d/metric3d_server_cfg.toml", DEPTH),
    M(39, "hrnet", "weights/scene_segmentation/hrnet/hrnetw48_ccd.static_bs1.onnx", "HRNET",
      "conf/model/scene_segmentation/hrnet/hrnetw48_ccd_fv_segmentation_cfg.toml",
      "conf/server/scene_segmentation/hrnet/hrnet_seg_server_config.toml", CITY),
    M(40, "msocrnet", "weights/scene_segmentation/msocrnet/msocrnet_fp16.static_bs1.onnx", "MSOCRNET",
      "conf/model/scene_segmentation/msocrnet/msocrnet_config.toml", "", OCR, http=False),
]


def unique_converts() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in MODELS:
        cid = row["convert_id"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append(row)
    return out
