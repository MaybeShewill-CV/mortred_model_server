#!/usr/bin/env python3
"""Stamp P2 local ONNX paths and provenance onto conf/onnx_sources.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/g/Codex/mortred_model_server")
PATH = ROOT / "conf" / "onnx_sources.json"

PATCH = {
    ("MOBILENETV2", None): {
        "onnx_path": "weights/classification/mobilenetv2/mobilenetv2.static_bs1.onnx",
        "onnx_dyn_path": "weights/classification/mobilenetv2/mobilenetv2.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "torchvision.models.mobilenet_v2 MobileNet_V2_Weights.IMAGENET1K_V1",
            "script": "scripts/export_onnx/export_classification.py",
            "notes": "NHWC wrapper permute(0,3,1,2). Graph is F32 0-255-scale tensors; caffe mean/std stays in C++. Dual files .static_bs1 / .dyn.",
        },
    },
    ("RESNET", None): {
        "onnx_path": "weights/classification/resnet/resnet.static_bs1.onnx",
        "onnx_dyn_path": "weights/classification/resnet/resnet.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "torchvision.models.resnet50 ResNet50_Weights.IMAGENET1K_V1",
            "script": "scripts/export_onnx/export_classification.py",
            "notes": "Same NHWC wrap as MOBILENETV2.",
        },
    },
    ("DENSENET", None): {
        "onnx_path": "weights/classification/densenet/densenet.static_bs1.onnx",
        "onnx_dyn_path": "weights/classification/densenet/densenet.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "torchvision.models.densenet121 DenseNet121_Weights.IMAGENET1K_V1",
            "script": "scripts/export_onnx/export_classification.py",
            "notes": "Same NHWC wrap as MOBILENETV2.",
        },
    },
    ("YOLOV5", None): {
        "onnx_path": "weights/object_detection/yolov5/yolov5l.static_bs1.onnx",
        "onnx_dyn_path": "weights/object_detection/yolov5/yolov5l.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "https://hf-mirror.com/ultralytics/yolov5/resolve/main/yolov5l.pt (v7.0) + ultralytics/yolov5 export.py",
            "script": "scripts/export_onnx/ (yolov5 export.py; output0 renamed to output)",
            "notes": "Output name output [1,25200,85]. Dyn pair deep-patched Reshape dim0. GPL-3.0; do not upload until license review.",
        },
    },
    ("YOLOV6", None): {
        "onnx_path": "weights/object_detection/yolov6/yolov6s.static_bs1.onnx",
        "onnx_dyn_path": "weights/object_detection/yolov6/yolov6s.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_onnx",
            "ref": "Meituan YOLOv6 0.3.0 yolov6s.onnx (gh-proxy github.com/meituan/YOLOv6)",
            "script": "scripts/export_onnx/adopt_official_onnx.py",
            "notes": "Already named outputs [1,8400,85] (anchor-free). C++ accepts [1,-1,85]. GPL-3.0.",
        },
    },
    ("NANODET", None): {
        "onnx_path": "weights/object_detection/nanodet/nanodet_plus_m_1x5.static_bs1.onnx",
        "onnx_dyn_path": "weights/object_detection/nanodet/nanodet_plus_m_1x5.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_onnx",
            "ref": "RangiLyu/nanodet official nanodet-plus-m-1.5x_416.onnx",
            "script": "scripts/export_onnx/common.py fold_weight_inputs + set_batch_dim",
            "notes": "Product 1x5 == plus-m-1.5x (9.5MB). Folded 260 leaked weight graph.inputs so only data remains. output [1,3598,112]. Extra sibling nanodet_plus_m_416 not product.",
        },
    },
    ("LIBFACE", "320x240"): {
        "product_type": "onnx",
        "runtime_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.onnx",
        "onnx_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.static_bs1.onnx",
        "onnx_dyn_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.dyn.onnx",
        "legacy_onnx_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "contract": {
            "source": "on_init+toml",
            "inputs": [
                {
                    "name": "input",
                    "layout": "NCHW",
                    "dtype": "F32",
                    "shape": [1, 3, "height", "width"],
                    "dynamic": True,
                }
            ],
            "outputs": [
                {"name": "cls_8"},
                {"name": "cls_16"},
                {"name": "cls_32"},
                {"name": "obj_8"},
                {"name": "obj_16"},
                {"name": "obj_32"},
                {"name": "bbox_8"},
                {"name": "bbox_16"},
                {"name": "bbox_32"},
                {"name": "kps_8"},
                {"name": "kps_16"},
                {"name": "kps_32"},
            ],
            "preprocess": "BGR 0-255; optional DIRECT_RESIZE from toml then right/bottom pad to /32",
        },
        "onnx_source": {
            "kind": "official_onnx",
            "ref": "opencv/opencv_zoo models/face_detection_yunet/face_detection_yunet_2026may.onnx",
            "script": "scripts/export_onnx/adopt_official_onnx.py + apply_reshape_batch_passthrough",
            "notes": "Original 2026may kept for product toml. static_bs1 = N=1, H/W still dynamic. dyn opens batch and rewrites 12 Reshape [1,-1,C] to [0,-1,C]. vs original ORT maxabs=0, dyn N=2 matches. C++ resizes to toml [320,240] then pads to /32.",
        },
    },
    ("LIBFACE", "640x480"): {
        "product_type": "onnx",
        "runtime_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.onnx",
        "onnx_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.static_bs1.onnx",
        "onnx_dyn_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.dyn.onnx",
        "legacy_onnx_path": "weights/object_detection/libfacedetection/face_detection_yunet_2026may.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "contract": {
            "source": "on_init+toml",
            "inputs": [
                {
                    "name": "input",
                    "layout": "NCHW",
                    "dtype": "F32",
                    "shape": [1, 3, "height", "width"],
                    "dynamic": True,
                }
            ],
            "outputs": [
                {"name": "cls_8"},
                {"name": "cls_16"},
                {"name": "cls_32"},
                {"name": "obj_8"},
                {"name": "obj_16"},
                {"name": "obj_32"},
                {"name": "bbox_8"},
                {"name": "bbox_16"},
                {"name": "bbox_32"},
                {"name": "kps_8"},
                {"name": "kps_16"},
                {"name": "kps_32"},
            ],
            "preprocess": "BGR 0-255; DIRECT_RESIZE to 640x480 (already /32)",
        },
        "onnx_source": {
            "kind": "official_onnx",
            "ref": "same face_detection_yunet_2026may.onnx as 320x240",
            "script": "scripts/export_onnx/adopt_official_onnx.py + apply_reshape_batch_passthrough",
            "notes": "Same dual pair. C++ resizes to toml [640,480]. Golden still uses this config.",
        },
    },
    ("DBNET", None): {
        "onnx_path": "weights/ocr/db_text_detector/db_model_large.static_bs1.onnx",
        "onnx_dyn_path": "weights/ocr/db_text_detector/db_model_large.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "paddle2onnx",
            "ref": "PaddleOCR ch_ppocr_server_v2.0_det_infer (bcebos) paddle2onnx opset 11",
            "script": "scripts/export_onnx/freeze_spatial.py + write_pair_from_static",
            "notes": "IO already x / sigmoid_0.tmp_0. Spatial frozen to product 544x960. Backbone family may differ from db_model_large.mnn; golden may drift.",
        },
    },
    ("BISENETV2", None): {
        "onnx_path": "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.static_bs1.onnx",
        "onnx_dyn_path": "weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "MaybeShewill-CV/bisenetv2-tensorflow cityscapes.ckpt (EMA) freeze softmax squeeze",
            "script": "scripts/export_onnx/export_bisenetv2.py",
            "notes": "Product MNN is this TF graph, not CoinCheung pth. input_tensor NHWC [1,512,1024,3] → final_output HWC softmax [512,1024,19]. Argmax vs product MNN agrees 1.0. Dyn only opens input batch; output is squeezed rank-3.",
        },
    },
    ("PPHUMAN_SEG", None): {
        "onnx_path": "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.static_bs1.onnx",
        "onnx_dyn_path": "weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "local human_pp_humansegv2_mobile_192x192_pretrained/model.pdparams (PPLiteSeg STDC1)",
            "script": "scripts/export_onnx/export_pphuman.py",
            "notes": "Dygraph pdparams → jit softmax wrap → paddle2onnx opset 11, freeze 192. IO x / softmax_0.tmp_0 [1,2,192,192]. vs product MNN argmax agree 1.0 maxabs~2e-4. Sibling lite 192 and v1-server 512 also exported. Product toml stays type=mnn.",
        },
    },
    ("MODNET", None): {
        "onnx_path": "weights/matting/modnet/modnet_hrnet_w18.static_bs1.onnx",
        "onnx_dyn_path": "weights/matting/modnet/modnet_hrnet_w18.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "paddle2onnx",
            "ref": "PaddleSeg modnet-hrnet_w18.zip",
            "script": "paddle2onnx + freeze 512 + write_pair_from_static",
            "notes": "img / sigmoid_2.tmp_0 frozen 512x512 (session requires static H/W).",
        },
    },
    ("PP_MATTING", None): {
        "onnx_path": "weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.static_bs1.onnx",
        "onnx_dyn_path": "weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "paddle2onnx",
            "ref": "PaddleSeg ppmatting-hrnet_w18-human_512.zip (bcebos); user also provided 1024/v2/resnet34 inference dirs",
            "script": "scripts/export_onnx/export_ppmatting.py",
            "notes": "Dynamic AdaptiveAvgPool ASPP (1/3/5) needs static feed before paddle2onnx 1.2.3. Freeze img to [1,3,512,512] via ProgramDesc InferShape (C++ on_init also requires area>0). IO img / tmp_75. vs Paddle maxabs~1e-6; vs product MNN corr~0.99993 meanabs~3e-4 (argmax fusion edge pixels). Sibling 1024 (same 512 freeze) and resnet34_vd 2048 also exported. v2-stdc1 is a different graph (sigmoid_5.tmp_0), not the HTTP product. Product toml stays type=mnn.",
        },
        "contract": {
            "source": "on_init+session",
            "inputs": [
                {
                    "name": "img",
                    "layout": "NCHW",
                    "dtype": "F32",
                    "shape": [1, 3, 512, 512],
                    "dynamic": False,
                }
            ],
            "outputs": [{"name": "tmp_75"}],
            "preprocess": "static 512 from session; RGB /255 mean/std 0.5; matte maps to source_size",
        },
        "license": {
            "redistribute": "p2_review",
            "notes": "PaddleSeg PP-Matting. Product toml stays type=mnn. Sibling MNNs stay until convert_mnn.",
        },
    },
    ("ENLIGHTEN_GAN", None): {
        "onnx_path": "weights/enhancement/enlighten_gan/enlightengan.static_bs1.onnx",
        "onnx_dyn_path": "weights/enhancement/enlighten_gan/enlightengan.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "local 200_net_G_A.pth Unet_resize_conv self_attention+skip+times_residual",
            "script": "scripts/export_onnx/export_enlighten.py",
            "notes": "Product C++ feeds NCHW input_src [1,3,H,W] + input_gray [1,1,H,W] already in [-1,1]/Rec.601 gray, H/W align-up-16. Local enlighten.onnx is the same Unet (weights maxabs 0) with 2x-1 + gray fused into a single `input` — wrong IO, not adopted. Dual files keep spatial dynamic; vs product MNN maxabs~3e-4 corr~1. Original enlighten.onnx kept. Do not freeze 256.",
        },
        "license": {
            "redistribute": "p2_review",
            "notes": "Unet_resize_conv from local 200_net_G_A.pth. Product toml stays type=mnn. Original enlighten.onnx is a fused single-input wrapper, not the interchange file.",
        },
    },
    ("ATTENTIVE_GAN_DERAIN", None): {
        "onnx_path": "weights/enhancement/attentive_gan_derain/attentive_gan_derain.static_bs1.onnx",
        "onnx_dyn_path": "weights/enhancement/attentive_gan_derain/attentive_gan_derain.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "MaybeShewill-CV/attentive-gan-derainnet derain_gan.ckpt-100000 freeze squeezed tanh skip_3",
            "script": "scripts/export_onnx/export_attentive_gan.py",
            "notes": "Product MNN is this TF graph at 240x360 (toml [240,320] is stale; C++ uses session shape). input_tensor NHWC [1,240,360,3] → final_output HWC [240,360,3]. Local attentive_gan_derain.pb is a text training GraphDef, not a frozen net. vs product MNN maxabs~1e-3 corr~1. Dyn only opens input batch; LSTM init constants stay N=1.",
        },
    },
    ("REAL_ESRGAN", None): {
        "onnx_path": "weights/enhancement/real_esrgan/realesr-general-x4v3.static_bs1.onnx",
        "onnx_dyn_path": "weights/enhancement/real_esrgan/realesr-general-x4v3.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "xinntao/Real-ESRGAN realesr-general-x4v3.pth (SRVGGNetCompact)",
            "script": "scripts/export_onnx/export_realesrgan.py",
            "notes": "NHWC RGB /255 in, NCHW out; dynamic H/W; static_bs1 keeps batch=1. Matches C++ contract.",
        },
    },
    ("SUPERPOINT", None): {
        "onnx_path": "weights/feature_point/superpoint/superpoint_120x160.static_bs1.onnx",
        "onnx_dyn_path": "weights/feature_point/superpoint/superpoint_120x160.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "MagicLeap superpoint_v1.pth",
            "script": "scripts/export_onnx/export_superpoint.py",
            "notes": "GRAY NCHW 120x160; output_1 65-ch; output_2 256-ch.",
        },
    },
    ("DINOV2", "vits14"): {
        "onnx_path": "weights/classification/dinov2/dinov2_vits14_pretrain.static_bs1.onnx",
        "onnx_dyn_path": "weights/classification/dinov2/dinov2_vits14_pretrain.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "timm vit_small_patch14_dinov2.lvd142m (Meta LVD-142M, img_size=224 cls-only)",
            "script": "scripts/export_onnx/export_dinov2_timm.py",
            "notes": "Official Meta pos-embed is 518; this pair interpolates to 224. Output [1,384] rank-2 (pooling=cls). Apache-2.0.",
        },
    },
    ("DINOV2", "vitb14"): {
        "onnx_path": "weights/classification/dinov2/dinov2_vitb14_pretrain.static_bs1.onnx",
        "onnx_dyn_path": "weights/classification/dinov2/dinov2_vitb14_pretrain.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "timm vit_base_patch14_dinov2.lvd142m img_size=224 cls-only",
            "script": "scripts/export_onnx/export_dinov2_timm.py",
            "notes": "Output [1,768]. Same 224 interpolation note as vits14.",
        },
    },
    ("DINOV2", "vitl14"): {
        "onnx_path": "weights/classification/dinov2/dinov2_vitl14_pretrain.static_bs1.onnx",
        "onnx_dyn_path": "weights/classification/dinov2/dinov2_vitl14_pretrain.dyn.onnx",
        "onnx_status": "local",
        "p1_upload": False,
        "onnx_source": {
            "kind": "official_ckpt_export",
            "ref": "timm vit_large_patch14_dinov2.lvd142m img_size=224 cls-only",
            "script": "scripts/export_onnx/export_dinov2_timm.py",
            "notes": "Output [1,1024]. About 1.2 GB per file.",
        },
    },
}


def main() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data["description"] = (
        "Per product-id ONNX interchange source, IO contract, and license. "
        "Dual-file rules and remaining gaps: docs/onnx-interchange.md. "
        "HF hosts ONNX (plus CLIP vocab) after all interchange files exist — P2 local files are not uploaded yet. "
        "Product conf/model/**/*.toml stay type=mnn or type=tensorrt except diffusion/MSOCRNET/SAM-decoder overlays and LIBFACE (YuNet 2026may type=onnx). "
        "Do not reverse MNN/.model/.engine. Bytes/sha256 live in conf/weights_manifest.json. "
        "Dual files: {stem}.static_bs1.onnx and {stem}.dyn.onnx."
    )
    for model in data["models"]:
        key = (model["id"], model.get("variant"))
        if model["id"] == "OPENAI_CLIP":
            model["onnx_status"] = "local"
            model["p1_upload"] = False
            model["onnx_source"] = {
                "kind": "official_ckpt_export",
                "ref": "OpenAI CLIP ViT-B/32 (clip.load)",
                "script": "scripts/export_onnx/export_clip.py + I32 Cast wrap on text",
                "notes": "visual F32 NCHW 224 L2-normed; textual INT32 [1,77] Cast to INT64 inside graph. Vocab stays non-ONNX.",
            }
            for graph in model.get("graphs", []):
                if graph.get("role") == "visual":
                    graph["onnx_path"] = "weights/openai_clip/vit-b-32/visual.static_bs1.onnx"
                    graph["onnx_dyn_path"] = "weights/openai_clip/vit-b-32/visual.dyn.onnx"
                elif graph.get("role") == "textual":
                    graph["onnx_path"] = "weights/openai_clip/vit-b-32/textual.static_bs1.onnx"
                    graph["onnx_dyn_path"] = "weights/openai_clip/vit-b-32/textual.dyn.onnx"
            continue
        if model["id"] == "SAM_PREDICTOR" and model.get("variant") == "vit_l":
            for graph in model.get("graphs", []):
                if graph.get("role") == "encoder":
                    graph["onnx_status"] = "blocked"
                    graph["onnx_source"] = {
                        "kind": "none",
                        "notes": "No matching official encoder ONNX adopted; do not reverse the ~1.2GB MNN.",
                    }
            model["onnx_source"] = {
                "kind": "partial",
                "notes": "Decoder already local ONNX; encoder still blocked.",
            }
            continue
        patch = PATCH.get(key)
        if not patch:
            # stamp preexisting sources
            if model.get("onnx_status") in {"hosted", "local"} and "onnx_source" not in model:
                model["onnx_source"] = {
                    "kind": "preexisting",
                    "ref": model.get("onnx_path"),
                    "notes": "Present before P2; keep original filename (not dual-renamed).",
                }
            continue
        model.update(patch)
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated", PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
