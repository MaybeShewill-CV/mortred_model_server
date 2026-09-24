# A 集合 TensorRT 静态 batch=1 基准（中文）

实测表。空单元格为未测或失败。trtexec Throughput 不是 HTTP RPS。表 3 不是发布 HTTP 值。

## 机器

| 项 | 值 |
|---|---|
| GPU | name, memory.total [MiB], memory.free [MiB], driver_version, compute_cap | NVIDIA GeForce RTX 2070 SUPER, 8192 MiB, 6924 MiB, 616.92, 7.5 |
| 日期 | 2026-09-24 |
| workspace | 6G (8 GB card; trtexec suffix G, not GiB) |
| 产品 toml | 未改；标定/HTTP 使用 conf/bench/set_a/<id>.server.toml → <id>.toml |

## 表 1 — 身份 / 转换

| id | catalog | fp | engine | convert |
|---|---|---|---|---|
| yolov8s | YOLOV8 | fp16 | weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16 | exists |
| yolov8n | YOLOV8 | fp16 | weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16 | ok |
| yolov8l | YOLOV8 | fp16 | weights/object_detection/yolov8/yolov8l.engine.static_bs1.fp16 | exists |
| yolov8x | YOLOV8 | fp16 | weights/object_detection/yolov8/yolov8x.engine.static_bs1.fp16 | ok |
| yolov6s | YOLOV6 | fp16 | weights/object_detection/yolov6/yolov6s.engine.static_bs1.fp16 | exists |
| yolov5l | YOLOV5 | fp16 | weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16 | exists |
| yolov7 | YOLOV7 | fp16 | weights/object_detection/yolov7/yolov7.engine.static_bs1.fp16 | exists |
| yolov7x | YOLOV7 | fp16 | weights/object_detection/yolov7/yolov7x.engine.static_bs1.fp16 | exists |
| nanodet_1x5 | NANODET | fp16 | weights/object_detection/nanodet/nanodet_plus_m_1x5.engine.static_bs1.fp16 | exists |
| nanodet_416 | NANODET | fp16 | weights/object_detection/nanodet/nanodet_plus_m_416.engine.static_bs1.fp16 | exists |
| mobilenetv2 | MOBILENETV2 | fp32 | weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32 | exists |
| resnet | RESNET | fp32 | weights/classification/resnet/resnet.engine.static_bs1.fp32 | exists |
| densenet | DENSENET | fp32 | weights/classification/densenet/densenet.engine.static_bs1.fp32 | exists |
| superpoint | SUPERPOINT | fp16 | weights/feature_point/superpoint/superpoint_120x160.engine.static_bs1.fp16 | ok |
| dbnet | DBNET | fp16 | weights/ocr/db_text_detector/db_model_large.engine.static_bs1.fp16 | ok |
| libface_320 | LIBFACE | fp16 | weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16 | ok |
| libface_640 | LIBFACE | (same engine libface) | — | — |
| centerface | CENTER_FACE | fp16 | weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16 | ok |
| pphuman_mobile | PPHUMAN_SEG | fp16 | weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.engine.static_bs1.fp16 | ok |
| pphuman_lite | PPHUMAN_SEG | fp16 | weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.engine.static_bs1.fp16 | ok |
| pphuman_server | PPHUMAN_SEG | fp16 | weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.engine.static_bs1.fp16 | ok |
| bisenetv2 | BISENETV2 | fp32 | weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.engine.static_bs1.fp32 | ok |
| modnet | MODNET | fp16 | weights/matting/modnet/modnet_hrnet_w18.engine.static_bs1.fp16 | ok |
| ppmatting_512 | PP_MATTING | fp16 | weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.engine.static_bs1.fp16 | ok |
| ppmatting_1024 | PP_MATTING | fp16 | weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.engine.static_bs1.fp16 | ok |
| ppmatting_resnet34 | PP_MATTING | fp16 | weights/matting/ppmatting/pp_humanmatting-resnet34_vd.engine.static_bs1.fp16 | ok |
| ppmatting_v2 | PP_MATTING | fp16 | weights/matting/ppmatting/ppmattingv2_stdc1_human_512.engine.static_bs1.fp16 | ok |
| attentive_gan | ATTENTIVE_GAN_DERAIN | fp32 | weights/enhancement/attentive_gan_derain/attentive_gan_derain.engine.static_bs1.fp32 | ok |
| enlightengan | ENLIGHTEN_GAN | fp16 | weights/enhancement/enlighten_gan/enlightengan.engine.static_bs1.fp16 | ok |
| realesrgan | REAL_ESRGAN | fp32 | weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32 | ok |
| dinov2_vits14 | DINOV2 | fp16 | weights/classification/dinov2/dinov2_vits14_pretrain.engine.static_bs1.fp16 | exists |
| dinov2_vitb14 | DINOV2 | fp16 | weights/classification/dinov2/dinov2_vitb14_pretrain.engine.static_bs1.fp16 | exists |
| dinov2_vitl14 | DINOV2 | fp16 | weights/classification/dinov2/dinov2_vitl14_pretrain.engine.static_bs1.fp16 | exists |
| depth_vits14 | DEPTH_ANYTHING | fp16 | weights/mono_depth_estimation/depth_anything/depth_anything_vits14.engine.static_bs1.fp16 | ok |
| depth_vitb14 | DEPTH_ANYTHING | fp16 | weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.engine.static_bs1.fp16 | ok |
| depth_vitl14 | DEPTH_ANYTHING | fp16 | weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.engine.static_bs1.fp16 | ok |
| metric3d_512 | METRIC3D | fp16 | weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16 | ok |
| metric3d_1088 | METRIC3D | fp16 | weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.engine.static_bs1.fp16 | ok |
| hrnet | HRNET | fp16 | weights/scene_segmentation/hrnet/hrnetw48_ccd.engine.static_bs1.fp16 | ok |
| msocrnet | MSOCRNET | fp16 | weights/scene_segmentation/msocrnet/msocrnet_fp16.engine.static_bs1.fp16 | ok |

## 表 2 — trtexec profile

| id | trtexec_rps | gpu_compute_ms | gpu_util_avg% | gpu_util_max% | cmd |
|---|---:|---:|---:|---:|---|
| yolov8s | 413.66 | 2.04 | 74.20 | 93.00 | logs/bench/set_a/trtexec/yolov8s.cmd.txt |
| yolov8n | 621.29 | 1.27 | 72.20 | 89.00 | logs/bench/set_a/trtexec/yolov8n.cmd.txt |
| yolov8l | 155.15 | 6.10 | 82.10 | 96.00 | logs/bench/set_a/trtexec/yolov8l.cmd.txt |
| yolov8x | 105.56 | 8.97 | 81.60 | 97.00 | logs/bench/set_a/trtexec/yolov8x.cmd.txt |
| yolov6s | 417.23 | 2.08 | 80.10 | 95.00 | logs/bench/set_a/trtexec/yolov6s.cmd.txt |
| yolov5l | 192.31 | 4.53 | 83.30 | 96.00 | logs/bench/set_a/trtexec/yolov5l.cmd.txt |
| yolov7 | 195.23 | 4.40 | 83.30 | 97.00 | logs/bench/set_a/trtexec/yolov7.cmd.txt |
| yolov7x | 125.89 | 7.32 | 81.40 | 98.00 | logs/bench/set_a/trtexec/yolov7x.cmd.txt |
| nanodet_1x5 | 592.87 | 1.47 | 69.50 | 85.00 | logs/bench/set_a/trtexec/nanodet_1x5.cmd.txt |
| nanodet_416 | 629.08 | 1.36 | 66.60 | 82.00 | logs/bench/set_a/trtexec/nanodet_416.cmd.txt |
| mobilenetv2 | 1532.90 | 0.65 | 65.30 | 91.00 | logs/bench/set_a/trtexec/mobilenetv2.cmd.txt |
| resnet | 446.37 | 2.24 | 75.80 | 97.00 | logs/bench/set_a/trtexec/resnet.cmd.txt |
| densenet | 225.09 | 4.44 | 75.20 | 87.00 | logs/bench/set_a/trtexec/densenet.cmd.txt |
| superpoint | 3538.71 | 0.20 | 48.70 | 80.00 | logs/bench/set_a/trtexec/superpoint.cmd.txt |
| dbnet | 328.33 | 2.71 | 84.70 | 98.00 | logs/bench/set_a/trtexec/dbnet.cmd.txt |
| libface | 1491.05 | 0.49 | 64.80 | 85.00 | logs/bench/set_a/trtexec/libface.cmd.txt |
| centerface | 803.46 | 1.00 | 75.00 | 92.00 | logs/bench/set_a/trtexec/centerface.cmd.txt |
| pphuman_mobile | 989.25 | 0.91 | 66.30 | 83.00 | logs/bench/set_a/trtexec/pphuman_mobile.cmd.txt |
| pphuman_lite | 1137.29 | 0.78 | 58.30 | 76.00 | logs/bench/set_a/trtexec/pphuman_lite.cmd.txt |
| pphuman_server | 142.64 | 6.75 | 85.90 | 99.00 | logs/bench/set_a/trtexec/pphuman_server.cmd.txt |
| bisenetv2 | 90.66 | 9.08 | 82.10 | 93.00 | logs/bench/set_a/trtexec/bisenetv2.cmd.txt |
| modnet | 97.02 | 10.02 | 84.60 | 95.00 | logs/bench/set_a/trtexec/modnet.cmd.txt |
| ppmatting_512 | 57.38 | 17.10 | 87.00 | 97.00 | logs/bench/set_a/trtexec/ppmatting_512.cmd.txt |
| ppmatting_1024 | 56.80 | 17.25 | 87.10 | 97.00 | logs/bench/set_a/trtexec/ppmatting_1024.cmd.txt |
| ppmatting_resnet34 | 38.71 | 23.64 | 86.00 | 99.00 | logs/bench/set_a/trtexec/ppmatting_resnet34.cmd.txt |
| ppmatting_v2 | 245.43 | 3.78 | 81.60 | 93.00 | logs/bench/set_a/trtexec/ppmatting_v2.cmd.txt |
| attentive_gan | 31.89 | 31.11 | 86.40 | 99.00 | logs/bench/set_a/trtexec/attentive_gan.cmd.txt |
| enlightengan | 418.00 | 2.22 | 80.50 | 95.00 | logs/bench/set_a/trtexec/enlightengan.cmd.txt |
| realesrgan | 37.88 | 25.45 | 90.20 | 100.00 | logs/bench/set_a/trtexec/realesrgan.cmd.txt |
| dinov2_vits14 | 294.82 | 3.39 | 80.50 | 97.00 | logs/bench/set_a/trtexec/dinov2_vits14.cmd.txt |
| dinov2_vitb14 | 418.32 | 2.39 | 79.90 | 97.00 | logs/bench/set_a/trtexec/dinov2_vitb14.cmd.txt |
| dinov2_vitl14 | 149.00 | 6.71 | 77.50 | 98.00 | logs/bench/set_a/trtexec/dinov2_vitl14.cmd.txt |
| depth_vits14 | 193.81 | 4.94 | 81.30 | 97.00 | logs/bench/set_a/trtexec/depth_vits14.cmd.txt |
| depth_vitb14 | 79.51 | 12.32 | 83.20 | 99.00 | logs/bench/set_a/trtexec/depth_vitb14.cmd.txt |
| depth_vitl14 | 26.13 | 37.99 | 82.80 | 99.00 | logs/bench/set_a/trtexec/depth_vitl14.cmd.txt |
| metric3d_512 | 21.60 | 45.65 | 87.50 | 99.00 | logs/bench/set_a/trtexec/metric3d_512.cmd.txt |
| metric3d_1088 | 6.81 | 145.15 | 87.80 | 100.00 | logs/bench/set_a/trtexec/metric3d_1088.cmd.txt |
| hrnet | 17.99 | 54.91 | 90.20 | 99.00 | logs/bench/set_a/trtexec/hrnet.cmd.txt |
| msocrnet | 0.16 | 6100.09 | 95.80 | 100.00 | logs/bench/set_a/trtexec/msocrnet.cmd.txt |

## 表 3 — 标定（非发布 RPS）

`--workers 1,2,4,8,16 --duration 15s --skip-joint`

| id | status | w* | decode_auto | rps@w* | report |
|---|---|---:|---|---:|---|
| yolov8s | ok | 8 | gpu | 219.32 | logs/bench/set_a/calibrate/yolov8s.json |
| yolov8n | ok | 4 | gpu | 287.68 | logs/bench/set_a/calibrate/yolov8n.json |
| yolov8l | ok | 2 | cpu | 110.97 | logs/bench/set_a/calibrate/yolov8l.json |
| yolov8x | ok | 2 | cpu | 108.24 | logs/bench/set_a/calibrate/yolov8x.json |
| yolov6s | ok | 4 | cpu | 234.70 | logs/bench/set_a/calibrate/yolov6s.json |
| yolov5l | ok | 4 | cpu | 137.26 | logs/bench/set_a/calibrate/yolov5l.json |
| yolov7 | ok | 4 | gpu | 187.87 | logs/bench/set_a/calibrate/yolov7.json |
| yolov7x | ok | 2 | cpu | 93.45 | logs/bench/set_a/calibrate/yolov7x.json |
| nanodet_1x5 | ok | 4 | cpu | 213.72 | logs/bench/set_a/calibrate/nanodet_1x5.json |
| nanodet_416 | ok | 4 | cpu | 200.06 | logs/bench/set_a/calibrate/nanodet_416.json |
| mobilenetv2 | ok | 4 | cpu | 850.10 | logs/bench/set_a/calibrate/mobilenetv2.json |
| resnet | ok | 2 | cpu | 231.32 | logs/bench/set_a/calibrate/resnet.json |
| densenet | ok | 2 | gpu | 158.57 | logs/bench/set_a/calibrate/densenet.json |
| superpoint | ok | 4 | gpu | 1318.22 | logs/bench/set_a/calibrate/superpoint.json |
| dbnet | ok | 4 | gpu | 130.95 | logs/bench/set_a/calibrate/dbnet.json |
| libface_320 | ok | 2 | gpu | 517.85 | logs/bench/set_a/calibrate/libface_320.json |
| libface_640 | ok | 4 | gpu | 237.02 | logs/bench/set_a/calibrate/libface_640.json |
| centerface | ok | 4 | gpu | 570.02 | logs/bench/set_a/calibrate/centerface.json |
| pphuman_mobile | ok | 4 | gpu | 413.80 | logs/bench/set_a/calibrate/pphuman_mobile.json |
| pphuman_lite | ok | 4 | gpu | 372.80 | logs/bench/set_a/calibrate/pphuman_lite.json |
| pphuman_server | ok | 4 | cpu | 145.23 | logs/bench/set_a/calibrate/pphuman_server.json |
| bisenetv2 | ok | 4 | gpu | 40.91 | logs/bench/set_a/calibrate/bisenetv2.json |
| modnet | ok | 8 | cpu | 20.40 | logs/bench/set_a/calibrate/modnet.json |
| ppmatting_512 | ok | 4 | cpu | 16.47 | logs/bench/set_a/calibrate/ppmatting_512.json |
| ppmatting_1024 | ok | 8 | cpu | 17.29 | logs/bench/set_a/calibrate/ppmatting_1024.json |
| ppmatting_resnet34 | ok | 8 | gpu | 7.79 | logs/bench/set_a/calibrate/ppmatting_resnet34.json |
| ppmatting_v2 | ok | 4 | gpu | 17.22 | logs/bench/set_a/calibrate/ppmatting_v2.json |
| attentive_gan | ok | 2 | cpu | 31.65 | logs/bench/set_a/calibrate/attentive_gan.json |
| enlightengan | ok | 4 | cpu | 135.61 | logs/bench/set_a/calibrate/enlightengan.json |
| realesrgan | ok | 8 | cpu | 6.76 | logs/bench/set_a/calibrate/realesrgan.json |
| dinov2_vits14 | ok | 2 | cpu | 188.29 | logs/bench/set_a/calibrate/dinov2_vits14.json |
| dinov2_vitb14 | ok | 2 | cpu | 259.50 | logs/bench/set_a/calibrate/dinov2_vitb14.json |
| dinov2_vitl14 | ok | 2 | cpu | 92.94 | logs/bench/set_a/calibrate/dinov2_vitl14.json |
| depth_vits14 | ok | 4 | gpu | 185.85 | logs/bench/set_a/calibrate/depth_vits14.json |
| depth_vitb14 | ok | 2 | cpu | 80.03 | logs/bench/set_a/calibrate/depth_vitb14.json |
| depth_vitl14 | ok | 2 | cpu | 25.94 | logs/bench/set_a/calibrate/depth_vitl14.json |
| metric3d_512 | ok | 2 | cpu | 20.00 | logs/bench/set_a/calibrate/metric3d_512.json |
| metric3d_1088 | ok | 2 | gpu | 6.11 | logs/bench/set_a/calibrate/metric3d_1088.json |
| hrnet | ok | 2 | cpu | 16.91 | logs/bench/set_a/calibrate/hrnet.json |
| msocrnet | n/a-bench | — | — | — | — |

## 表 4 — 真实 HTTP（发布值）

| id | w* | conc | http_rps_cpu | p99_cpu | http_rps_gpu | p99_gpu | eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| yolov8s | 8 | 32 | 276.40 | 163.55 | 297.85 | 139.09 | yes |
| yolov8n | 4 | 16 | 293.54 | 80.70 | 311.63 | 80.04 | yes |
| yolov8l | 2 | 8 | 104.26 | 82.41 | 77.73 | 106.93 | yes |
| yolov8x | 2 | 8 | 108.81 | 75.74 | 92.55 | 87.77 | yes |
| yolov6s | 4 | 16 | 240.89 | 102.14 | 158.99 | 102.29 | yes |
| yolov5l | 4 | 16 | 147.90 | 115.19 | 104.75 | 153.24 | yes |
| yolov7 | 4 | 16 | 154.77 | 1002.51 | 158.93 | 150.08 | yes |
| yolov7x | 2 | 8 | 93.30 | 90.26 | 74.80 | 108.03 | yes |
| nanodet_1x5 | 4 | 16 | 220.31 | 106.95 | 204.15 | 83.42 | yes |
| nanodet_416 | 4 | 16 | 223.31 | 103.71 | 216.97 | 91.98 | yes |
| mobilenetv2 | 4 | 16 | 1004.37 | 30.72 | 567.72 | 29.38 | yes |
| resnet | 2 | 8 | 285.83 | 28.41 | 231.37 | 34.90 | yes |
| densenet | 2 | 8 | 139.28 | 123.64 | 122.95 | 128.79 | yes |
| superpoint | 4 | 16 | 1546.24 | 17.46 | 1011.81 | 26.64 | yes |
| dbnet | 4 | 16 | 141.31 | 140.16 | 272.86 | 59.41 | yes |
| libface_320 | 2 | 8 | 491.17 | 23.35 | 503.88 | 19.81 | yes |
| libface_640 | 4 | 16 | 208.64 | 126.64 | 358.99 | 71.75 | yes |
| centerface | 4 | 16 | 604.14 | 42.36 | 462.30 | 53.83 | yes |
| pphuman_mobile | 4 | 16 | 435.63 | 55.13 | 410.07 | 56.28 | yes |
| pphuman_lite | 4 | 16 | 405.52 | 58.54 | 373.65 | 62.67 | yes |
| pphuman_server | 4 | 16 | 145.73 | 184.65 | 144.97 | 184.90 | yes |
| bisenetv2 | 4 | 16 | 42.20 | 417.57 | 53.19 | 341.73 | yes |
| modnet | 8 | 32 | 20.74 | 1676.66 | 22.79 | 1599.44 | yes |
| ppmatting_512 | 4 | 16 | 16.35 | 1048.16 | 18.34 | 989.44 | yes |
| ppmatting_1024 | 8 | 32 | 17.48 | 1898.51 | 17.96 | 1812.86 | yes |
| ppmatting_resnet34 | 8 | 32 | 6.91 | 4162.50 | 17.70 | 1858.70 | yes |
| ppmatting_v2 | 4 | 16 | 17.38 | 1026.33 | 18.79 | 918.76 | yes |
| attentive_gan | 2 | 8 | 31.58 | 253.41 | 30.54 | 262.46 | yes |
| enlightengan | 4 | 16 | 137.51 | 122.73 | 119.77 | 141.87 | yes |
| realesrgan | 8 | 32 | 6.03 | 4442.49 | 6.01 | 4356.35 | yes |
| dinov2_vits14 | 2 | 8 | 190.79 | 99.35 | 153.79 | 104.49 | yes |
| dinov2_vitb14 | 2 | 8 | 271.23 | 34.19 | 208.97 | 40.64 | yes |
| dinov2_vitl14 | 2 | 8 | 95.47 | 86.89 | 86.85 | 156.24 | yes |
| depth_vits14 | 4 | 16 | 183.65 | 104.02 | 172.25 | 97.00 | yes |
| depth_vitb14 | 2 | 8 | 79.72 | 107.33 | 75.04 | 107.53 | yes |
| depth_vitl14 | 2 | 8 | 25.67 | 308.89 | 25.21 | 314.29 | yes |
| metric3d_512 | 2 | 8 | 20.13 | 408.67 | 19.42 | 422.72 | yes |
| metric3d_1088 | 2 | 8 | 6.07 | 1228.09 | 6.01 | 1244.88 | yes |
| hrnet | 2 | 8 | 16.80 | 479.79 | 16.31 | 494.31 | yes |
| msocrnet | — | — | — | — | — | — | — |

## Gaps（失败/非 HTTP，来自 convert JSON）

- `msocrnet`: convert+profile only (`http=False`)
