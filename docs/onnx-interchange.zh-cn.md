# ONNX 交换格式导出

Hugging Face（`MaybeShewill-CV/mortred_model_server`）是 **ONNX 交换格式**仓库，
不是可移植 engine 仓库。每个产品 id 的源、IO 合同、许可证在
[`conf/onnx_sources.json`](../conf/onnx_sources.json)；字节与 sha256 在
[`conf/weights_manifest.json`](../conf/weights_manifest.json)。

本页是导出 / 收养规则和缺口清单。拉取与部署见 [deployment.zh-cn.md](deployment.zh-cn.md) §9。
不支持边界：[unsupported-boundaries.zh-cn.md](unsupported-boundaries.zh-cn.md)。

## 规则

- 磁盘上成对：`{stem}.static_bs1.onnx`（batch=1）和 `{stem}.dyn.onnx`（dim0 为符号
  `batch`）。空间维按产品图：该冻 H/W 就冻，C++ 走动态则保留 `height`/`width`。
- 原来就有的 `{stem}.onnx` 保留（产品 toml、`trt_engines.json`、已上 HF 的名字）。
- **禁止**从 `.mnn` / `.model` / `.engine` 反推 ONNX。
- 产品 `conf/model/**/*.toml` 保持 `type=mnn` 或 `type=tensorrt`，例外：
  - 本来就是 ONNX 的：diffusion（DDPM / DDIM / CLS_COND_DDIM / LDM）、MSOCRNET、SAM decoder
  - **LIBFACE** 已切到 YuNet 2026may（`type=onnx`，12 头解码）
- `conf/ci/*_onnx_hosted.toml` 只给 CI 证 ORT CPU，不要把 `conf/server` 指过去。
- HF 上传仍推迟，等该上的产品图都齐。`onnx_sources.json` 里 P2 本地文件的
  `p1_upload` 保持 false。

## 怎么导出 / 收养

在仓库根目录。优先官方 checkpoint / 官方 ONNX，再 `write_pair_from_static`。

| 用途 | 脚本 |
|---|---|
| 已有静态图写成 dual 对（只改 batch 维） | `scripts/export_onnx/adopt_official_onnx.py` |
| 同上，并把 dyn 里 Reshape `[1,-1,…]` 改成 `[0,-1,…]` | `scripts/export_onnx/adopt_legacy_pairs.py`（`deep: True`） |
| 先冻 rank-4 H/W 再成对 | `scripts/export_onnx/freeze_spatial.py` |
| 打印 IO | `scripts/export_onnx/dump_io.py` |
| 回写 `conf/onnx_sources.json` | `scripts/export_onnx/update_onnx_sources_p2.py` |
| 按 `static_bs1` 重写 CI overlay | `scripts/export_onnx/write_ci_overlays.py` |
| 公共函数 | `scripts/export_onnx/common.py`（`write_pair_from_static`、`set_batch_dim`、`apply_reshape_batch_passthrough`） |

分族导出脚本：`export_classification.py`、`export_bisenetv2.py`、
`export_attentive_gan.py`、`export_enlighten.py`、`export_ppmatting.py`、
`export_pphuman.py`、`export_realesrgan.py`、`export_superpoint.py`、
`export_dinov2_timm.py`、`export_clip.py`；YOLO / NanoDet / DBNet 的收养路径写在
`onnx_sources.json` 的 `onnx_source.script`。

LibFace（ORT + 与 C++ 相同的 YuNet 解码）：

```bash
python3 scripts/verify_libface_yunet.py
```

## 清单（2026-09-23）

对照 `conf/onnx_sources.json` 与 `weights/`。
**50 张图：48 张磁盘上已有 dual 对（36 `local` + 12 `hosted`）；2 张 blocked；0 张 missing。**

已有 dual 的 HTTP 产品图：分类（MobileNetV2 / ResNet / DenseNet / DINOv2）、
YOLO v5/v6/v7/v8s、NanoDet、**LibFace YuNet**、CenterFace、DBNet、BiSeNetV2、
PP-HumanSeg v2-mobile、HRNet、MODNet、PP-Matting 512、EnlightenGAN、AttentiveGAN、
Real-ESRGAN、SuperPoint、Depth Anything、Metric3D、SAM AMG（MobileSAM encoder + sm86 decoder）、
diffusion UNet / KL decoder。

Bench / 旁路 dual（不是 HTTP 产品行）：LightGlue extractor/matcher、OpenAI CLIP
visual/textual、MSOCRNET fp16、FastSAM s/x、nano_sam、vit_l **decoder**、
YOLOv8 n/l/x、YOLOv7x、PP-HumanSeg lite/server、PP-Matting 1024 / resnet34 / v2。

### 还缺

| 图 | 状态 | 原因 |
|---|---|---|
| **SAM_PREDICTOR vit_l encoder** | blocked | decoder 已有 dual ONNX。encoder 是 `sam_vit_l_encoder.mnn`（约 1.2 GB），没有收养官方 encoder ONNX。不要从 MNN 反推。 |
| **RTDETR** | blocked | 脚手架（`MODEL_NOT_IMPLEMENTED`），不在 HTTP catalog。注册之前跳过。 |

没有剩下的 `onnx_status=missing`。原先 LibFace loc/conf 阻塞已关掉：C++ 读 YuNet
`cls_*` / `obj_*` / `bbox_*` / `kps_*`；交换格式是
`face_detection_yunet_2026may.{static_bs1,dyn}.onnx`（原始 2026may 保留）。

### 不是缺口、但要知道

- SAM AMG **sm61** engine 没有对应 ONNX，交换格式用 **sm86** decoder 对。
- 分类三件套是 torchvision ImageNet-1K V1 + NHWC wrap，C++ 仍走 caffe mean/std（相对 caffe MNN，golden 可能漂）。
- DBNet 用的是 PaddleOCR `ch_ppocr_server_v2.0_det_infer` 冻 544×960，backbone 未必等于 `db_model_large.mnn`。
- DINOv2 把官方 518 pos-embed 插值到 224。
- CLIP 词表（`bpe_simple_vocab_16e6.txt`）不是 ONNX。
- `weights/llm/embedding/` 不在这份 CV catalog 里。

## 新增一张交换图

1. 导出或收养，保证 `{stem}.static_bs1.onnx` 和 `{stem}.dyn.onnx` 都在，且 IO 名字/形状对上 `onnx_sources.json` 的 `contract` 和 C++。
2. 不要把产品 `type` 改成 `onnx`，除非运行时权重已经换成这张图、并且检测器已按该图改过（文档里的例子是 LibFace）。
3. 用 `update_onnx_sources_p2.py`（或 legacy stamper）盖 `onnx_source`。
4. 若该 id 要在 CI 证 ORT CPU，加 `conf/ci/*_onnx_hosted.toml`。
5. 在剩下的 blocked 图收养或明确放弃之前，不要上传 HF。
