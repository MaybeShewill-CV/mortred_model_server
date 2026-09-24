# ONNX interchange export

Hugging Face (`MaybeShewill-CV/mortred_model_server`) is the **ONNX interchange**
store, not a portable-engine store. Per-id contract, source, and license live in
[`conf/onnx_sources.json`](../conf/onnx_sources.json). Bytes and sha256 live in
[`conf/weights_manifest.json`](../conf/weights_manifest.json).

This page is the export/adopt rulebook and the remaining-gap list. Serving and
fetch: [deployment.md](deployment.md) §9. Out of scope: [unsupported-boundaries.md](unsupported-boundaries.md).

## Policy

- Dual files on disk: `{stem}.static_bs1.onnx` (batch=1) and `{stem}.dyn.onnx`
  (symbolic `batch` on dim0). Spatial dims stay as the product graph needs them
  (frozen H/W, or `height`/`width` when the C++ path is dynamic).
- Keep the original `{stem}.onnx` when it already existed (product toml,
  `trt_engines.json`, hosted HF names).
- Do **not** reverse `.mnn` / `.model` / `.engine` into ONNX.
- Product `conf/model/**/*.toml` stay `type=mnn` or `type=tensorrt`, except:
  - already-ONNX product ids: diffusion (DDPM / DDIM / CLS_COND_DDIM / LDM),
    MSOCRNET, SAM decoders
  - **LIBFACE** after the YuNet 2026may cutover (`type=onnx`, 12-head decode)
- CI-only overlays under `conf/ci/*_onnx_hosted.toml` prove ORT CPU. Do not
  point `conf/server` at those files.
- HF upload is deferred until every product graph that should ship is ready.
  `p1_upload` in `onnx_sources.json` stays false for the P2 locals.

## How to export or adopt

From the repo root. Prefer official checkpoints / official ONNX, then
`write_pair_from_static`.

| Job | Script |
|---|---|
| Copy a static graph into the dual pair (batch dim only) | `scripts/export_onnx/adopt_official_onnx.py` |
| Same, plus Reshape `[1,-1,…]` → `[0,-1,…]` on the dyn file | `scripts/export_onnx/adopt_legacy_pairs.py` (`deep: True`) |
| Freeze rank-4 H/W then pair | `scripts/export_onnx/freeze_spatial.py` |
| Dump IO | `scripts/export_onnx/dump_io.py` |
| Stamp `conf/onnx_sources.json` | `scripts/export_onnx/update_onnx_sources_p2.py` |
| Rewrite CI overlays to `static_bs1` | `scripts/export_onnx/write_ci_overlays.py` |
| Shared helpers | `scripts/export_onnx/common.py` (`write_pair_from_static`, `set_batch_dim`, `apply_reshape_batch_passthrough`) |

Family exporters (from-scratch): `export_classification.py`, `export_bisenetv2.py`,
`export_attentive_gan.py`, `export_enlighten.py`, `export_ppmatting.py`,
`export_pphuman.py`, `export_realesrgan.py`, `export_superpoint.py`,
`export_dinov2_timm.py`, `export_clip.py`, plus YOLO / NanoDet / DBNet adopt
paths recorded in `onnx_sources.json` `onnx_source.script`.

LibFace check (ORT + the C++ YuNet decoder):

```bash
python3 scripts/verify_libface_yunet.py
```

## Inventory (2026-09-23)

Counted from `conf/onnx_sources.json` against files under `weights/`.
**50 graphs: 48 have a dual pair on disk (36 `local` + 12 `hosted`); 2 blocked;
0 missing.**

HTTP product graphs with dual files include classification (MobileNetV2 / ResNet /
DenseNet / DINOv2), YOLO v5/v6/v7/v8s, NanoDet, **LibFace YuNet**, CenterFace,
DBNet, BiSeNetV2, PP-HumanSeg v2-mobile, HRNet, MODNet, PP-Matting 512,
EnlightenGAN, AttentiveGAN, Real-ESRGAN, SuperPoint, Depth Anything, Metric3D,
SAM AMG (MobileSAM encoder + sm86 decoder), and the diffusion UNets / KL decoder.

Bench / sibling dual files (not HTTP product rows): LightGlue extractor/matcher,
OpenAI CLIP visual/textual, MSOCRNET fp16, FastSAM s/x, nano_sam, vit_l **decoder**,
YOLOv8 n/l/x, YOLOv7x, PP-HumanSeg lite/server, PP-Matting 1024 / resnet34 / v2.

### Still open

| Graph | Status | Why |
|---|---|---|
| **SAM_PREDICTOR vit_l encoder** | blocked | Decoder already has dual ONNX. Encoder is `sam_vit_l_encoder.mnn` (~1.2 GB). No official encoder ONNX adopted. Do not reverse the MNN. |
| **RTDETR** | blocked | Scaffold only (`MODEL_NOT_IMPLEMENTED`). Not in the HTTP catalog. Skip until registered. |

No remaining `onnx_status=missing` rows. The previous LibFace loc/conf block is
closed: C++ reads YuNet `cls_*` / `obj_*` / `bbox_*` / `kps_*`; interchange is
`face_detection_yunet_2026may.{static_bs1,dyn}.onnx` (original 2026may kept).

### Honest caveats (not “missing”)

- SAM AMG **sm61** engine has no matching ONNX; interchange uses the **sm86**
  decoder pair.
- Classification trio is torchvision ImageNet-1K V1 + NHWC wrap; C++ still
  applies caffe mean/std (golden may drift vs caffe MNN).
- DBNet used PaddleOCR `ch_ppocr_server_v2.0_det_infer` frozen 544×960; backbone
  may differ from `db_model_large.mnn`.
- DINOv2 pairs interpolate official 518 pos-embed to 224.
- CLIP vocab (`bpe_simple_vocab_16e6.txt`) is not ONNX.
- LLM `weights/llm/embedding/` is out of this CV catalog.

## Adding a new interchange graph

1. Export or adopt so both `{stem}.static_bs1.onnx` and `{stem}.dyn.onnx` exist
   and match the C++ IO names/shapes in `onnx_sources.json` `contract`.
2. Do not change product `type` to `onnx` unless the runtime weight is gone and
   the detector was rewritten for that graph (LibFace is the documented case).
3. Stamp `onnx_source` via `update_onnx_sources_p2.py` (or the legacy stamper).
4. Add a `conf/ci/*_onnx_hosted.toml` overlay if the id should prove ORT CPU.
5. Leave HF upload off until the remaining blocked graphs are adopted or waived.
