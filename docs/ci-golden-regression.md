# GPU Golden Regression CI

This document explains the two self-hosted GPU jobs added in
`.github/workflows/ci.yml` (`gpu-pr-gate`, `gpu-nightly-full`): what they run,
what the runner needs on disk, and how to refresh the committed golden
expectations when an intentional numerical change lands.

## Why this exists

The golden tests in `test/model_golden_test.cc` silently `GTEST_SKIP()` whenever
weights/engines referenced by the model configs are missing. On GitHub's hosted
runners there is no NVIDIA GPU, so before these jobs existed the entire golden
zoo was skipped on every CI run - a numerical regression in preprocess / NMS /
decode shipped undetected until a human noticed wrong output.

These jobs close that gap with real CUDA + TensorRT execution on a
maintainer-owned runner.

## Self-hosted runner requirements

| Item | Requirement |
|---|---|
| OS | Ubuntu 22.04 LTS |
| NVIDIA driver | >= 535.x |
| CUDA / TensorRT | 11.8 / 8.6.1 - must match the versions vendored under `3rd_party/` |
| Runner labels | `self-hosted`, `X64`, `gpu` |
| Concurrency | Register exactly one runner process per machine; jobs then serialize naturally instead of fighting over one GPU |

## Persistent cache layout

Both jobs expect the caches at `/opt/mortred-cache/` (create once per runner):

```
/opt/mortred-cache/
`-- weights/                 # Full runtime asset tree mirroring the gitignored
                             # weights/ layout: ONNX sources AND the TRT engines
                             # matching this runner's GPU/TRT pair, e.g.
                             #   object_detection/yolov5/yolov5s.engine
                             #   classification/mobilenetv2/mobilenetv2.engine
```

CI symlinks `weights` into this cache. Engines are GPU-architecture- and
TRT-version-specific: when either changes on the runner (new driver, TRT
bump), wipe `weights/**/*.engine` and let one manual nightly run rebuild them
from ONNX before PR traffic resumes.

## PR smoke subset (Job H)

Six cross-family cases chosen so that one red fingerprint catches ~80% of
regressions while staying inside a 15-minute budget on warm cache:

| Case | Family | Output contract exercised |
|---|---|---|
| `yolov5_detection` | object detection | json boxes |
| `nanodet_detection` | object detection | anchor-free decode path |
| `centerface_detection` | face detection | landmark output contract |
| `dbnet_text_detection` | OCR | json text boxes |
| `mobilenetv2_classification` | classification | scores with `k_score_tol` |
| `fastsam_segmentation` | segmentation | fingerprint png with `k_fingerprint_diff` |

## Nightly full regression (Job I)

Runs all 24 committed golden cases plus `model_lifecycle_unittest`,
`backend_unittest` and `gateway_e2e_test`. Triggered by the daily schedule or
manually via `workflow_dispatch`.

## One-time baseline calibration

Golden expected values only form a meaningful regression guard once they were
generated on the same physical reference environment as the gate itself.
After setting up a fresh runner:

1. Let Job I run once via manual dispatch. If every case passes, the committed
   goldens already match the runner - done.
2. If cases fail with drift beyond tolerance, regenerate **on the runner**:

   ```bash
   MORTRED_UPDATE_GOLDEN=1 ./build-gpu/bin/model_golden_test
   ```

3. Commit the refreshed `test/golden/*` files in a dedicated data-only commit,
   separate from any logic change, so review shows exactly which numbers moved.
4. From then on the runner is the canonical reference point for "correct".

## Refreshing goldens after an intentional change

When a refactor legitimately shifts numbers (e.g. preprocessing change):

```bash
# on your GPU dev box or directly on the runner
MORTRED_UPDATE_GOLDEN=1 ./build-full/bin/model_golden_test
git add test/golden/
git commit -m "golden: refresh after <change>"
```

Update-mode runs `GTEST_SKIP` each case right after writing the new artifact,
so one pass regenerates everything reachable and leaves genuinely broken models
(crash/init failure) still failing loudly.
