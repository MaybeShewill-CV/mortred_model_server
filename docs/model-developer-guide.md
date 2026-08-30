# Model Developer Guide

This is the task-oriented companion to
[how_to_add_new_model.md](how_to_add_new_model.md), which explains the layer
itself. Read that page first for the architecture; use this one when you
actually need to get something done.

Every command below is runnable from the repository root.

---

## Path 1: add a classification model in ten minutes

```bash
# 1. see which tasks the scaffolder supports
python scripts/new_model.py --list-tasks

# 2. preview the files it would create, without writing anything
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet \
    --backend mnn --dry-run

# 3. generate them
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet \
    --backend mnn
```

That produces five files:

| File | What you fill in |
|---|---|
| `src/models/classification/efficient_net.h` | members you need |
| `src/models/classification/efficient_net.inl` | `preprocess` / `postprocess` / `on_init` |
| `conf/model/classification/efficientnet/efficientnet_config.toml` | weight path, params |
| `test/efficient_net_output_contract_unittest.cc` | the real output shape |
| `docs/models/classification/efficientnet.md` | status and TODOs |

At this point the model **compiles** and every hook returns
`MODEL_NOT_IMPLEMENTED`, so it cannot be served by accident.

Fill in the three hooks - copy the shape of
[mobilenetv2.inl](../src/models/classification/mobilenetv2.inl):

```cpp
// on_init: derive the input size from the session, do not hardcode it
const auto info = SessionIoValidator(session())
                      .input().f32().rank(4).nhwc().channels(3).static_shape().validate();

// preprocess: one call, no hand-written resize/normalize/memcpy
return ImagePipeline(image)
    .resize(_m_input_size)
    .bgr_to_rgb()
    .to_float()
    .scale(1.0f / 255.0f)
    .nhwc(session().inputs().front().name);

// postprocess: always go through OutputReader so a malformed tensor
// becomes MODEL_OUTPUT_CONTRACT_FAILED instead of a partial result
auto view = OutputReader(outputs, outputs.front().name)
                .f32().shape({1, -1}).finite().read();
```

Then paste the two snippets the scaffolder printed: the catalog row in
`src/factory/classification_task.h` and the test target in
`test/CMakeLists.txt`.

---

## Path 2: add a detection model in ten minutes

Same three commands with `--task object_detection`. Differences from
classification:

- The output contract is `std_object_detection_output` (or
  `std_face_detection_output` for faces - see below).
- Decode is model-specific and stays in your detector. Reuse what is shared:
  [`detector_common.h`](../src/models/segment_anything/../object_detection/detector_common.h)
  already handles request-geometry scaling, named f32 output validation,
  per-class NMS, top-k and category filling.
- **Two output contracts means two catalogs.** `object_detection` keeps
  `catalog()` (generic boxes) and `face_catalog()` (boxes + landmarks). Do not
  merge them into one type-erased list - see
  [obj_detection_task.h](../src/factory/obj_detection_task.h).

Reference: [yolov8_detector.inl](../src/models/object_detection/yolov8_detector.inl).

---

## Path 3: write an output contract

Every model that decodes tensors must reject malformed output with
`MODEL_OUTPUT_CONTRACT_FAILED` instead of producing a half-decoded result.
The rejection matrix is generated for you:

```cpp
// test/<file>_output_contract_unittest.cc
POSTPROCESS_CONTRACT_TEST(EfficientNet, mat_input,
                          std_classification_output, "output", 1, 1000);
```

One line buys seven tests, each independently filterable:

```
rejects_missing_output   rejects_wrong_dtype    rejects_wrong_rank
rejects_wrong_shape      rejects_short_buffer   rejects_nan
rejects_inf
```

While the model is still a scaffold every variant fails with
`MODEL_NOT_IMPLEMENTED`, which the harness accepts as an explicit rejection -
so the macro works before your decoder exists.

When you have a real decoder, replace the placeholder shape with the actual
one (it must be concrete, not dynamic) and add a fixture that asserts decoded
values. Reference:
[object_detection_output_contract_unittest.cc](../test/object_detection_output_contract_unittest.cc).

For the model side, use `OutputReader` rather than building a
`TensorContract` by hand:

```cpp
auto view = OutputReader(outputs, "output")
                .f32()          // dtype
                .shape({1, -1}) // rank + shape, -1 = any
                .finite()       // reject NaN / Inf
                .read();
if (!view.ok()) {
    return view.status;
}
```

---

## Path 4: add a golden case

One line in [model_golden_test.cc](../test/model_golden_test.cc):

```cpp
GOLDEN_CLASSIFICATION_CASE(efficientnet_classification,
    "conf/model/classification/efficientnet/efficientnet_config.toml",
    "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
    jinq::factory::classification::create_efficientnet_classifier,
    std_classification_output);
```

Available macros, one per output contract:

| Macro | Output |
|---|---|
| `GOLDEN_CLASSIFICATION_CASE` | class_id / category / scores |
| `GOLDEN_OBJECT_DETECTION_CASE` | boxes |
| `GOLDEN_FACE_DETECTION_CASE` | boxes + landmarks |
| `GOLDEN_SCENE_SEGMENTATION_CASE` | segmentation mask |
| `GOLDEN_MATTING_CASE` | alpha mask |
| `GOLDEN_ENHANCEMENT_CASE` | enhanced image |
| `GOLDEN_TEXT_REGION_CASE` | OCR text regions |
| `GOLDEN_KEYPOINT_CASE` | feature points |
| `GOLDEN_RAW_MAT_CASE` | a bare `cv::Mat` |

Generate the baseline, then confirm it passes without the env var:

```bash
LD_LIBRARY_PATH=<build>/lib:3rd_party/libs \
MORTRED_UPDATE_GOLDEN=1 <build>/bin/model_golden_test \
    --gtest_filter='model_golden.efficientnet_classification'

LD_LIBRARY_PATH=<build>/lib:3rd_party/libs \
    <build>/bin/model_golden_test \
    --gtest_filter='model_golden.efficientnet_classification'
```

Weights missing on this machine? The case skips itself - that is expected and
lets the suite run on CPU-only dev boxes.

Cases that are genuinely different stay hand-written rather than forced
through a macro: the three batch-equivalence cases, SAM prompt / AMG, and the
CLIP two-tower case. Their value is exactly the flow a macro would hide.

---

## Path 5: prove you changed nothing

A green test suite only proves the model still works. Identical hashes prove
the numbers did not change by a single bit:

```bash
python scripts/golden_drift_check.py --record   # before a migration
python scripts/golden_drift_check.py --check    # after it
```

`test/golden_baseline.json` is the committed record. Any difference in the 27
case names, their declaration order, or the 25 baseline hashes fails the
check. This guard caught two real regressions during the phase 6 migration:
a dropped `/255` normalisation in nanodet and a removed BGR-to-RGB conversion
in real_esrgan that the then-current grayscale golden image could not detect.

One blind spot to know about: the guard protects the baseline files, and the
tests protect the tolerance. Neither catches a regression the test input is
insensitive to. When you add a golden case, prefer a **colour** input - a
grayscale image cannot detect a channel-order swap because R == G == B there.

---

## Path 6: debug a shape or dtype error

**Read the error literally.** Since phase 2 the catalog test and since phase 7
`SessionIoValidator` name the offending engine, direction and tensor:

```
visual input [input]: unexpected session io dtype: input:i32[1,3,8,8]
sam encoder session is null
```

The first says engine `visual`, its `input`, tensor `input`, expected f32 and
got i32. The second says `sessions()` has no entry named `encoder`.

Common causes:

| Symptom | Likely cause |
|---|---|
| `unexpected session io dtype` | config `type` does not match the model file |
| `rank` / `shape` mismatch | you packed NHWC into an NCHW model, or vice versa |
| `expected static [N,H,W,3]` | the engine was exported with a dynamic batch |
| `session is null` | the name passed to `session("...")` is not in `sessions()` |
| `model section [...] missing` | the TOML section name differs from the one passed to the constructor |

To inspect what a session actually exposes:

```cpp
for (const auto &info : session().inputs()) {
    LOG(INFO) << "input " << info.to_string();
}
```

`TensorInfo::to_string()` prints name, dtype, shape and whether it is dynamic.
To assert the contract instead of printing it, use `SessionIoValidator` -
see `on_init` in any migrated model.

---

## When not to use the shared helpers

The helpers have a defined scope. Going outside it is a deliberate decision,
not a failure:

| Helper | Scope | Deliberately not |
|---|---|---|
| `ImagePipeline` | resize, crop, colour convert, normalise, pack to NCHW/NHWC | keep-ratio resize with padding (depth, fastsam), non-image inputs |
| `OutputReader` | named f32 output contract | int32 token tensors, multi-output decode ordering |
| `SessionIoValidator` | one named input / output pair | optional inputs, alternative outputs |
| `GOLDEN_*_CASE` | standard seven-step single-image case | batch equivalence, multi-session flows |
| `MultiSessionModel` | fixed set of distinct engines addressed by name | concurrent session pools, model composition |

Two models kept hand-written code for exactly these reasons and the reasoning
is recorded:

- **SamAutoMaskGenerator** - a pool of N identical sessions recycled through a
  concurrent queue and driven by Workflow parallel series. That is a pooled
  executor, not a multi-engine lifecycle.
- **LDM** - DDIM and DDPM share one `shared_ptr` to the same latent UNet, and
  its sub-models are themselves complete `BackendCvModel`s loaded from two
  external TOML files. That is a composition of models, not one multi-engine
  model.

If your model falls outside a helper's scope, keep the hand-written code and
add a comment saying why. Do not bend the helper until it fits.

---

## Current state and known gaps

Status after phase 7:

- Phases 0-7 of the
  [P4 plan](model-developer-experience-p4.zh-cn.md) are complete; phase 8
  (this document) is the last one.
- 10 of 11 model families are on the runtime toolkit. Hand-written
  `std::memcpy` went from 30 to 3 and `ImagePipeline` usage from 3 to 31.
- Three families have **no golden case**: depth, lightglue and LDM. Changes to
  those models are validated by compilation and the full suite only - there is
  no fingerprint guard. Closing that gap is worth doing before the next change
  to any of them.
- `enlightengan` (dual tensor output, custom luma, alpha extraction) is
  deliberately not on `ImagePipeline`.
