# Model Developer Guide

Single entry for **adding and hardening a CV model** in this repo. For mounting
a finished model on HTTP (`conf/server`, OpenAPI, consistency), see
[how_to_add_new_server.md](how_to_add_new_server.md). Contract review checklist:
[model-contract-governance.md](model-contract-governance.md).

Every command below is runnable from the repository root.

---

## Layer overview

All CV models inherit
[`BackendCvModel<INPUT, OUTPUT>`](../src/models/backend/backend_cv_model.h):

```text
init:      parse [SECTION.backend] -> create InferenceSession -> on_init([SECTION.params])
run_impl:  prepare_inputs -> session.run -> postprocess(context)
```

A standard single-image model implements **preprocess** (cv::Mat → named
tensors) and **postprocess** (named tensors + request geometry → task output).
Backend plumbing (MNN / ORT / TensorRT sessions, dtype & shape checks, copies)
lives under [`src/models/backend/`](../src/models/backend/) and is not repeated
per model.

Prefer the [runtime toolkit](../src/models/backend/model_runtime.h):
`ImagePipeline`, `OutputReader`, `SessionIoValidator`. Reference models:
[mobilenetv2](../src/models/classification/mobilenetv2.inl) (MNN),
[yolov8_detector](../src/models/object_detection/yolov8_detector.inl) (TRT),
[ddpm_unet](../src/models/diffusion/ddpm_unet.inl) (ORT, non-image
`prepare_inputs`).

IO types: [`src/models/io/`](../src/models/io) (prefer task `std_*_output`).
Request-scoped geometry belongs in `InferenceContext`, never in model members
(dynamic batch). Malformed tensors must return `MODEL_OUTPUT_CONTRACT_FAILED`
via `OutputReader` / [`f32_output.h`](../src/models/backend/f32_output.h).

Skeleton:

```cpp
template<typename INPUT, typename OUTPUT>
class MyModel : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    MyModel() : jinq::models::BackendCvModel<INPUT, OUTPUT>("MY_MODEL") {}
  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;
    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const jinq::models::backend::InferenceContext& context,
        OUTPUT& output) override;
    jinq::common::StatusCode on_init(const toml::table& params) override;  // optional
};
```

Config shape (full keys: [about_model_configuration.md](about_model_configuration.md)):

```toml
[MY_MODEL]
[MY_MODEL.backend]
type = "mnn"                # mnn | onnx | tensorrt
model_file_path = "../weights/my_model/model.mnn"
device = "gpu"
threads = 4

[MY_MODEL.params]
score_threshold = 0.25
```

Catalog: one `CvModelEntry` in `src/factory/<task>_task.h` (see Path 1).
HTTP-only families use [`cv_catalog.h`](../src/factory/cv_catalog.h);
benchmark-only families use [`model_catalog.h`](../src/factory/model_catalog.h)
(CLIP / SAM predictor / FastSAM — no HTTP by product lock). Multiple output
contracts → multiple typed catalogs (`catalog()` / `face_catalog()` in
[`obj_detection_task.h`](../src/factory/obj_detection_task.h)).

---

## Path 1: add a classification model in ten minutes

```bash
python scripts/new_model.py --list-tasks
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet \
    --backend mnn --dry-run
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet \
    --backend mnn
```

Produces header / `.inl` / TOML / output-contract unittest (and prints catalog +
CMake snippets it does not auto-apply). Unimplemented hooks return
`MODEL_NOT_IMPLEMENTED` so a half-finished model cannot be served.
`src/models/object_detection/rtdetr_detector.*` is the checked-in scaffold canary.

Fill hooks like [mobilenetv2.inl](../src/models/classification/mobilenetv2.inl):

```cpp
const auto info = SessionIoValidator(session())
                      .input().f32().rank(4).nhwc().channels(3).static_shape().validate();

return ImagePipeline(image)
    .resize(_m_input_size)
    .bgr_to_rgb()
    .to_float()
    .scale(1.0f / 255.0f)
    .nhwc(session().inputs().front().name);

auto view = OutputReader(outputs, outputs.front().name)
                .f32().shape({1, -1}).finite().read();
```

Paste the scaffolder's catalog row into `src/factory/classification_task.h` and
the test target into `test/CMakeLists.txt`.

---

## Path 2: add a detection model in ten minutes

Same commands with `--task object_detection`. Differences:

- Output: `std_object_detection_output` or `std_face_detection_output`.
- Reuse [`detector_common.h`](../src/models/object_detection/detector_common.h)
  for validation / NMS / top-k. **Geometry is not one helper:** Ultralytics YOLO
  needs `ImagePipeline::letterbox` + `unmap_letterbox_bbox`; stretch
  `GeometryScale` is for NanoDet / CenterFace / LibFace.
- Two contracts → `catalog()` and `face_catalog()` — do not type-erase them.

Reference: [yolov8_detector.inl](../src/models/object_detection/yolov8_detector.inl).

---

## Path 3: write an output contract

```cpp
POSTPROCESS_CONTRACT_TEST(EfficientNet, mat_input,
                          std_classification_output, "output", 1, 1000);
```

One macro → seven filterable tests (`rejects_missing_output`, wrong dtype/rank/shape,
short buffer, nan, inf). Scaffold models may still return `MODEL_NOT_IMPLEMENTED`
(harness accepts that as explicit rejection).

```cpp
auto view = OutputReader(outputs, "output")
                .f32().shape({1, -1}).finite().read();
if (!view.ok()) {
    return view.status;
}
```

---

## Path 4: add a golden case

Hosted CI fail-closes only a small MNN CPU set; GPU smoke is a maintainer gate.
See [ci-golden-regression.md](ci-golden-regression.md).

```cpp
GOLDEN_CLASSIFICATION_CASE(efficientnet_classification,
    "conf/model/classification/efficientnet/efficientnet_config.toml",
    "demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG",
    jinq::factory::classification::create_efficientnet_classifier,
    std_classification_output);
```

Macros: `GOLDEN_CLASSIFICATION_CASE`, `GOLDEN_OBJECT_DETECTION_CASE`,
`GOLDEN_FACE_DETECTION_CASE`, `GOLDEN_SCENE_SEGMENTATION_CASE`,
`GOLDEN_MATTING_CASE`, `GOLDEN_ENHANCEMENT_CASE`, `GOLDEN_TEXT_REGION_CASE`,
`GOLDEN_KEYPOINT_CASE`, `GOLDEN_RAW_MAT_CASE`.

```bash
LD_LIBRARY_PATH=<build>/lib:3rd_party/libs \
MORTRED_UPDATE_GOLDEN=1 <build>/bin/model_golden_test \
    --gtest_filter='model_golden.efficientnet_classification'
```

Missing weights → case skips (expected on CPU-only boxes).

---

## Path 5: prove you changed nothing

```bash
python scripts/golden_drift_check.py --record   # before a migration
python scripts/golden_drift_check.py --check    # after it
```

`test/golden_baseline.json` records case names and golden file hashes. Prefer
**colour** golden inputs so channel-order swaps are visible.

---

## Path 6: debug a shape or dtype error

Read the error literally (`SessionIoValidator` / catalog tests name engine,
direction, tensor). Common causes: config `type` vs file mismatch; NHWC/NCHW
pack wrong; dynamic batch vs static expect; `session("...")` name missing from
`sessions()`; TOML section ≠ constructor section string.

```cpp
for (const auto &info : session().inputs()) {
    LOG(INFO) << "input " << info.to_string();
}
```

---

## When not to use the shared helpers

| Helper | Scope | Deliberately not |
|---|---|---|
| `ImagePipeline` | resize, crop, colour, normalise, NCHW/NHWC | keep-ratio pad (depth, fastsam), non-image inputs |
| `OutputReader` | named f32 contract | int32 tokens, multi-output ordering |
| `SessionIoValidator` | one named IO pair | optional / alternative IO |
| `GOLDEN_*_CASE` | single-image seven-step | batch equivalence, multi-session flows |
| `MultiSessionModel` | fixed named engines | concurrent pools, model composition |

Examples kept hand-written on purpose: **SamAutoMaskGenerator** (session pool),
**LDM** (composed BackendCvModels). Comment why when you step outside a helper.

---

## Verify build

```bash
cmake --preset full && cmake --build --preset full
scripts/run_tests.sh build/full -R model_golden_test --output-on-failure
python3 scripts/check_consistency.py
```

---

## Known gaps

- Depth, LightGlue, and LDM still lack golden cases.
- `enlightengan` stays off `ImagePipeline` (dual tensor / custom luma / alpha).
