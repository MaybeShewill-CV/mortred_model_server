# How To Add New Model (Unified Backend Layer)

## Step 0: Scaffold the boring parts (recommended)

```bash
python scripts/new_model.py --list-tasks
python scripts/new_model.py --task object_detection --name rtdetr \
    --class RtdetrDetector --backend tensorrt --dry-run
python scripts/new_model.py --task object_detection --name rtdetr \
    --class RtdetrDetector --backend tensorrt
```

This generates the header, the `.inl`, the TOML config, an output contract
test and a README, and prints the registration snippets it deliberately does
not apply for you (catalog entry, test target, golden case). The scaffold
compiles immediately; every unimplemented hook returns
`MODEL_NOT_IMPLEMENTED`, so a half-finished model can never be served by
accident. `src/models/object_detection/rtdetr_detector.*` is a checked-in
example of exactly this output and doubles as the canary that keeps the
templates compilable.

The rest of this document explains what the scaffold leaves for you to write.

All CV models now inherit from
[`jinq::models::BackendCvModel<INPUT, OUTPUT>`](../src/models/backend/backend_cv_model.h).
The base class implements the full lifecycle:

```text
init:      parse [SECTION.backend] -> create InferenceSession -> on_init([SECTION.params])
run_impl:  prepare_inputs -> session.run -> postprocess(context)
```

A standard single-image model only implements **preprocess** (cv::Mat to named
tensors) and **postprocess** (named tensors plus request geometry to task
output). Backend plumbing
(MNN / ONNX Runtime / TensorRT session management, dtype & shape validation,
dynamic shape handling, host/device copies) lives in
[`src/models/backend/`](../src/models/backend/) and is never repeated per model.

## Step 1: Pick the IO types

IO types live in [src/models/io/](../src/models/io), one header per task.
`common_input.h` holds the shared inputs (`mat_input`, `file_input`,
`base64_input`, `pair_mat_input`) and each task header holds its own
`std_*_output`. Include only the task header you need - the old
[model_io_define.h](../src/models/model_io_define.h) still works but is a
compatibility aggregate that pulls in every task. The loadable image inputs
work with the default `prepare_inputs` path; task default outputs
(`std_*_output`) are the recommended choice.

## Step 2: Write the model class

Reference implementations (read these first):

- [mobilenetv2.h](../src/models/classification/mobilenetv2.h) / [.inl](../src/models/classification/mobilenetv2.inl) — MNN single image classification
- [yolov8_detector.h](../src/models/object_detection/yolov8_detector.h) / [.inl](../src/models/object_detection/yolov8_detector.inl) — TensorRT detection with decode + NMS
- [ddpm_unet.h](../src/models/diffusion/ddpm_unet.h) / [.inl](../src/models/diffusion/ddpm_unet.inl) — ONNX Runtime with non-image inputs (`prepare_inputs` override)

```cpp
template<typename INPUT, typename OUTPUT>
class MyModel : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    MyModel() : jinq::models::BackendCvModel<INPUT, OUTPUT>("MY_MODEL") {}

  private:
    // image -> named input tensors (required for image models)
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    // named output tensors -> task output
    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const jinq::models::backend::InferenceContext& context,
        OUTPUT& output) override;

    // optional: read model specific keys from [MY_MODEL.params]
    jinq::common::StatusCode on_init(const toml::table& params) override;
};
```

Notes:

- The constructor passes the config section name (`"MY_MODEL"`).
- Prefer the [runtime toolkit](../src/models/backend/model_runtime.h) over
  hand-writing these steps. `ImagePipeline` covers resize / crop / colour
  conversion / normalisation and packs to NCHW or NHWC in one call;
  `OutputReader` replaces hand-built output contracts;
  `SessionIoValidator` replaces hand-written session shape walks. Every
  migrated model under [src/models/](../src/models) is an example.
- `session().inputs()` / `session().outputs()` expose
  `TensorInfo{name, dtype, shape, dynamic}`; derive the network input size from
  it instead of hardcoding.
- Emit tensors with `backend::Tensor::make<float>(shape)`; the shape must be
  concrete and match the layout the model file expects (nhwc for MNN
  TENSORFLOW-style exports with `input_layout = "nhwc"`, nchw for
  TensorRT / ONNX models).
- Multi-output models pick tensors by `name` (see `find_output` in
  [tensor_contract.h](../src/models/backend/tensor_contract.h)).
- Object detection models reuse
  [`detector_common.h`](../src/models/object_detection/detector_common.h) for
  request-geometry scaling, named f32 output validation, NMS/top-k/category
  finalization, and NCHW packing. Keep model-specific decode logic in the
  detector instead of moving it behind another base class.
- Non-image inputs (token ids, latent vectors, image pairs) override
  `prepare_inputs` instead of `preprocess`.
- Request-scoped data (source image size, network size, crop geometry) must be
  carried in `InferenceContext`; never store it in a model member because one
  worker may be processing a dynamic batch.
- Dense image outputs (masks, alpha, depth and enhanced images) are resized to
  `context.source_size` only after the source geometry has been validated.
  Coordinate-producing models use the shared request-geometry helpers instead
  of dividing by an input-size member.
- Validate backend outputs through
  [`f32_output.h`](../src/models/backend/f32_output.h) before decoding. A
  malformed tensor must return `MODEL_OUTPUT_CONTRACT_FAILED`, not produce a
  partially decoded task result.
- Multi-engine models (encoder + decoder) configure `<key>_backend` sub-tables,
  build extra sessions with `make_session("<key>_backend")` and orchestrate
  them in the `run_sessions` override.
- If the engines are a fixed set of distinct sessions addressed by name,
  inherit [`MultiSessionModel`](../src/models/backend/multi_session_model.h)
  and declare them through `sessions()` instead of hand-writing the
  create / validate / reset sequence. It deliberately does **not**
  orchestrate the runs.

## Step 3: Write the config

```toml
[MY_MODEL]
[MY_MODEL.backend]
type = "mnn"                # mnn | onnx | tensorrt
model_file_path = "../weights/my_model/model.mnn"
device = "cuda"             # cpu | cuda
threads = 4
input_layout = "nhwc"       # mnn only: auto | nhwc | nchw

[MY_MODEL.params]
score_threshold = 0.25
```

See [about_model_configuration.md](about_model_configuration.md) for the full
key reference. Old `BACKEND_DICT` / `XXX_TRT` / `XXX_ONNX` / `XXX_MNN`
three-section configs are gone; use
[`scripts/migrate_model_config.py`](../scripts/migrate_model_config.py) to
migrate them (`--dry-run` first, `--check` in CI).

## Step 4: Register in the task catalog

Every task owns an explicit catalog in `src/factory/<task>_task.h`. Adding a
served model is now one row plus its creator - no hand-written server
registration lambda, no copied `CvServerSpec` block:

```cpp
// src/factory/my_task.h
template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_my_model(const std::string& name) {
    (void)name;
    return std::make_unique<MyModel<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::my_task::std_my_task_output;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry>& catalog() {
    static const std::vector<Entry> entries = {
        Entry{"MY_MODEL", "My model display name", "MY_MODEL_SERVER",
              &create_my_model<jinq::server::Base64Input, Output>,
              &jinq::server::response::fill_my_task},
    };
    return entries;
}
```

`factory::cv_catalog::create_server(catalog(), "MY_MODEL", server_name)` does
the rest: it registers the creator in `ServerFactory<BaseAiServer>` and builds
the generic `CvModelServer<Output>`.

Two shapes exist on purpose:

- [`factory/cv_catalog.h`](../src/factory/cv_catalog.h) - models mounted on the
  generic CV server (`CvModelEntry<OUTPUT>` carries the worker creator and the
  response filler).
- [`factory/model_catalog.h`](../src/factory/model_catalog.h) - model families
  consumed directly by benchmarks and in-process callers, which have no HTTP surface yet
  (CLIP, SAM predictor, FastSAM).

If a task has more than one output contract, split it into one typed catalog
per contract instead of type-erasing the list - see `catalog()` and
`face_catalog()` in [`obj_detection_task.h`](../src/factory/obj_detection_task.h).

`test/model_catalog_unittest.cc` fails the build when a catalog row references a
TOML section or `model_config_file_path` that does not exist, or when the model
or server section is duplicated across tasks.

## Step 5: Verify

```bash
cmake --preset full && cmake --build --preset full
scripts/run_tests.sh build/full -R model_golden_test --output-on-failure
```

Register it with one macro from
[model_golden_registry.h](../test/model_golden_registry.h) - see the
[developer guide](model-developer-guide.md) for the full list and the
two-command baseline workflow. Prove a refactor changed nothing with
[golden_drift_check.py](../scripts/golden_drift_check.py), and cover the
rejection matrix with
[POSTPROCESS_CONTRACT_TEST](../test/model_contract_test_util.h).

Add a golden case with real weights to
[test/model_golden_test.cc](../test/model_golden_test.cc) (tolerances are per
task: score/box-IoU for detection, fingerprint diff for dense outputs).
