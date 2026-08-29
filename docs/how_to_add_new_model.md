# How To Add New Model (Unified Backend Layer)

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

Input types live in [model_io_define.h](../src/models/model_io_define.h).
`mat_input`, `file_input` and `base64_input` are loadable images and work with
the default `prepare_inputs` path. Task default outputs (`std_*_output`) are the
recommended choice.

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

## Step 4: Register in the factory

Add a `create_my_model` function in the task factory and a server spec if the
model should be served over HTTP. The `BaseAiModel` contract is unchanged, so
apps and servers need no backend knowledge.

## Step 5: Verify

```bash
cmake --preset full && cmake --build --preset full
scripts/run_tests.sh build/full -R model_golden_test --output-on-failure
```

Add a golden case with real weights to
[test/model_golden_test.cc](../test/model_golden_test.cc) (tolerances are per
task: score/box-IoU for detection, fingerprint diff for dense outputs).
