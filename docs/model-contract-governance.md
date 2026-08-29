# Model Contract Governance

This page records the contract rules applied across the model layer. It is the
review checklist for new models and for changes to existing model families.

## Request geometry

`InferenceContext` is request-scoped and carries two sizes:

| Field | Meaning |
|---|---|
| `source_size` | Size of the image supplied by the user |
| `network_size` | Concrete tensor size used for this inference |

Spatial models must not derive either size from mutable model members in
`postprocess`. Use [`request_geometry.h`](../src/models/backend/request_geometry.h):

- `make_geometry_scale` validates both sizes before coordinate mapping.
- `scale_bbox` / `scale_point` map network coordinates to source coordinates.
- `validated_source_size` validates the destination for dense output resizing.

This policy is applied to object detection, scene segmentation, OCR, matting,
enhancement, SuperPoint, DepthAnything, Metric3D and FastSAM. Multi-session
models which currently process requests synchronously were audited with the
same rule; their request geometry remains local to one `run_sessions` call.
Latent-space diffusion models and CLIP encoders have no source-image geometry,
but their f32 outputs use the same output-contract boundary.

## Output contracts

Floating-point outputs are validated through
[`f32_output.h`](../src/models/backend/f32_output.h):

- missing output -> `MODEL_EMPTY_OUTPUT`
- dtype/rank/shape/buffer error -> `MODEL_OUTPUT_CONTRACT_FAILED`
- non-finite f32 value -> `MODEL_OUTPUT_CONTRACT_FAILED`

`model_output_contract_unittest` covers the shared helper and representative
models from classification, segmentation, OCR, matting and enhancement.
`object_detection_output_contract_unittest` covers detector-specific layouts.

Integer argmax outputs use `validate_output_tensor` directly with the exact
`DType::I32` / `DType::I64` layout; finite-value checks apply to f32 outputs.

## Configuration limits

The repository-wide image defaults protect services from oversized decoded
inputs:

```toml
max_image_pixels = 16777216
max_image_side = 8192
```

A model may explicitly raise `max_image_pixels` when its normal user input is
a full-size camera image. The override must be documented in that model
configuration. MODNet and PPMatting, for example, accept 24 MP portrait photos.

## Golden regression

Contract tests reject malformed tensors before task decoding. Golden tests
then lock the numerical behavior of valid models:

- generate goldens on the reference GPU environment;
- rerun normally at least three times after generation;
- commit golden data separately from logic changes;
- never refresh goldens merely to make a regression pass.

The local full GPU regression currently executes the complete committed
`model_golden_test` suite. Weight-free environments skip weighted cases by
design rather than reporting them as passed.
