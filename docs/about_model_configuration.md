# About Model Configuration

All model configurations live in `conf/model/`. The unified backend layer uses
a two-table schema per model section:

```toml
[YOLOV8]                       # model section, one per model
[YOLOV8.backend]               # backend selection (shared by all models)
type = "tensorrt"              # mnn | onnx | tensorrt
model_file_path = "../weights/object_detection/yolov8/yolov8s.engine"
device = "cuda"                # cpu | cuda (default cpu)
device_id = 0                  # optional, cuda device index
threads = 4                    # cpu threads for mnn/onnx (default 4)
input_layout = "auto"          # mnn only: auto | nhwc | nchw
precision_mode = 0             # mnn only: BackendConfig::PrecisionMode
power_mode = 0                 # mnn only: BackendConfig::PowerMode
input_names = ["images"]       # optional, defaults to the model file io
output_names = ["output0"]     # optional, filters aux output nodes

[YOLOV8.params]                # model specific keys, consumed by on_init
model_score_threshold = 0.25
model_nms_threshold = 0.5
model_input_image_size = [640, 640]   # [height, width]; must match fixed model inputs
max_image_pixels = 16777216           # decoded-pixel safety limit
max_image_side = 8192
class_names = ['person', 'bicycle']
```

## Key reference

| key | scope | description |
|-----|-------|-------------|
| `type` | backend | inference engine: `mnn`, `onnx` (ONNX Runtime) or `tensorrt` |
| `model_file_path` | backend | weights file (`.mnn`, `.onnx`, `.engine`) |
| `device` | backend | `cpu` or `cuda`; TRT engines always run on cuda |
| `device_id` / `gpu_device_id` | backend | cuda device index (alias accepted) |
| `threads` | backend | intra-op threads for mnn / onnx cpu |
| `gpu_mem_limit_mb` | backend (onnx+cuda) | CUDA EP arena cap **per session/worker**; default **2048**; `0` = unlimited (legacy). Override with `MORTRED_ORT_GPU_MEM_LIMIT_MB`. MNN/TRT ignore it. |
| `input_layout` | backend (mnn) | host tensor byte order: `nhwc` for TF-style exports, `nchw` for CHW exports, `auto` follows the model file |
| `precision_mode` / `power_mode` | backend (mnn) | `MNN::BackendConfig` modes |
| `input_names` / `output_names` | backend | io name override/filter, useful for models exposing auxiliary outputs |
| `max_image_pixels` / `max_image_side` | image model params | decoded input safety limits; defaults are 16777216 and 8192 |
| `model_input_image_size` | fixed-image model params | `[height, width]`; must match the session input H/W |
| everything else | params | model specific (thresholds, class names, sizes); key names are unchanged from the historical configs |

Multi-engine models (SAM encoder + decoder, lightglue extractor + matcher) use
one `<key>_backend` sub-table per engine instead of the primary `backend`
table, and orchestrate the sessions in `run_sessions`.

## Migration from the old schema

The historical schema (`[SECTION]` + `backend_type` + `[SECTION_TRT]` /
`[SECTION_ONNX]` / `[SECTION_MNN]` + `[BACKEND_DICT]`) is no longer supported.
Migrate with:

```bash
python scripts/migrate_model_config.py --dry-run   # preview + report
python scripts/migrate_model_config.py             # in-place migration
python scripts/migrate_model_config.py --check     # CI gate (exit 1 on drift)
```

Mapping: `compute_backend -> device`, `gpu_device_id -> device_id`,
`model_threads_num -> threads`, `backend_precision_mode -> precision_mode`,
`backend_power_mode -> power_mode`, `trt -> tensorrt`; all other keys move
into `[SECTION.params]` with unchanged names and semantics.

## Testing notes

`model_golden_test` rewrites `../`-prefixed paths relative to the repo root and
forces `backend.device = "cpu"` (and the legacy `compute_backend`) so cpu-only
CI stays deterministic; TensorRT engines still require a GPU and are skipped
when unavailable.
