# 模型配置说明

所有模型配置都位于 `conf/model/`。统一推理后端层要求每个模型使用
`[模型名.backend]` 与 `[模型名.params]` 两张表：

```toml
[YOLOV8]
[YOLOV8.backend]
type = "tensorrt"              # mnn | onnx | tensorrt
model_file_path = "../weights/object_detection/yolov8/yolov8s.engine"
device = "gpu"                # cpu | gpu，缺省 gpu
device_id = 0                  # 可选，CUDA 设备序号
threads = 4                    # MNN / ONNX CPU 线程数，默认 4
input_layout = "auto"          # 仅 MNN：auto | nhwc | nchw
precision_mode = 0             # 仅 MNN：BackendConfig::PrecisionMode
power_mode = 0                 # 仅 MNN：BackendConfig::PowerMode
input_names = ["images"]       # 可选，默认枚举模型 I/O
output_names = ["output0"]     # 可选，可过滤辅助输出

[YOLOV8.params]
model_score_threshold = 0.25
model_nms_threshold = 0.5
model_input_image_size = [640, 640]   # [height, width]；必须与固定输入 H/W 一致
max_image_pixels = 16777216           # 解码后像素上限
max_image_side = 8192
class_names = ["person", "bicycle"]
```

## 字段速查

| 字段 | 所属表 | 含义 |
|---|---|---|
| `type` | `backend` | 推理引擎：`mnn`、`onnx` 或 `tensorrt` |
| `model_file_path` | `backend` | 权重或 engine 文件 |
| `device` | `backend` | `cpu` 或 `gpu`；缺省为 `gpu`。`type=tensorrt` 且 `device=cpu` 是配置错误 |
| `device_id` / `gpu_device_id` | `backend` | CUDA 设备序号，二者等价 |
| `threads` | `backend` | MNN / ONNX CPU 推理线程数 |
| `gpu_mem_limit_mb` | `backend`（onnx+cuda） | CUDA EP arena **每个 session/worker** 上限；默认 **2048** MiB；`0` = 不限制。环境变量 `MORTRED_ORT_GPU_MEM_LIMIT_MB` 可覆盖。MNN/TRT 忽略。 |
| `input_layout` | `backend` | MNN host 张量布局：`nhwc`、`nchw` 或按模型自动识别 |
| `precision_mode` / `power_mode` | `backend` | MNN `BackendConfig` 配置 |
| `input_names` / `output_names` | `backend` | I/O 名称覆盖或过滤 |
| `max_image_pixels` / `max_image_side` | 图像 `params` | 解码输入安全上限，默认 16777216 / 8192 |
| `model_input_image_size` | 固定尺寸图像 `params` | `[height, width]`，必须匹配 session 输入 H/W |
| `min_box_area_px` | YOLOv5 / v6 / v7 / v8 `params` | 丢掉解码坐标系里 `width*height`（**letterbox 网络像素**）低于该值的框；默认 `5`；在 letterbox unmap **之前**比较 |
| `sample_size` | 扩散 `[DDPM_SAMPLER]` / `[DDIM_SAMPLER]` / `[LDM_SAMPLER]` | 生成画布 `[height, width]`，必须匹配 UNet 权重。HTTP 在 adapter init 时种入模板；不是请求 `params` 键 |
| 其他字段 | `params` | 模型特有参数；名称和历史配置保持一致 |

## 多引擎模型

多引擎模型不使用 primary `[backend]`，而是为每个引擎配置一张
`<key>_backend` 子表，并在 `run_sessions()` 中编排多个 session：

```toml
[SAM_PREDICTOR]

[SAM_PREDICTOR.encoder_backend]
type = "tensorrt"
model_file_path = "../weights/sam/mobile_sam/sm61/mobile_sam_encoder.engine"
device = "gpu"

[SAM_PREDICTOR.decoder_backend]
type = "tensorrt"
model_file_path = "../weights/sam/mobile_sam/sm61/mobile_sam_decoder.engine"
device = "gpu"
```

当前多引擎模型包括：

- SAM predictor：encoder + prompt decoder
- SAM automatic mask generator：encoder + AMG decoder
- LightGlue：SuperPoint extractor + matcher
- OpenAI CLIP：visual encoder + text encoder

Diffusion sampler 是采样调度器，不是单次推理模型。DDPM / DDIM /
class-conditioned DDIM / LDM 继续作为 `BaseAiModel` 编排层；真正持有
session 的是 `DDPMUNet`、`ClsCondDDPMUNet` 和 `AutoEncoderKL`。

## 从旧 schema 迁移

历史结构：

```toml
[SECTION]
backend_type = "trt"

[SECTION_TRT]
model_file_path = "..."

[BACKEND_DICT]
trt = 0
onnx = 1
mnn = 2
```

该结构已不支持。使用迁移脚本：

```bash
python scripts/migrate_model_config.py --dry-run
python scripts/migrate_model_config.py
python scripts/migrate_model_config.py --check
```

字段映射：

```text
compute_backend       -> backend.device
gpu_device_id         -> backend.device_id
model_threads_num     -> backend.threads
backend_precision_mode -> backend.precision_mode
backend_power_mode    -> backend.power_mode
trt                   -> backend.type = "tensorrt"
其他模型特有字段        -> params.*
```

## 测试说明

`model_golden_test` 会把 `../` 前缀路径改写为仓库根路径，并将
`backend.device` 强制为 `cpu`。产品 TensorRT 配置（例如 `yolov8_config.toml`）
在 CPU 上是配置错误（`type=tensorrt` 禁止 `device=cpu`）。Fork hosted CI 用
`conf/ci/yolov8_onnx_hosted.toml` 证明 YOLOv8 decode，不是 serving 的 `.engine`。
TensorRT golden 仍走维护者 GPU 路径。DDPM HTTP 出图由
`conf/ci/ddpm_onnx_fewstep.toml` + `model_golden.ddpm_celeba_hq_fewstep`
证明（不进 hosted；ONNX 约 143MiB）。
