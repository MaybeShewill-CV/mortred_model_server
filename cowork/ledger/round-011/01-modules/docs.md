# 模块评分

- id: docs
- 权重: 10
- 分数: 7.2
- 读过的路径: `docs/api-contract.md`、`docs/unsupported-boundaries.md`、`docs/deployment.md`、`docs/benchmark-set-a.md`、`docs/openapi.json`（范围内路径与 `components.schemas`）、`README.md` 构建段（136–166）与 TensorRT 段（297–308）、`README.zh-cn.md` 构建段（112–183）与 TensorRT 段（356–366）、`CMakePresets.json` 的 `full` / `full-cpu`、打分时各任务 `catalog()` / `face_catalog()`

## 优势

`docs/deployment.md` 把 Linux 上的三条安装入口、profile、权重、本机 pack 的 TensorRT、三令牌、升级回滚、监控和 `mortredctl doctor` 排障写成可执行步骤。`docs/unsupported-boundaries.md` 写明只支持 Linux x64、HTTP OCR 只有 `dbnet` 检测、CUDA 12.x 与 TensorRT 10.x、engine 须在本机本卡重建。`docs/api-contract.md` 写了统一信封、状态码到 HTTP 的映射，以及模型端口先鉴权再按 IP 限流。

这 7 个 id 在 `docs/api-contract.md` 有与 `docs/openapi.json` 数据 schema 一致的响应示例，请求体用同文件 261–267 行的共用信封（`images` 必填）：`mobilenetv2`（132–140 行，`ClassificationResult`）、`yolov8s`（142–155 行，`DetectionResult`）、`pphuman_server`（157–165 行，`SegmentationResult`）、`dbnet`（167–178 行，`TextRegionResult`）、`modnet`（181–188 行，`MattingResult`）、`realesrgan`（190–197 行，`EnhancementResult`）、`metric3d_512`（199–206 行，`DepthResult`）。

## 劣势

H1：以下 32 个 id 没有一条与 `docs/openapi.json` 一致的请求和响应示例。`docs/openapi.json` 全文没有 `example`。`docs/api-contract.md` 只给上面 7 个 id 写了 `results[].data`。缺的 id：`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`hrnet`、`msocrnet`、`attentive_gan`、`enlightengan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_1088`。`libface` 与 `centerface` 的 data schema 是 `FaceResult`（必填 `landmarks`）。`superpoint` 的 data schema 是 `FeaturePointResult`。`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`msocrnet` 的 data `$ref` 是没有字段的 `EnvelopeData`。改完后，上述每个 id 都能在文档里对到一条请求示例和一条响应示例，并且对得上该 id 在 `docs/openapi.json` 里的 request schema 与写出字段的 data schema；人脸示例含 `landmarks`，`superpoint` 示例含 `score`、`location`、`descriptor`。

H3：范围内 39 个 id 都没有一行同时写出硬件、复现命令、指标和日期。`docs/benchmark-set-a.md` 第 9 行是 GPU（NVIDIA GeForce RTX 2070 SUPER，8192 MiB），第 10 行日期是 2026-09-24，这两行没有 id、指标和命令。表 2 各行有指标和 `logs/bench/set_a/trtexec/<id>.cmd.txt` 路径，没有硬件、日期和命令正文。表 4 有 HTTP 指标，没有硬件、命令和日期；`msocrnet` 在表 3、表 4 为空。表 1 用 `libface_320`、`libface_640`，catalog id 是 `libface`。缺这一行的 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。改完后，每个 id 有一行同时含硬件、可直接执行的复现命令、指标和日期。

C1：中文和英文 README 的 GPU 源码构建命令不一致。`README.zh-cn.md` 171 行有 `sudo ./scripts/install_deps.sh --nvidia`。`README.md` 141–156 行的 GPU 依赖行是 `./scripts/install_deps.sh --all` 与 `./scripts/install_deps.sh --check`。两边的 `cmake --preset full && cmake --build --preset full` 与 `cmake --preset full-cpu && cmake --build --preset full-cpu` 一致（`README.md` 151–152，`README.zh-cn.md` 173–174），并与 `CMakePresets.json` 的 configure preset `full-cpu`（62 行）、`full`（75 行）及同名 build preset（138、143 行）一致。CPU 档 `--cpu --all` 与 `--cpu --check` 两边都有。改完后，两份 README 源码构建代码块的 GPU 可执行行集合相同，并且仍包含上述两条 `cmake --preset` 命令。

`docs/deployment.md` 108 行和 253 行把 cpu 目录写成 `resnet50`，catalog id 与 `conf/server/classification/resnet/resnet_server_cpu_config.toml` 的 `model` 是 `resnet`。同文件 108 行把 `MsOcrNet` 写成 Bench-only、不是 SME HTTP 路径，`scene_segmentation` 的 `catalog()` 与 `docs/openapi.json` 仍注册 `msocrnet`。改完后，这两处 cpu 名称是 `resnet`，Bench-only 列表不再包含 `msocrnet`。

## 不计毛病

`docs/unsupported-boundaries.md` 写明 HTTP OCR 只有 `dbnet` 检测、没有识别。识别器不进入队列。`dbnet` 仍计入上面的 H1、H3。SAM、diffusion、CLIP、LightGlue 不在最高标准范围表里。

## 依据

范围 id 以打分时的 `catalog()` / `face_catalog()` 为准，共 39 个，与口径表一致。H1 用 `docs/api-contract.md` 的命名示例对照 `docs/openapi.json` 的 data `$ref`：分类、检测、分割、OCR、抠图、增强、深度七个示例与对应 Result schema 的必填字段一致；人脸、特征点、三个 DINOv2、`msocrnet` 在契约页没有示例。`docs/openapi.json` 检索 `example` 为 0 次。H3 通读 `docs/benchmark-set-a.md`：硬件和日期在第 9–10 行，指标在表 2 与表 4，复现位置是 cmd 文件路径或表 3 上方的 `--workers 1,2,4,8,16 --duration 15s --skip-joint`，没有任何一行四项齐备。C1 对照两份 README 的源码构建代码块与 `CMakePresets.json` 的 `full`、`full-cpu`。`resnet` 对照 `src/factory/classification_task.h` 的 `catalog()` 与 cpu server toml。`msocrnet` 对照 `src/factory/scene_segmentation_task.h` 的 `catalog()`。

## 未读

无
