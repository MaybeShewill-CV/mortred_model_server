# 部署与依赖改造计划（安装脚本 / Docker / 权重下载 / TRT 引擎重建）

> 目标：把"用户手动编译第三方库拷贝进 3rd_party + 百度网盘下权重 + 无容器"的部署方式，
> 改造为"一条命令装依赖、一条命令建镜像、一条命令下权重、一条命令生成硬件适配引擎"。

## 1. 现状证据（已核实，2026-08 快照）

| 事实 | 证据 |
|---|---|
| 构建依赖硬编码 3rd_party | 顶层 `CMakeLists.txt:34-39` `file(GLOB)` 搜 `3rd_party/libs/*.so`；`:62-64` SYSTEM include `3rd_party/include`（11 个 vendored 头目录：MNN/workflow/onnxruntime/TensorRT-8.6.1.6/rapidjson/toml11/fmt/stb_image/stl_container/indicators/openssl） |
| 本机 vendored 集合为 CUDA 11 时代 | `3rd_party/libs` 42 个 .so：libcudart.so.11.0 / libnvinfer.so.8.6.1 / libcudnn.so.8 / libonnxruntime.so.1.18.0 / libworkflow.so.0.10.9 / libfmt.so.9 / libMNN.so + libMNN_Cuda_Main.so |
| 依赖获取脚本是半自动的 | `scripts/setup_full_deps.sh`：需用户先手动设置 `MNN_ROOT_DIR/WORKFLOW_ROOT_DIR/...` 指向已编译源码根，仅补缺 |
| 无容器构建 | 仓库无 `Dockerfile`、无 `docker-compose`、无 install 目标 |
| 权重依赖人工下载 | README：模型权重经百度网盘（提取码 1y98）或 HF 页 `MaybeShewill-CV/mortred_model_server` |
| 权重目录现状 | `weights/`：41 个 `.onnx` + 46 个 `.mnn`，**0 个 `.trt/.engine`** |
| 引擎文件是配置的硬需求 | `conf/model/**/*.toml` 中 `[XXX_TRT]` 段以 `model_file_path` 引用 `<weights>/.../*.engine`（如 `yolov8s.engine`）；本地无对应文件 → 版本错配/缺失 |
| 转换工具已存在 | `src/apps/model_tools/trt_converter/onnx2trt_converter.cpp`：`exe <onnx> <engine> [fp_mode 0|1] [profile.json]`；`TRT_PRECISION_FP32=0 / FP16=1`（`onnx2trt_model_builder.h:20-22`） |
| 动态 shape 模型需 optimization profile | lightglue extractor/matcher、SAM、depth_anything、metric3d 等（WSL 构建报告记载了重建 profile） |

## 2. 设计决策

1. **保持 3rd_party 作为安装目标**（本目标明确要求装进 `3rd_party/{include,libs}`，与现有 GLOB+SYSTEM include 构建系统兼容）；`find_package` 化作为后续可选演进，不阻塞本计划。
2. **版本线**：默认对齐本机已验证集合（CUDA 11.8 / TRT 8.6.1 / cuDNN 8 / onnxruntime 1.18 / MNN 2.7 / workflow 0.10.9）；**同时提供 CUDA 12 + TRT 10 升级线**（需先完成源码迁移工作包 P0，见 §4）。两条线由 `install_deps.sh` 的 `--cuda-version 11|12` 与 `--tensorrt-version` 参数切换。
3. **引擎重建是必须项**：config 期望的 `.engine` 与用户硬件（GPU 架构/TRT 版本）强相关，必须由脚本按本机生成，不能分发。
4. **权重源**：HF（可脚本化、有版本 hash），百度网盘降级为文档中的备选。

## 3. 交付物与验收（4 项）

| # | 交付物 | 验收标准 |
|---|---|---|
| D1 | `scripts/install_deps.sh` | 干净 clone 后单命令装全 3rd_party（`--check` 输出全部依赖版本与存在性）；幂等可重跑 |
| D2 | `Dockerfile` + `.dockerignore` + `docker-compose.yml` + `docker_entrypoint.sh` | `docker build` 全自动成功；`docker run` 后 `/api/health`=200、`/api/catalog` 列出 22 server |
| D3 | `scripts/fetch_weights.py` | 按清单从 HF 下载权重到 `weights/`（含 onnx/mnn，支持子集与断点续传）；不依赖百度网盘 |
| D4 | `scripts/convert_trt_engines.sh` + `conf/trt_engines.json` | 按清单把 onnx 转成 config 期望路径的 `.engine`（FP16 优先，动态 shape 带 profile）；转换后对应模型 benchmark 可加载 |

## 4. 工作包与实施顺序（合计约 6 人日）

### P0（前置，约 1.5 人日，需 GPU 机/CI 编译验证）：TRT 版本适配源码迁移
- 仅当走 CUDA 12/TRT 10 线时需要：13 个文件 include 去版本化（`"TensorRT-8.6.1.6/NvInfer*.h"` → `<NvInfer*.h>`）、9 个文件 `->destroy()` → `delete`（TRT 10 移除该 API）、MNN CUDA 12 后端验证。
- 走 CUDA 11/TRT 8.6 基线线时 P0 可跳过（现有源码即兼容）。

### P1（约 1.5 人日）：`scripts/install_deps.sh`
- 用法：`./scripts/install_deps.sh [--cuda-version 11|12] [--prefix] [--mnn-tag] [--workflow-tag] [--onnxruntime-ver] [--check] [--offline DIR]`
- 三类来源：
  - **NVIDIA deb**（apt 仓库，接受 NVIDIA 许可）：CUDA toolkit、TensorRT（8.6.1.6 或 10.3.0 视线）、cuDNN（8 或 9）→ 头/库拷入 `3rd_party/{include,libs}`
  - **源码构建**（钉 tag，`cmake --install` 到临时前缀再拷入）：MNN（`-DMNN_CUDA=ON`）、workflow（0.10.9，`make`）
  - **官方二进制**（sha256 钉死）：onnxruntime linux-x64-gpu tarball
- 其余小库（glog/eigen/opencv/fmt/rapidjson/toml11/stb/moodycamel）走 vcpkg（复用 `vcpkg.json`）或 apt，拷入 3rd_party。
- 幂等：每个依赖一个 `3rd_party/.install-stamp/<dep>` 文件；`--check` 校验 `nvcc --version`、`NvInferVersion.h` 宏、各 `.so` 存在并打印版本。

### P2（约 1 人日）：Docker 全自动构建
- 多阶段：`deps-builder`（跑 `install_deps.sh` 填充 3rd_party，BuildKit 缓存）→ `build`（`cmake -DMORTRED_BUILD_FULL=ON` + 单元/e2e 测试 + 生成全部二进制）→ `runtime`（`nvidia/cuda:11.8.0-runtime-ubuntu20.04` 或 CUDA12 对应 tag，只装运行库 + `/opt/mortred` 安装树 + frontend）。
- `HEALTHCHECK` curl `/api/health`；权重与引擎经 volume 挂载（`-v weights:/opt/mortred/weights`）；`APP_AUTH_TOKEN`/端口由 env 注入。

### P3（约 0.5 人日）：`scripts/fetch_weights.py`
- 清单 `weights/manifest.json`（路径 + sha256 + HF repo 内路径）；用 `huggingface_hub` `snapshot_download` 或逐文件下载；`--only <category|model>` 子集；`--check` 校验已存在文件的 sha256；断点续传。
- HF 仓库：`MaybeShewill-CV/mortred_model_server`（README 已声明）；百度网盘写入文档备选。

### P4（约 1 人日）：TRT 引擎重建 `scripts/convert_trt_engines.sh` + `conf/trt_engines.json`
- 清单条目示例：
  ```json
  {
    "models": [
      {"name": "yolov8s", "onnx": "weights/object_detection/yolov8/yolov8s.onnx",
       "engine": "weights/object_detection/yolov8/yolov8s.engine", "fp": 1, "profile": null},
      {"name": "lightglue_extractor", "onnx": "weights/feature_point/lightglue/extractor.onnx",
       "engine": "weights/feature_point/lightglue/extractor.engine", "fp": 1,
       "profile": "conf/trt_profiles/lightglue_extractor.json"}
    ]
  }
  ```
- 脚本：`_bin/onnx2trt_converter.out <onnx> <engine> <fp> <profile>`；对缺失 onnx 给出"先 `fetch_weights.py --only <model>`"的明确提示；结束用 `--verify` 跑对应模型 benchmark 确认引擎可加载。
- profile JSON 结构对齐 `Onnx2TrtModelBuilder` 的 optimization profile 解析（WSL 报告记载：extractor 图像 [64..512]、matcher 关键点 [1..2048]、sam decoder 固定 128 点）。

### P5（约 0.5 人日）：CI 与文档
- CI：新增 job 用 `install_deps.sh --check` 做门禁；Docker build 进 nightly（GPU runner 上跑 golden + 引擎重建验证）。
- 文档：`docs/deployment.md`（三命令工作流 + 两条版本线 + 常见故障）；README 部署章节替换百度网盘说明为 `fetch_weights.py`。

## 5. 风险与处置

| # | 风险 | 处置 |
|---|---|---|
| R1 | TRT 10 破坏性 API（`destroy()` 移除等） | 走 CUDA12 线前先完成 P0 并在 GPU 编译验证；基线线（CUDA11/TRT8.6）不受影响 |
| R2 | MNN CUDA 12 后端版本兼容 | P1 的 `--mnn-tag` 参数可钉验证过的 tag；CI 矩阵覆盖 |
| R3 | onnxruntime 源码构建慢 | 官方 release tarball + sha256 |
| R4 | 引擎与 config 期望路径/名称不一致（如 yolov8s.onnx 本地缺失） | 清单显式化 + fetch_weights 提供 onnx 子集下载 + `--verify` benchmark 兜底 |
| R5 | 镜像体积（CUDA+TRT ~3-5GB） | runtime 阶段只装运行库；权重/引擎不入镜像 |
| R6 | 本环境无 GPU，无法实测引擎转换与 Docker build | 实施顺序 P1→P3 可本环境完成（脚本+静态校验）；P2/P4 的实测项放 GPU 机/CI 验证并记录结果 |

## 6. 执行顺序建议

1. P1 `install_deps.sh`（本环境可开发 + `--check` 静态验证）
2. P3 `fetch_weights.py`（本环境可开发 + 对小文件实测下载）
3. P0 若走 CUDA12/TRT10 线（需 GPU 编译验证）
4. P4 引擎重建清单与脚本（GPU 机实测）
5. P2 Docker（GPU 机/CI 构建验证）
6. P5 文档与 CI 收尾
