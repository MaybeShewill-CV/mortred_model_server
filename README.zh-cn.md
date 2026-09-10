<div id="top" align="center">

  <h1 align="center">
    <img src="./resources/images/icon.png" alt='icon.png' height="180px" width="180px"/>
  </h1>

  ![icon](./resources/images/iconv1.gif)

   Mortred-AI-Web-Server: 一个面向DL模型的Web服务器

   | [English](README.md) | [中文](README.zh-cn.md) |

   [![CI](https://github.com/MaybeShewill-CV/mortred_model_server/actions/workflows/ci.yml/badge.svg)](https://github.com/MaybeShewill-CV/mortred_model_server/actions/workflows/ci.yml)

</div>

这是一个面向单机 CPU/GPU 的 CV 推理设备：**一个 catalog id = 一个 OS 进程**。客户端打 **mortred-gateway**（`:8080`）；监督器（`:8787`）管进程树。推理后端是 [MNN](https://github.com/alibaba/MNN)、ONNX Runtime 和 TensorRT，HTTP 层用 [workflow](https://github.com/sogou/workflow)。训练仍在 `tensorflow` / `pytorch` 侧完成。

欢迎你反馈任何你发现的bug，本人还是一个c with struct 弱鸡 :upside_down_face:

模型文件可以访问我的 [Hugging Face Page](https://huggingface.co/MaybeShewill-CV/mortred_model_server)

整个项目的简要架构图如下

<p align="center">
  <img src='./resources/images/simple_architecture.png' alt='simple_architecture' height="400px" width="500px">
</p>

欢迎你提出改进意见或者pr来帮助我把它建设的更好 :smile::fire:

# `文档目录`

* [快速开始](#快速开始)
* [Benchmark](#benchmark)
* [模型说明](#模型说明)
* [文档教程](#文档教程)
* [网络服务器配置说明](#网络服务器配置说明)
* [HTTP API 契约](./docs/api-contract.zh-cn.md)
* [长任务 `/jobs` 客户验收](./docs/async-jobs-customer-test.zh-cn.md)
* [Model_Zoo](#model_zoo)

# `快速开始`

> Linux 是唯一受支持的平台。两条部署 profile，一个开关驱动构建、依赖、模型目录和权重子集：
>
> | | `gpu`（默认） | `cpu` |
> |---|---|---|
> | 后端 | MNN-CUDA / ORT-CUDA / TensorRT | MNN-CPU / ORT-CPU |
> | 硬件 | NVIDIA GPU + CUDA 11/12 | 任意 x64 |
> | 模型 | HTTP catalog 全量 | 精选（mobilenetv2、resnet50） |
>
> 三个入口，同一套 `mortredctl`：选一条即可，终点都是 `mortredctl doctor`。
> 完整运维手册见 [docs/deployment.zh-cn.md](docs/deployment.zh-cn.md)。

### 入口一：一行 bootstrap（最快）

```bash
curl -fsSL https://raw.githubusercontent.com/MaybeShewill-CV/mortred_model_server/main/scripts/bootstrap.sh | bash
```

探测硬件（有 NVIDIA → `gpu`，否则 `cpu`）。有 Docker 则打印 compose 轨道。无
Docker 则解析 GitHub 最新 **release tag**，再下载
`mortred_model_server-<version>-<profile>-linux-x64.tar.gz`（**没有**
`...-latest-...` 这种 tarball 文件名）。若还没有 Release，则 WARN 并打印源码构建路径。

### 入口二：docker compose

```bash
git clone https://github.com/MaybeShewill-CV/mortred_model_server.git
cd mortred_model_server
python3 scripts/fetch_weights.py --profile cpu        # 或: gpu
./scripts/mortredctl_init-trust.sh                    # 三个互异 token
set -a && . conf/local/trust.env && set +a
docker compose --profile cpu up -d                    # 或: --profile gpu
curl -fs http://localhost:8787/api/v1/health
```

### 入口三：release tarball + systemd（裸机）

从 [Releases](https://github.com/MaybeShewill-CV/mortred_model_server/releases)
下载 `mortred_model_server-<version>-<profile>-linux-x64.tar.gz`，校验 `.sha256`，然后：

```bash
mkdir unpack && tar -xzf mortred_model_server-*-linux-x64.tar.gz -C unpack
cd unpack                                          # 平铺：install.sh、opt/、deploy/
sudo ./install.sh
sudo /opt/mortred/bin/mortredctl.out init-trust --force --out /etc/mortred/supervisor.env
cd /opt/mortred && python3 scripts/fetch_weights.py --profile cpu
sudo systemctl start mortred-supervisor
```

### 第一小时：mortredctl

```bash
mortredctl init [--profile cpu|gpu]
mortredctl init-trust
mortredctl init-edge --mode lan
mortredctl prepare [--pack FILE]
mortredctl calibrate [--pack FILE]
mortredctl doctor
mortredctl doctor --strict            # 缺 engine、缺占用 stamp、占用门关闭、安全警告都会失败
mortredctl status | catalog
```

GPU：用 `mortredctl prepare` 在本机转 **当前 pack** 的 TensorRT engine，不要默认转整个 zoo。见 [部署说明](docs/deployment.zh-cn.md) §10。

### 从源码构建

手装 CUDA / MNN / WORKFLOW / OpenCV / TensorRT 不是快速开始；那是源码构建路径。
两条 CMake 路径：

- **tests-only**：只构建 `common` 与单元测试（CI / 快速验证）。
- **full build**：全部模型与服务，需要 `3rd_party` 下的引擎。

#### 路径 A：tests-only


方案 A1 - 系统包（推荐，与 CI 一致）：

```bash
sudo apt-get install -y build-essential cmake \
  libopencv-dev libgoogle-glog-dev libeigen3-dev libgtest-dev libssl-dev
# Ubuntu 22.04 的 libgtest-dev 自带预编译库与 CMake 配置，可直接 find_package(GTest)

cd $PROJECT_ROOT_DIR
cmake -B build -DMORTRED_BUILD_FULL=OFF
cmake --build build --target check -j10
ctest --test-dir build --output-on-failure
```

方案 A2 - vcpkg（可选；仅本地开发用，CI 不使用）：

```bash
# 1. 安装 vcpkg（或复用已有实例）
git clone https://github.com/microsoft/vcpkg.git /path/to/vcpkg
/path/to/vcpkg/bootstrap-vcpkg.sh -disableMetrics

# 2. 配置（vcpkg 会按 vcpkg.json 自动安装 opencv/glog/eigen3/gtest）
cd $PROJECT_ROOT_DIR
cmake -B build -DMORTRED_BUILD_FULL=OFF \
      -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

# 3. 构建并运行单元测试
cmake --build build --target check -j10
ctest --test-dir build --output-on-failure
```

`vcpkg.json` 中故意不写死 `builtin-baseline`；若你的 vcpkg 实例要求显式 baseline，执行一次
`vcpkg x-update-baseline --add-initial-baseline` 后重新配置即可。

#### 路径 B：full build

```bash
# 1. 校验/补齐 vendored 第三方依赖
#    （MNN / WORKFLOW / ONNXRUNTIME / TensorRT + CUDA）。
#    缺失时按提示设置对应的 *_ROOT_DIR 环境变量后重试。
./scripts/setup_full_deps.sh

# 2. 配置并构建
mkdir build && cd build
cmake ..            # 可选：追加 -DCMAKE_TOOLCHAIN_FILE=... 以同时使用 vcpkg
make -j10
```

默认可执行文件输出到 `$PROJECT_ROOT_DIR/_bin`，动态库输出到 `$PROJECT_ROOT_DIR/_lib`；
两者均可用 `-DMORTRED_BIN_OUTPUT_DIR=...` 与 `-DMORTRED_LIB_OUTPUT_DIR=...` 覆盖。

常用 CMake 选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `MORTRED_BUILD_FULL` | `ON` | 构建全部模型/服务/工具（需要 CUDA 与 vendored 引擎）；置 `OFF` 进入 tests-only 模式。 |
| `MORTRED_ENABLE_WERROR` | `OFF` | 将编译器警告视为错误（`-Wall -Wextra -Werror`），供 CI 质量门禁使用。 |
| `MORTRED_BIN_OUTPUT_DIR` | `$PROJECT_ROOT_DIR/_bin` | 可执行文件输出目录。 |
| `MORTRED_LIB_OUTPUT_DIR` | `$PROJECT_ROOT_DIR/_lib` | 动态库输出目录。 |

项目提供了 CMake Presets（见 `CMakePresets.json`）：

```bash
cmake --preset tests-only
cmake --build --preset tests-only
ctest --preset tests-only
```

仓库目录规范与源码/配置/可执行文件映射见 [docs/repository-layout.md](docs/repository-layout.md)。

**Step 3:** 下载项目提供的一些预训练模型 :tea::tea::tea:

通过内置脚本自动下载预训练模型（Hugging Face 源，无需手动下载）：

```bash
cd $PROJECT_ROOT_DIR
python3 scripts/fetch_weights.py            # 下载全部权重到 weights/
python3 scripts/fetch_weights.py --check    # 校验完整性（sha256）
```

如果本机 GPU/TRT 版本与预置引擎不匹配，请按 [部署说明](docs/deployment.zh-cn.md) §10
为本机 **pack** 生成 engine（`mortredctl prepare`），不要默认转整个 zoo：

```bash
cd $PROJECT_ROOT_DIR
mortredctl prepare --pack conf/packs/yolov8.toml
```

完成后的文件夹结构应该如图所示。

<p align="left">
  <img src='./resources/images/weights_folder_structure.png' alt='weights_folder_architecture'>
</p>

**Step 4:** 测试 MobileNetv2 基准测试工具

至此你已经完成的项目的编译工作，可以开始测试体验项目提供的预训练模型了。统一基准测试入口是 `$PROJECT_ROOT_DIR/_bin/mortred-model-benchmark.out`，用 `--model` 选择 catalog 里的模型。

现在你可以通过如下方式来进行 `mobilenetv2` 图像分类基准测试

```bash
cd $PROJECT_ROOT_DIR/_bin
./mortred-model-benchmark.out --model MOBILENETV2 ../conf/model/classification/mobilenetv2/mobilenetv2_config.toml
```

如果没有任何错误的话（应该不会有:dog:），你可以看到如下的测试结果，包含使用的模型，模型预测耗时、fps等信息

<p align="left">
  <img src='./resources/images/mobilenetv2_demo_benchmark.png' alt='mobilenetv2_demo_benchmark'>
</p>

**Step 5:** 运行 MobileNetV2 图像分类服务器

有关网络服务器的一些细节参数可以查看 [网络服务器配置说明](#网络服务器配置说明)。下面让我们愉快的开启服务

```bash
cd $PROJECT_ROOT_DIR/_bin
./mortred-model-server.out --model MOBILENETV2 ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml
```

按照默认的配置文件（`conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml`），服务端口为`9002`，`worker_nums=1` 个模型 worker 等待被调用。项目中含有一个简单的python客户端来测试该服务，使用方法如下

```bash
cd $PROJECT_ROOT_DIR
python3 scripts/server/test_server.py --server mobilenetv2 --mode single --times 3
```

客户端 POST 统一信封（`images[]`），打印 HTTP 状态和截断后的 UnifiedResponse。分类结果在 `results[].data`。下面截图是**历史输出**（旧 `{code,msg,data}` 信封），不要当现行契约。
![mobilenetv2_server_exam_output](./resources/images/exam_server_output.png)
![mobilenetv2_client_exam_output](./resources/images/exam_client_output.png)

你可以在下文的 [模型说明](#模型说明) 章节获取更多的服务示例 :point_down::point_down::point_down:

# `Benchmark`

基准测试环境如下：

**OS:** Ubuntu 20.04.5 LTS / 5.15.0-87-generic

**MEMORY:** 32G DIMM DDR4 Synchronous 2666 MHz

**CPU:** Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz

**GCC:** gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0

**GPU:** GeForce RTX 3080

**CUDA:** CUDA Version: 11.5

**GPU Driver:** Driver Version: 495.29.05

### DL模型推理基准测试

所有模型的测试过程都重复推理若干次以抵消GPU的warmup损耗，并且没有任何的io时间被算入

`Benchmark 代码段`
![benchmakr_code_snappit](./resources/images/benchmark_code_snappit.png)

* [Model_Zoo 所有模型的详细基准测试结果](./docs/model_inference_benchmark.zh-cn.md)
* [关于模型推理的配置文件说明](./docs/about_model_configuration.zh-cn.md)

# `模型说明`

* [图像分类服务部署说明与示例](./docs/tutorials_of_classification_model_server.zh-cn.md)
* [图像分割服务部署说明与示例](./docs/tutorials_of_segmentation_model_server.zh-cn.md)
* [图像目标检测服务部署说明与示例](./docs/tutorials_of_object_detection_model_server.zh-cn.md)
* [图像增强服务部署说明与示例](./docs/tutorials_of_enhancement_model_server.zh-cn.md)
* [图像特征点检测服务部署说明与示例](./docs/tutorials_of_feature_point_model_server.zh-cn.md)

# `文档教程`

* [快速添加新的DL模型](./docs/how_to_add_new_model.zh-cn.md) :fire::fire:
* [快速添加新的DL服务](./docs/how_to_add_new_server.zh-cn.md) :fire::fire:
* [模型开发者指南（任务路径 / 契约 / golden / 调试）](./docs/model-developer-guide.md)
* [推理 CI（托管 MNN 冒烟 vs 维护者 GPU golden）](./docs/ci-golden-regression.md)
* [P4：现代模型开发者体验改造计划](./docs/model-developer-experience-p4.zh-cn.md)

# `网络服务器配置说明`

* [模型网络服务器配置说明](./docs/about_model_server_configuration.zh-cn.md)
* [HTTP API 契约（含网关拓扑、鉴权、状态码映射、过载行为）](./docs/api-contract.zh-cn.md)

# `Model Zoo`

HTTP 可服务（`mortred-model-server.out --list` / catalog id）：

| 任务 | Catalog id |
|---|---|
| 分类 | `MOBILENETV2` `RESNET` `DENSENET` |
| 检测 | `YOLOV5` `YOLOV6` `YOLOV7` `YOLOV8` `NANODET` |
| 人脸 | `LIBFACE` `CENTER_FACE` |
| OCR | `DBNET` |
| 分割 | `BISENETV2` `PPHUMAN_SEG` `HRNET` |
| 抠图 | `MODNET` `PP_MATTING` |
| 增强 | `ENLIGHTEN_GAN` `ATTENTIVE_GAN_DERAIN` `REAL_ESRGAN` |
| 特征点 | `SUPERPOINT` |
| 嵌入 | `DINOV2` |
| 深度 | `METRIC3D` `DEPTH_ANYTHING` |
| SAM | `SAM_AMG` |
| 扩散 | `DDPM` `DDIM` `CLS_COND_DDIM` `LDM` |

Bench-only（无 HTTP catalog）：`OPENAI_CLIP`、`LIGHTGLUE`、`SAM_PREDICTOR`、`FAST_SAM`、`MSOCRNET`。

Scaffold、未实现、不在 HTTP catalog：`RTDETR`。无 MOT。

# `部署说明`

## 一键安装第三方依赖

通过单个脚本把全部第三方依赖（MNN / WORKFLOW / ONNXRUNTIME / TensorRT / CUDA /
fmt / 头文件库）构建并安装进 `3rd_party/{include,libs}`，无需手动编译与拷贝：

```bash
./scripts/install_deps.sh --all     # 构建/安装全部（CUDA 11 基线线）
./scripts/install_deps.sh --check   # 校验完整性并打印版本
./scripts/install_deps.sh --cuda-version 12   # 切换到 CUDA 12 / TRT 10 线
```

## Docker（全自动构建运行环境）

```bash
docker build -t mortred_model_server:gpu .
docker run --gpus all -p 127.0.0.1:8080:8080 -p 127.0.0.1:8787:8787 \
  -v $PWD/weights:/opt/mortred/weights \
  -e MORTRED_GATEWAY_AUTH_TOKEN=your-inference-token \
  -e MORTRED_API_TOKEN=your-management-token \
  -e MORTRED_METRICS_TOKEN=your-scrape-token \
  mortred_model_server:gpu
# 或：docker compose --profile gpu up -d（CPU：--profile cpu；见 docker-compose.yml）
```

镜像会自动构建全部依赖与完整项目、运行单元/e2e 测试并交付控制面；
模型权重通过 volume 挂载，不内置于镜像。容器内拓扑：`mortred-supervisor`
（管理面 :8787，内嵌 Web UI + REST API）监督 `mortred-gateway`（数据面
:8080，推理统一入口）与全部模型进程；模型进程仅绑定 127.0.0.1，不再
逐端口暴露。compose 与 `docker run` 示例把 8080/8787 绑在宿主机
`127.0.0.1` 上。对外暴露必须由主机网络上的 Nginx 终结 TLS
（`mortredctl init-edge`，[deploy/nginx](deploy/nginx)）；不要在没有边缘时把
这些端口发到 `0.0.0.0`（Bearer 会明文传输）。网关 `GET /metrics` 在环回上也要
`MORTRED_METRICS_TOKEN`。缺推理/管理身份、缺 scrape token、scrape 与其它身份相同、
或未设 `MORTRED_EXPOSE=docker|unsafe` 的通配绑定都会拒绝启动。
`mortredctl doctor` 会对非环回监听、缺失 scrape token 和过短/相同的 token 告警；
`doctor --strict` 会因这些警告失败。TLS 仍在 Nginx 上终结。

## TensorRT 引擎重建（硬件适配）

Engine 绑定本机 GPU / TRT。日常只转 **当前 pack**（见 [部署指南 §10](docs/deployment.zh-cn.md)）：

```bash
mortredctl prepare --pack conf/packs/yolov8.toml
```

全量 zoo 仍可用 `scripts/convert_trt_engines.sh`（需要 `trtexec`：
`sudo ./scripts/install_deps.sh --nvidia`）。`MORTRED_AUTO_BUILD_ENGINES=true`
会转整个 zoo，默认关闭。

# `TODO`

* [ ] 增加更多的DL模型

# `开发状态`

![repo-status](https://repobeats.axiom.co/api/embed/b8c3f964c5afc4776f62a12bcd1e76c57ac554ca.svg "Repobeats analytics image")

# `致谢`

mortred_model_server 项目参考、借鉴了以下项目:

* <https://github.com/sogou/workflow>
* <https://github.com/alibaba/MNN>
* <https://github.com/PaddlePaddle/PaddleSeg>
* <https://github.com/Tencent/rapidjson>
* <https://github.com/ToruNiina/toml11>
