# syntax=docker/dockerfile:1
# Mortred Model Server - 全自动构建运行环境
#
# 基线线：CUDA 11.8 + TensorRT 8.6.1 + cuDNN 8（与 3rd_party 已备集合一致）。
# CUDA 12 / TRT 10 线：替换 base 为 12.x 并在 install_deps.sh 加 --cuda-version 12
# （引擎转换使用外部 trtexec，需用与本机 TRT 版本匹配的 CLI 重建）。
#
# 构建：docker build -t mortred_model_server .
# 运行：docker run --gpus all -p 8787:8787 -v <weights>:/opt/mortred/weights -e APP_AUTH_TOKEN=... mortred_model_server
#
# CPU profile（无 NVIDIA GPU 的机器）：
# 构建：docker build --target mortred-cpu -t mortred_model_server:cpu .
# 运行：docker run -p 8787:8787 -p 8080:8080 -v <weights>:/opt/mortred/weights -e APP_AUTH_TOKEN=... mortred_model_server:cpu
# （TensorRT 编译排除，catalog 只暴露 *_cpu 配置的精选模型；compose 用 --profile cpu）

# ---------- 阶段 1：第三方依赖（install_deps.sh 全自动） ----------
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git cmake curl ca-certificates jq \
        libssl-dev libgoogle-glog-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src/
# root 环境下 --all 会同时安装 NVIDIA deb（CUDA/TRT/cuDNN 头与库）并构建 MNN/workflow/onnxruntime/fmt
RUN ./scripts/install_deps.sh --all \
    && ./scripts/install_deps.sh --check

# ---------- 阶段 2：full build + 测试 + 安装树 ----------
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS build

# CI 质量门禁等场景注入额外 CMake 开关（如 -DMORTRED_ENABLE_WERROR=ON）
ARG EXTRA_CMAKE_FLAGS=""

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev libeigen3-dev libopencv-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY --from=deps /src/3rd_party /src/3rd_party
COPY . /src/
# 注：测试目标均为 EXCLUDE_FROM_ALL（test/CMakeLists.txt），必须显式构建 check 目标
# （先构建全部测试再跑 ctest --output-on-failure；LD_LIBRARY_PATH 供测试加载动态库）
RUN cmake -S /src -B /src/build \
        -DMORTRED_BUILD_FULL=ON -DMORTRED_INSTALL=ON -DCMAKE_BUILD_TYPE=Release \
        ${EXTRA_CMAKE_FLAGS} \
    && cmake --build /src/build -j"$(nproc)" \
    && LD_LIBRARY_PATH=/src/_lib:/src/3rd_party/libs cmake --build /src/build --target check -j"$(nproc)" \
    && cmake --install /src/build --prefix /opt/mortred

# ---------- 阶段 3：运行时（只装运行库） ----------
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    MORTRED_PROJECT_ROOT=/opt/mortred \
    MORTRED_API_HOST=0.0.0.0 \
    MORTRED_API_PORT=8787 \
    MORTRED_GATEWAY_HOST=0.0.0.0 \
    MORTRED_GATEWAY_PORT=8080 \
    MORTRED_AUTOSTART=true

# TensorRT / cuDNN / OpenCL / glog / OpenCV 运行库（NVIDIA apt + ubuntu apt）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        ocl-icd-libopencl1 libssl1.1 \
        libgoogle-glog0v5 libopencv-core4.2 libopencv-imgproc4.2 libopencv-imgcodecs4.2 \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libnvinfer8 libnvinfer-plugin8 libnvonnxparser8 libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/mortred /opt/mortred

# 权重与引擎不进入镜像：运行时挂载 -v <weights>:/opt/mortred/weights
VOLUME ["/opt/mortred/weights"]

# supervisor (management) 8787 + gateway (inference) 8080; model servers stay loopback
EXPOSE 8787 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://localhost:8787/api/v1/health || exit 1

ENTRYPOINT ["/opt/mortred/scripts/docker_entrypoint.sh"]

# ============================================================================
# CPU profile stages (docker build --target mortred-cpu): ubuntu base, no CUDA
# base image; MNN-CPU + ORT-CPU via install_deps.sh --cpu; TensorRT compiled
# out; the runtime exports MORTRED_PROFILE=cpu so the catalog only shows the
# curated *_cpu model configs.
# ============================================================================

# ---------- 阶段 1c：cpu 第三方依赖 ----------
FROM ubuntu:22.04 AS deps-cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git cmake curl jq ca-certificates \
        libssl-dev libgoogle-glog-dev libeigen3-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src/
RUN ./scripts/install_deps.sh --cpu --all \
    && ./scripts/install_deps.sh --cpu --check

# ---------- 阶段 2c：cpu full build + 测试 + 安装树 ----------
FROM ubuntu:22.04 AS build-cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev libeigen3-dev libopencv-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY --from=deps-cpu /src/3rd_party /src/3rd_party
COPY . /src/
RUN cmake -S /src -B /src/build \
        -DMORTRED_BUILD_FULL=ON -DMORTRED_BUILD_PROFILE=cpu \
        -DMORTRED_INSTALL=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /src/build -j"$(nproc)" \
    && LD_LIBRARY_PATH=/src/_lib:/src/3rd_party/libs cmake --build /src/build --target check -j"$(nproc)" \
    && cmake --install /src/build --prefix /opt/mortred

# ---------- 阶段 3c：cpu 运行时 ----------
FROM ubuntu:22.04 AS mortred-cpu
# default target remains the gpu runtime above; mortred-cpu is opt-in

ENV DEBIAN_FRONTEND=noninteractive \
    MORTRED_PROJECT_ROOT=/opt/mortred \
    MORTRED_PROFILE=cpu \
    MORTRED_API_HOST=0.0.0.0 \
    MORTRED_API_PORT=8787 \
    MORTRED_GATEWAY_HOST=0.0.0.0 \
    MORTRED_GATEWAY_PORT=8080 \
    MORTRED_AUTOSTART=true

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libssl3 \
        libgoogle-glog0v6 libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build-cpu /opt/mortred /opt/mortred

VOLUME ["/opt/mortred/weights"]
EXPOSE 8787 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8787/api/v1/health || exit 1
ENTRYPOINT ["/opt/mortred/scripts/docker_entrypoint.sh"]