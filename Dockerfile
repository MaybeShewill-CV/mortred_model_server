# syntax=docker/dockerfile:1
# Mortred Model Server - 全自动构建运行环境
#
# 基线线：CUDA 11.8 + TensorRT 8.6.1 + cuDNN 8（与 3rd_party 已备集合一致）。
# CUDA 12 / TRT 10 线：替换 base 为 12.x 并在 install_deps.sh 加 --cuda-version 12
# （需先完成 TRT10 源码迁移，见 docs/deployment-and-deps-plan.md 工作包 P0）。
#
# 构建：docker build -t mortred_model_server .
# 运行：docker run --gpus all -p 8787:8787 -v <weights>:/opt/mortred/weights -e APP_AUTH_TOKEN=... mortred_model_server

# ---------- 阶段 1：第三方依赖（install_deps.sh 全自动） ----------
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git cmake curl ca-certificates \
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
RUN cmake -S /src -B /src/build \
        -DMORTRED_BUILD_FULL=ON -DMORTRED_INSTALL=ON -DCMAKE_BUILD_TYPE=Release \
        ${EXTRA_CMAKE_FLAGS} \
    && cmake --build /src/build -j"$(nproc)" \
    && (cd /src/build && ctest --output-on-failure || true) \
    && cmake --install /src/build --prefix /opt/mortred

# ---------- 阶段 3：运行时（只装运行库） ----------
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    APP_PROJECT_ROOT=/opt/mortred \
    APP_LISTEN_HOST=0.0.0.0 \
    APP_LISTEN_PORT=8787

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

# web console 8787；模型端口 9001-9072 按需映射
EXPOSE 8787 9001 9002 9003 9010 9011 9012 9020 9030 9031 9040 9041 9050 9051 9052 9053 9054 9055 9056 9060 9070 9071 9072

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://localhost:8787/api/health || exit 1

ENTRYPOINT ["/opt/mortred/scripts/docker_entrypoint.sh"]
