#!/usr/bin/env bash
# docker_entrypoint.sh - 容器启动入口：环境注入 + 启动 web console（app_server）。
#
# 模型子进程由 console 在容器内管理；权重/引擎经 volume 挂载到
# /opt/mortred/weights（镜像内不包含权重）。
set -euo pipefail

export APP_PROJECT_ROOT="${APP_PROJECT_ROOT:-/opt/mortred}"
export APP_LISTEN_HOST="${APP_LISTEN_HOST:-0.0.0.0}"
export APP_LISTEN_PORT="${APP_LISTEN_PORT:-8787}"
# 运行库：优先安装树 lib（含 3rd_party 库），再兜底系统路径
export LD_LIBRARY_PATH="/opt/mortred/lib:${LD_LIBRARY_PATH:-}"

if [ -z "${APP_AUTH_TOKEN:-}" ] && [ "${APP_LISTEN_HOST}" != "127.0.0.1" ] && [ "${APP_LISTEN_HOST}" != "localhost" ]; then
    echo "[entrypoint] WARNING: 非回环监听未配置 APP_AUTH_TOKEN，console 将拒绝启动（fail-closed）" >&2
fi

echo "[entrypoint] starting mortred web console on ${APP_LISTEN_HOST}:${APP_LISTEN_PORT}"
exec /opt/mortred/bin/app_server
