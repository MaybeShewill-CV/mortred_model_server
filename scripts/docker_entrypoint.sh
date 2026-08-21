#!/usr/bin/env bash
# docker_entrypoint.sh - container entrypoint: inject environment + start web console (app_server).
#
# Model subprocesses are managed by the console inside the container; weights/engines are
# volume-mounted at /opt/mortred/weights (the image contains no weights).
set -euo pipefail

export APP_PROJECT_ROOT="${APP_PROJECT_ROOT:-/opt/mortred}"
export APP_LISTEN_HOST="${APP_LISTEN_HOST:-0.0.0.0}"
export APP_LISTEN_PORT="${APP_LISTEN_PORT:-8787}"
# Runtime libs: prefer the installed tree lib (contains 3rd_party libs), then fall back to system paths
export LD_LIBRARY_PATH="/opt/mortred/lib:${LD_LIBRARY_PATH:-}"
# Engine conversion tool (trtexec) default location: installed tree bin/ (copied by install_deps.sh --nvidia)
export TRTEXEC="${TRTEXEC:-/opt/mortred/bin/trtexec}"
# Installed tree layout: directory names the console uses to spawn model subprocesses
# (installed tree is bin/lib, 3rd_party libs merged into lib; see CMakeLists.txt MORTRED_INSTALL;
# running from the source tree keeps defaults _bin/_lib/3rd_party/libs)
export APP_BIN_DIR="${APP_BIN_DIR:-bin}"
export APP_LIB_DIR="${APP_LIB_DIR:-lib}"
export APP_LIBS_DIR="${APP_LIBS_DIR:-lib}"

if [ -z "${APP_AUTH_TOKEN:-}" ] && [ "${APP_LISTEN_HOST}" != "127.0.0.1" ] && [ "${APP_LISTEN_HOST}" != "localhost" ]; then
    echo "[entrypoint] WARNING: non-loopback listening without APP_AUTH_TOKEN; console will refuse to start (fail-closed)" >&2
fi

echo "[entrypoint] starting mortred web console on ${APP_LISTEN_HOST}:${APP_LISTEN_PORT}"
exec /opt/mortred/bin/app_server
