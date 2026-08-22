# docker_entrypoint.sh - container entrypoint: inject environment + start the
# mortred-supervisor (control plane). The supervisor manages mortred-gateway and
# all model servers inside the container; weights are volume-mounted.
set -euo pipefail

export APP_PROJECT_ROOT="${APP_PROJECT_ROOT:-/opt/mortred}"
export MORTRED_PROJECT_ROOT="${APP_PROJECT_ROOT}"
export MORTRED_API_HOST="${MORTRED_API_HOST:-0.0.0.0}"
export MORTRED_API_PORT="${MORTRED_API_PORT:-8787}"
export MORTRED_GATEWAY_HOST="${MORTRED_GATEWAY_HOST:-0.0.0.0}"
export MORTRED_GATEWAY_PORT="${MORTRED_GATEWAY_PORT:-8080}"
# containers are service deployments: autostart everything eligible by default
export MORTRED_AUTOSTART="${MORTRED_AUTOSTART:-true}"
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

if [ -z "${MORTRED_API_TOKEN:-}" ] && [ "${MORTRED_API_HOST}" != "127.0.0.1" ]; then
    echo "[entrypoint] WARNING: non-loopback supervisor without MORTRED_API_TOKEN; it will refuse to start (fail-closed)" >&2
fi
if [ -z "${MORTRED_GATEWAY_AUTH_TOKEN:-}" ] && [ "${MORTRED_GATEWAY_HOST}" != "127.0.0.1" ]; then
    echo "[entrypoint] WARNING: non-loopback gateway without MORTRED_GATEWAY_AUTH_TOKEN; it will refuse to start (fail-closed)" >&2
fi

echo "[entrypoint] starting mortred-supervisor on ${MORTRED_API_HOST}:${MORTRED_API_PORT} (gateway ${MORTRED_GATEWAY_HOST}:${MORTRED_GATEWAY_PORT})"
exec /opt/mortred/bin/mortred-supervisor.out
