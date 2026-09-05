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
# containers autostart the machine pack (default demo), not the whole catalog
export MORTRED_AUTOSTART="${MORTRED_AUTOSTART:-true}"
export MORTRED_PACK="${MORTRED_PACK:-$APP_PROJECT_ROOT/conf/packs/demo.toml}"
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

# Optional gpu convenience: convert MISSING TRT engines before the supervisor
# autostarts (conversion is minutes-long and hardware-specific, so it is an
# explicit opt-in). No-op in the cpu profile (no engines there).
if [ "${MORTRED_AUTO_BUILD_ENGINES:-false}" = "true" ] \
        && [ "${MORTRED_PROFILE:-gpu}" = "gpu" ] \
        && [ -x /opt/mortred/scripts/convert_trt_engines.sh ]; then
    echo "[entrypoint] MORTRED_AUTO_BUILD_ENGINES=true: converting missing TRT engines"
    /opt/mortred/scripts/convert_trt_engines.sh || {
        echo "[entrypoint] WARNING: engine conversion failed; TRT models will not start" >&2
    }
fi

echo "[entrypoint] starting mortred-supervisor on ${MORTRED_API_HOST}:${MORTRED_API_PORT} (gateway ${MORTRED_GATEWAY_HOST}:${MORTRED_GATEWAY_PORT})"
exec /opt/mortred/bin/mortred-supervisor.out
