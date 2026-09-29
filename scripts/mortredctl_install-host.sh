#!/usr/bin/env bash
# mortredctl_install-host.sh - host install (invoked by `mortredctl install-host`).
#
# Requires root. Copies the install tree, creates the mortred user, writes
# /etc/mortred/supervisor.env (mode 600 if created), installs systemd units,
# and enables them. Does not start units. Does not fetch weights or prepare.
#
#   mortredctl install-host --from-tarball-root DIR
#   mortredctl install-host --from-build DIR
#   sudo bash scripts/mortredctl_install-host.sh --prefix /opt/mortred \
#       --from-tarball-root unpack [--with-edge]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=install_host_common.sh
. "$ROOT/scripts/install_host_common.sh"

PREFIX="/opt/mortred"
FROM_TARBALL=""
FROM_BUILD=""
WITH_EDGE=0

usage() {
    sed -n '2,11p' "$0"
    echo ""
    echo "usage: mortredctl install-host [--prefix DIR] --from-tarball-root DIR | --from-build DIR [--with-edge]"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --from-tarball-root)
            FROM_TARBALL="$2"
            shift 2
            ;;
        --from-build)
            FROM_BUILD="$2"
            shift 2
            ;;
        --with-edge)
            WITH_EDGE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -n "$FROM_TARBALL" ] && [ -n "$FROM_BUILD" ]; then
    echo "[ERROR] pass only one of --from-tarball-root or --from-build" >&2
    exit 1
fi
if [ -z "$FROM_TARBALL" ] && [ -z "$FROM_BUILD" ]; then
    echo "[ERROR] need --from-tarball-root DIR or --from-build DIR" >&2
    exit 1
fi

mortred_install_host_require_root "sudo mortredctl install-host"

if [ "$WITH_EDGE" -eq 1 ]; then
    mortred_install_host_require_nginx
fi

TREE_SRC=""
UNIT_SRC=""
EDGE_SRC="$ROOT/deploy/nginx/mortred-edge.service"
PROFILE=""

if [ -n "$FROM_TARBALL" ]; then
    FROM_TARBALL="$(cd "$FROM_TARBALL" && pwd)"
    TREE_SRC="$FROM_TARBALL/opt/mortred"
    UNIT_SRC="$FROM_TARBALL/deploy/mortred-supervisor.service"
    if [ -f "$FROM_TARBALL/deploy/nginx/mortred-edge.service" ]; then
        EDGE_SRC="$FROM_TARBALL/deploy/nginx/mortred-edge.service"
    fi
    PROFILE="$(cat "$TREE_SRC/PROFILE" 2>/dev/null || echo gpu)"
    echo "== install-host (tarball-root: $FROM_TARBALL -> $PREFIX) =="
    mortred_install_host_ensure_user
    mortred_install_host_copy_tree "$PREFIX" "$TREE_SRC"
elif [ -n "$FROM_BUILD" ]; then
    FROM_BUILD="$(cd "$FROM_BUILD" && pwd)"
    command -v cmake >/dev/null 2>&1 || mortred_install_host_fail "cmake not found (needed for --from-build)"
    UNIT_SRC="$ROOT/deploy/mortred-supervisor.service"
    echo "== install-host (cmake --install $FROM_BUILD --prefix $PREFIX) =="
    mortred_install_host_ensure_user
    cmake --install "$FROM_BUILD" --prefix "$PREFIX"
    chown -R mortred:mortred "$PREFIX"
    PROFILE="$(cat "$PREFIX/PROFILE" 2>/dev/null || true)"
fi

mortred_install_host_env
mortred_install_host_supervisor_unit "$PREFIX" "$UNIT_SRC" "$PROFILE"
mortred_install_host_enable_supervisor

if [ "$WITH_EDGE" -eq 1 ]; then
    mortred_install_host_edge_unit "$EDGE_SRC"
fi

echo "== install-host done (enabled, not started) =="
echo "next: fill /etc/mortred/supervisor.env (mortredctl init-trust), then sudo systemctl start mortred-supervisor"
