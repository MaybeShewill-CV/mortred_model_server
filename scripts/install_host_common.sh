#!/usr/bin/env bash
# install_host_common.sh - shared host install steps (sourced, not executed).
# Used by tarball_install.sh and mortredctl_install-host.sh.
# Installs the tree, mortred user, /etc/mortred/supervisor.env (mode 600),
# systemd units, and enable (never start). Does not fetch weights or prepare.

mortred_install_host_fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

mortred_install_host_require_root() {
    local hint="${1:-sudo mortredctl install-host}"
    [ "$(id -u)" -eq 0 ] || mortred_install_host_fail "run as root: $hint"
}

mortred_install_host_ensure_user() {
    id -u mortred >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin mortred
}

mortred_install_host_copy_tree() {
    local prefix="$1" src="$2"
    [ -d "$src" ] || mortred_install_host_fail "install tree missing: $src"
    mkdir -p "$prefix"
    cp -a "$src/." "$prefix/"
    chown -R mortred:mortred "$prefix"
}

mortred_install_host_env() {
    mkdir -p /etc/mortred
    if [ ! -f /etc/mortred/supervisor.env ]; then
        cat > /etc/mortred/supervisor.env <<'EOF'
# REQUIRED (gateway/supervisor refuse to start without all three, distinct):
# MORTRED_API_TOKEN=<management token>
# MORTRED_GATEWAY_AUTH_TOKEN=<inference token>
# MORTRED_METRICS_TOKEN=<metrics scrape token>
# Generate (overwrites this placeholder):
#   sudo /opt/mortred/bin/mortredctl.out init-trust --force --out /etc/mortred/supervisor.env
EOF
        chmod 600 /etc/mortred/supervisor.env
    fi
}

mortred_install_host_supervisor_unit() {
    local prefix="$1" unit_src="$2" profile="${3:-}"
    [ -f "$unit_src" ] || mortred_install_host_fail "missing unit: $unit_src"
    local tmp
    tmp="$(mktemp)"
    cp "$unit_src" "$tmp"
    if [ "$prefix" != "/opt/mortred" ]; then
        sed -i "s|/opt/mortred|$prefix|g" "$tmp"
    fi
    if [ "$profile" = "cpu" ]; then
        sed -i '\|^Environment=MORTRED_PROFILE=|d; s|^\[Service\]|[Service]\nEnvironment=MORTRED_PROFILE=cpu|' \
            "$tmp"
    fi
    cp "$tmp" /etc/systemd/system/mortred-supervisor.service
    rm -f "$tmp"
}

mortred_install_host_require_systemd() {
    command -v systemctl >/dev/null 2>&1 || mortred_install_host_fail "systemctl not found (systemd required)"
}

mortred_install_host_enable_supervisor() {
    mortred_install_host_require_systemd
    systemctl daemon-reload
    systemctl enable mortred-supervisor
}

mortred_install_host_require_nginx() {
    if command -v nginx >/dev/null 2>&1; then
        return 0
    fi
    if [ -x /usr/sbin/nginx ]; then
        return 0
    fi
    mortred_install_host_fail "--with-edge requires nginx"
}

mortred_install_host_edge_unit() {
    local edge_src="$1"
    mortred_install_host_require_nginx
    [ -f "$edge_src" ] || mortred_install_host_fail "missing edge unit: $edge_src"
    mortred_install_host_require_systemd
    cp "$edge_src" /etc/systemd/system/mortred-edge.service
    systemctl daemon-reload
    systemctl enable mortred-edge
}
