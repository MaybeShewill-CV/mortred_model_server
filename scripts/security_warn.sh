#!/usr/bin/env bash
# security_warn.sh - plaintext-HTTP / token-quality warnings for `mortredctl doctor`.
# Default exit 0 (except --self-test). --strict exits 1 if any warning fired.
# Fail-closed process start is separate. Does not print token values.

set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WARNED=0

# Drop CR (Windows/WSL checkouts), surrounding quotes, and edge spaces.
sanitize_scalar() {
    local s
    s="$(printf '%s' "${1:-}" | tr -d '\r')"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    s="${s#\"}"
    s="${s%\"}"
    s="${s#\'}"
    s="${s%\'}"
    printf '%s' "$s"
}

is_loopback_host() {
    local h c i dots rest
    h="$(sanitize_scalar "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "$h" in
        localhost|::1|'[::1]') return 0 ;;
    esac
    case "$h" in
        127.*)
            rest="${h#127.}"
            dots=0
            i=0
            while [ "$i" -lt "${#rest}" ]; do
                c="${rest:$i:1}"
                if [ "$c" = "." ]; then
                    dots=$((dots + 1))
                else
                    case "$c" in
                        [0-9]) ;;
                        *) return 1 ;;
                    esac
                fi
                i=$((i + 1))
            done
            [ "$dots" -eq 2 ]
            return
            ;;
    esac
    return 1
}

warn() {
    echo "  [warn] $*"
    WARNED=$((WARNED + 1))
}

# First KEY=VALUE in a file wins only when the name is not already in the
# environment (process env overrides files, matching C++ getenv).
load_kv_file() {
    local f="$1" line key val
    [ -n "$f" ] && [ -f "$f" ] && [ -r "$f" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
        esac
        case "$line" in
            export\ *) line="${line#export }" ;;
        esac
        case "$line" in
            MORTRED_API_TOKEN=*|MORTRED_GATEWAY_AUTH_TOKEN=*|MORTRED_METRICS_TOKEN=*|MORTRED_GATEWAY_HOST=*|MORTRED_API_HOST=*)
                key="${line%%=*}"
                val="${line#*=}"
                val="${val%\"}"
                val="${val#\"}"
                val="${val%\'}"
                val="${val#\'}"
                val="$(sanitize_scalar "$val")"
                if [ -z "${!key:-}" ]; then
                    export "$key=$val"
                fi
                ;;
        esac
    done <"$f"
}

toml_section_key() {
    local file="$1" section="$2" key="$3"
    [ -f "$file" ] || return 0
    awk -v sec="$section" -v want="$key" '
        { gsub(/\r/, "") }
        /^[[:space:]]*#/ { next }
        /^\[/ {
            insec = ($0 ~ "^\\[" sec "\\]")
            next
        }
        insec {
            line = $0
            sub(/[[:space:]]*#.*$/, "", line)
            if (line ~ "^[[:space:]]*" want "[[:space:]]*=") {
                sub(/^[^=]*=[[:space:]]*/, "", line)
                sub(/^"/, "", line)
                sub(/"$/, "", line)
                print line
                exit
            }
        }
    ' "$file"
}

host_from_ss() {
    local port="$1"
    [ "${SECURITY_WARN_SKIP_SS:-}" = "1" ] && return 0
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | grep -Eq "0\\.0\\.0\\.0:${port}([^0-9]|\$)|\\[::\\]:${port}([^0-9]|\$)"; then
        echo "0.0.0.0"
    fi
}

run_warnings() {
    local gw_host api_host api_tok gw_tok metrics_tok ss_host toml="$ROOT/conf/mortred.toml"

    load_kv_file "/etc/mortred/supervisor.env"
    load_kv_file "$ROOT/conf/supervisor.env"

    gw_host="${MORTRED_GATEWAY_HOST:-}"
    if [ -z "$(sanitize_scalar "$gw_host")" ]; then
        gw_host="$(toml_section_key "$toml" gateway host || true)"
    fi
    gw_host="$(sanitize_scalar "$gw_host")"
    gw_host="${gw_host:-127.0.0.1}"

    api_host="${MORTRED_API_HOST:-}"
    if [ -z "$(sanitize_scalar "$api_host")" ]; then
        api_host="$(toml_section_key "$toml" supervisor api_host || true)"
    fi
    api_host="$(sanitize_scalar "$api_host")"
    api_host="${api_host:-127.0.0.1}"

    if ! is_loopback_host "$gw_host"; then
        warn "gateway listen $gw_host is not loopback (plain HTTP). Terminate TLS at Nginx; see deploy/nginx and mortredctl init-edge"
    fi
    if ! is_loopback_host "$api_host"; then
        warn "supervisor listen $api_host is not loopback (plain HTTP). Terminate TLS at Nginx; see deploy/nginx and mortredctl init-edge"
    fi

    ss_host="$(host_from_ss 8080 || true)"
    if [ "$ss_host" = "0.0.0.0" ] && is_loopback_host "$gw_host"; then
        warn "host has 0.0.0.0:8080 listening (compose/docker publish). Bearer tokens are plaintext on that interface; bind 127.0.0.1 or put Nginx in front (mortredctl init-edge)"
    fi
    ss_host="$(host_from_ss 8787 || true)"
    if [ "$ss_host" = "0.0.0.0" ] && is_loopback_host "$api_host"; then
        warn "host has 0.0.0.0:8787 listening. Keep the management port off the public internet; see deploy/nginx"
    fi

    api_tok="${MORTRED_API_TOKEN:-}"
    gw_tok="${MORTRED_GATEWAY_AUTH_TOKEN:-}"
    if [ -n "$api_tok" ] && [ "${#api_tok}" -lt 32 ]; then
        warn "MORTRED_API_TOKEN is shorter than 32 characters"
    fi
    if [ -n "$gw_tok" ] && [ "${#gw_tok}" -lt 32 ]; then
        warn "MORTRED_GATEWAY_AUTH_TOKEN is shorter than 32 characters"
    fi
    if [ -n "$api_tok" ] && [ -n "$gw_tok" ] && [ "$api_tok" = "$gw_tok" ]; then
        warn "MORTRED_API_TOKEN and MORTRED_GATEWAY_AUTH_TOKEN are identical; use two independent values"
    fi

    metrics_tok="${MORTRED_METRICS_TOKEN:-}"
    if [ -z "$metrics_tok" ]; then
        warn "MORTRED_METRICS_TOKEN is unset; the gateway refuses to start (GET /metrics is never public). Run mortredctl init-trust"
    fi
    if [ -n "$metrics_tok" ] && [ "${#metrics_tok}" -lt 32 ]; then
        warn "MORTRED_METRICS_TOKEN is shorter than 32 characters"
    fi
    if [ -n "$metrics_tok" ] && [ -n "$gw_tok" ] && [ "$metrics_tok" = "$gw_tok" ]; then
        warn "MORTRED_METRICS_TOKEN matches MORTRED_GATEWAY_AUTH_TOKEN; Prometheus would then hold inference privilege"
    fi
    if [ -n "$metrics_tok" ] && [ -n "$api_tok" ] && [ "$metrics_tok" = "$api_tok" ]; then
        warn "MORTRED_METRICS_TOKEN matches MORTRED_API_TOKEN; Prometheus would then hold management privilege"
    fi

    if [ "$WARNED" -eq 0 ]; then
        echo "  [ok]   security warnings (loopback listen / token length)"
    fi
}

run_self_test() {
    local failed=0
    expect_loopback() {
        if is_loopback_host "$1"; then
            echo "  [ok]   loopback $1"
        else
            echo "  [FAIL] expected loopback: $1"
            failed=$((failed + 1))
        fi
    }
    expect_not_loopback() {
        if is_loopback_host "$1"; then
            echo "  [FAIL] expected non-loopback: $1"
            failed=$((failed + 1))
        else
            echo "  [ok]   non-loopback $1"
        fi
    }
    expect_loopback 127.0.0.1
    expect_loopback "$(printf '127.0.0.1\r')"
    expect_loopback 127.0.0.2
    expect_loopback localhost
    expect_loopback ::1
    expect_loopback '[::1]'
    expect_not_loopback 0.0.0.0
    expect_not_loopback 192.168.1.1
    expect_not_loopback 127.evil
    expect_not_loopback 127.0.0
    expect_not_loopback 10.0.0.1

    # Isolated warning run: non-loopback + short + identical tokens.
    local out
    out="$(
        SECURITY_WARN_SKIP_SS=1 \
        MORTRED_GATEWAY_HOST=0.0.0.0 \
        MORTRED_API_HOST=0.0.0.0 \
        MORTRED_API_TOKEN=tokA \
        MORTRED_GATEWAY_AUTH_TOKEN=tokA \
        bash "$ROOT/scripts/security_warn.sh"
    )"
    echo "$out" | grep -q 'gateway listen 0.0.0.0' || {
        echo "  [FAIL] missing gateway non-loopback warning"
        failed=$((failed + 1))
    }
    echo "$out" | grep -q 'supervisor listen 0.0.0.0' || {
        echo "  [FAIL] missing supervisor non-loopback warning"
        failed=$((failed + 1))
    }
    echo "$out" | grep -q 'MORTRED_API_TOKEN is shorter' || {
        echo "  [FAIL] missing short API token warning"
        failed=$((failed + 1))
    }
    echo "$out" | grep -q 'identical' || {
        echo "  [FAIL] missing identical-token warning"
        failed=$((failed + 1))
    }
    echo "$out" | grep -q 'MORTRED_METRICS_TOKEN is unset' || {
        echo "  [FAIL] missing unset-metrics warning"
        failed=$((failed + 1))
    }
    if echo "$out" | grep -q 'tokA'; then
        echo "  [FAIL] warning output leaked a token value"
        failed=$((failed + 1))
    else
        echo "  [ok]   warnings do not print token values"
    fi

    out="$(
        SECURITY_WARN_SKIP_SS=1 \
        MORTRED_GATEWAY_HOST=127.0.0.1 \
        MORTRED_API_HOST=127.0.0.1 \
        MORTRED_METRICS_TOKEN="$(printf 'c%.0s' {1..32})" \
        MORTRED_API_TOKEN="$(printf 'a%.0s' {1..32})" \
        MORTRED_GATEWAY_AUTH_TOKEN="$(printf 'b%.0s' {1..32})" \
        bash "$ROOT/scripts/security_warn.sh"
    )"
    echo "$out" | grep -q '\[ok\]' || {
        echo "  [FAIL] loopback + distinct 32-char tokens should be clean"
        failed=$((failed + 1))
        echo "$out"
    }

    if SECURITY_WARN_SKIP_SS=1 \
        MORTRED_GATEWAY_HOST=127.0.0.1 \
        MORTRED_API_HOST=127.0.0.1 \
        MORTRED_API_TOKEN="$(printf 'a%.0s' {1..32})" \
        MORTRED_GATEWAY_AUTH_TOKEN="$(printf 'b%.0s' {1..32})" \
        MORTRED_METRICS_TOKEN="$(printf 'c%.0s' {1..32})" \
        bash "$ROOT/scripts/security_warn.sh" --strict; then
        echo "  [ok]   --strict exits 0 on loopback + distinct tokens"
    else
        echo "  [FAIL] --strict should pass on loopback + distinct 32-char tokens"
        failed=$((failed + 1))
    fi
    if SECURITY_WARN_SKIP_SS=1 \
        MORTRED_GATEWAY_HOST=0.0.0.0 \
        MORTRED_API_HOST=127.0.0.1 \
        MORTRED_API_TOKEN="$(printf 'a%.0s' {1..32})" \
        MORTRED_GATEWAY_AUTH_TOKEN="$(printf 'b%.0s' {1..32})" \
        MORTRED_METRICS_TOKEN="$(printf 'c%.0s' {1..32})" \
        bash "$ROOT/scripts/security_warn.sh" --strict; then
        echo "  [FAIL] --strict should fail on non-loopback gateway (plaintext HTTP warning)"
        failed=$((failed + 1))
    else
        echo "  [ok]   --strict exits 1 on non-loopback gateway"
    fi

    crlf_toml="$(mktemp)"
    printf '[gateway]\r\nhost = "127.0.0.1"\r\n[supervisor]\r\napi_host = "127.0.0.1"\r\n' >"$crlf_toml"
    if [ "$(toml_section_key "$crlf_toml" gateway host)" = "127.0.0.1" ] \
        && [ "$(toml_section_key "$crlf_toml" supervisor api_host)" = "127.0.0.1" ]; then
        echo "  [ok]   CRLF toml host parse"
    else
        echo "  [FAIL] CRLF toml host parse"
        failed=$((failed + 1))
    fi
    rm -f "$crlf_toml"

    if [ "$failed" -ne 0 ]; then
        echo "== security_warn self-test failed: $failed =="
        return 1
    fi
    echo "== security_warn self-test passed =="
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    run_self_test
    exit $?
fi

STRICT=0
if [ "${1:-}" = "--strict" ]; then
    STRICT=1
fi

if [ "$STRICT" = 1 ]; then
    echo "== security warnings (--strict: warnings fail doctor) =="
else
    echo "== security warnings (never fail doctor unless --strict) =="
fi
run_warnings
if [ "$STRICT" = 1 ] && [ "$WARNED" -gt 0 ]; then
    echo "[FAIL] $WARNED security warning(s); mortredctl doctor --strict"
    exit 1
fi
exit 0
