#!/usr/bin/env bash
# check_api_keys_hashing.sh - guard against weak API keys in conf/api_keys.toml.
#
# The gateway compares an unsalted SHA-256 of the presented bearer secret, so
# key entropy IS the security. Two detectable failure modes:
#   1. a config that still carries the hash of the documented example secret
#      ("your-secret-key") — a leaked file then hands over a working key;
#   2. a `hash` value that is not 64 lowercase hex chars (truncated / copied
#      from the wrong field).
#
#   ./scripts/check_api_keys_hashing.sh [FILE]     # default conf/api_keys.toml
#   ./scripts/check_api_keys_hashing.sh --self-test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_TEST=0
FILE=""

case "${1:-}" in
    --self-test) SELF_TEST=1 ;;
    -h|--help)
        echo "usage: check_api_keys_hashing.sh [FILE] | --self-test"; exit 0 ;;
    "") ;;
    *) FILE="$1" ;;
esac

KNOWN_BAD_HASH="$(printf 'your-secret-key' | sha256sum | cut -d' ' -f1)"

check_file() {  # path -> 0 ok / 1 problem
    local path="$1" fails=0 hash_lines line value
    if [ ! -f "$path" ]; then
        echo "no api key file at $path (static tokens only)"
        return 0
    fi
    if grep -qiF "$KNOWN_BAD_HASH" "$path"; then
        echo "[ERROR] $path contains the hash of the documented example secret" >&2
        echo "  that key is public: mint a random one with ./scripts/mint_api_key.sh" >&2
        fails=$((fails + 1))
    fi
    hash_lines="$(grep -cE '^[[:space:]]*hash[[:space:]]*=' "$path" || true)"
    if [ "$hash_lines" -eq 0 ]; then
        echo "[warn] $path has no [keys.*] entries: it is not authentication" >&2
        return 0
    fi
    while IFS= read -r line; do
        value="$(printf '%s' "$line" | sed -E 's/^[[:space:]]*hash[[:space:]]*=[[:space:]]*"?([0-9a-fA-F]+)"?.*/\1/')"
        if ! printf '%s' "$value" | grep -qE '^[0-9a-f]{64}$'; then
            echo "[ERROR] $path: hash value is not 64 lowercase hex chars: $line" >&2
            fails=$((fails + 1))
        fi
    done < <(grep -E '^[[:space:]]*hash[[:space:]]*=' "$path")
    if [ "$fails" -eq 0 ]; then
        echo "$path: $hash_lines key hash(es) OK (64 hex, no known-weak example)"
        return 0
    fi
    return 1
}

self_test() {
    local fails=0 tmp good bad short nonhex rc
    tmp="$(mktemp -d)"
    good="$(printf 'k' | sha256sum | cut -d' ' -f1)"
    printf '[keys.a]\nhash = "%s"\nscope = "inference"\n' "$good" > "$tmp/good.toml"
    printf '[keys.a]\nhash = "%s"\n' "$KNOWN_BAD_HASH" > "$tmp/bad.toml"
    printf '[keys.a]\nhash = "%s"\n' "${good:0:32}" > "$tmp/short.toml"
    printf '[keys.a]\nhash = "zz%s"\n' "${good:2}" > "$tmp/nonhex.toml"
    printf '# only comments\n' > "$tmp/empty.toml"

    check_file "$tmp/good.toml" >/dev/null 2>&1; rc=$?
    [ "$rc" -eq 0 ] || { echo "FAIL: a good key file was rejected"; fails=$((fails + 1)); }
    for bad in bad short nonhex; do
        check_file "$tmp/$bad.toml" >/dev/null 2>&1 && {
            echo "FAIL: $bad.toml was accepted"; fails=$((fails + 1)); }
    done
    check_file "$tmp/missing.toml" >/dev/null 2>&1; rc=$?
    [ "$rc" -eq 0 ] || { echo "FAIL: a missing file must not fail"; fails=$((fails + 1)); }
    check_file "$tmp/empty.toml" >/dev/null 2>&1; rc=$?
    [ "$rc" -eq 0 ] || { echo "FAIL: a comment-only file must not fail"; fails=$((fails + 1)); }

    rm -rf "$tmp"
    echo "SUMMARY FAILS=$fails"
    return "$fails"
}

if [ "$SELF_TEST" -eq 1 ]; then
    self_test
    exit $?
fi

check_file "${FILE:-$ROOT/conf/api_keys.toml}"
