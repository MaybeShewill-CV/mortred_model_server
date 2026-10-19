#!/usr/bin/env bash
# mint_api_key.sh - mint a high-entropy API key for conf/api_keys.toml.
#
# The gateway verifies keys by comparing an unsalted SHA-256 of the presented
# bearer secret (fast by design: this is a bearer token, not a password). That
# is only safe when the SECRET carries the entropy, so NEVER mint a key from a
# memorable phrase: a leaked config makes sha256("your-secret-key") trivially
# reversible. This script generates 32 random bytes (64 hex chars).
#
#   ./scripts/mint_api_key.sh --name client-a --scope inference
#   ./scripts/mint_api_key.sh --name ci --scope inference --append conf/api_keys.toml
#
# The plaintext key is printed once; only the hash is written.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="client-a"
SCOPE="inference"
OUT=""
APPEND=0
SELF_TEST=0

usage() {
    cat <<'USAGE'
usage: mint_api_key.sh [--name NAME] [--scope inference|admin] [--append FILE]
       mint_api_key.sh --self-test
  prints the plaintext key once plus the [keys.NAME] block for conf/api_keys.toml
  --append FILE  append the block to FILE (mode 600) instead of stdout
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --scope) SCOPE="$2"; shift 2 ;;
        --append) OUT="$2"; APPEND=1; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

mint_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    else
        head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
    fi
}

hash_of() {
    printf '%s' "$1" | sha256sum | cut -d' ' -f1
}

key_block() {  # name hash scope
    printf '[keys.%s]\nhash = "%s"\nscope = "%s"\nenabled = true\n' "$1" "$2" "$3"
}

self_test() {
    local fails=0 a b ha hb block mode
    a="$(mint_key)"
    b="$(mint_key)"
    [[ "$a" =~ ^[0-9a-f]{64}$ ]] || { echo "FAIL: minted key is not 64 hex chars ($a)"; fails=$((fails + 1)); }
    [ "$a" != "$b" ] || { echo "FAIL: two mints produced the same key"; fails=$((fails + 1)); }
    ha="$(hash_of "$a")"
    [[ "$ha" =~ ^[0-9a-f]{64}$ ]] || { echo "FAIL: hash is not 64 hex chars"; fails=$((fails + 1)); }
    [ "$ha" = "$(hash_of "$a")" ] || { echo "FAIL: hash is not deterministic"; fails=$((fails + 1)); }
    block="$(key_block "$NAME" "$ha" "$SCOPE")"
    printf '%s\n' "$block" | grep -q "^hash = \"$ha\"$" || { echo "FAIL: block misses the hash"; fails=$((fails + 1)); }
    printf '%s\n' "$block" | grep -q "^scope = \"$SCOPE\"$" || { echo "FAIL: block misses the scope"; fails=$((fails + 1)); }
    local tmp
    tmp="$(mktemp -d)"
    OUT="$tmp/api_keys.toml"
    ( umask 077; printf '%s\n' "$block" > "$OUT" )
    mode="$(stat -c %a "$OUT")"
    [ "$mode" = "600" ] || { echo "FAIL: appended file mode $mode != 600"; fails=$((fails + 1)); }
    rm -rf "$tmp"
    echo "SUMMARY FAILS=$fails"
    return "$fails"
}

if [ "$SELF_TEST" -eq 1 ]; then
    self_test
    exit $?
fi

KEY="$(mint_key)"
HASH="$(hash_of "$KEY")"
BLOCK="$(key_block "$NAME" "$HASH" "$SCOPE")"

if [ "$APPEND" -eq 1 ]; then
    mkdir -p "$(dirname "$OUT")"
    ( umask 077; printf '\n%s\n' "$BLOCK" >> "$OUT" )
    chmod 600 "$OUT" 2>/dev/null || true
    echo "appended [keys.$NAME] to $OUT (mode 600)"
    echo "the key is printed ONCE and is not recoverable from the file:"
else
    echo "add this block to conf/api_keys.toml:"
    printf '\n%s\n\n' "$BLOCK"
    echo "the key is printed ONCE (only its hash is stored):"
fi
printf '  %s\n' "$KEY"
echo
echo "note: verification is unsalted SHA-256 by design, so never reuse a"
echo "memorable phrase as the key; rotate by minting a new one and removing the old"
echo "block (the gateway hot-reloads conf/api_keys.toml)."
