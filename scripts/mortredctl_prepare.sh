#!/usr/bin/env bash
# mortredctl_prepare.sh - `mortredctl prepare` entry.
# Checks the pack file and arguments before the downstream convert.
# Does not change TensorRT convert semantics.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
ROOT="$(cd "$ROOT" && pwd)"
PACK="${MORTRED_PACK:-$ROOT/conf/packs/demo.toml}"
ORIG=("$@")

usage() {
    cat <<'EOF'
mortredctl prepare — pack-scoped TensorRT prepare (local engines).

Usage:
  mortredctl prepare [--pack FILE] [--force] [--skip-ready] [--require-ready]

The entry checks that the pack file exists, then runs prepare_pack.sh.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --pack)
            if [ $# -lt 2 ]; then
                echo "[ERROR] step: args missing value for --pack" >&2
                exit 1
            fi
            PACK="$2"
            shift 2
            ;;
        --force|--skip-ready|--require-ready)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] step: args unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$PACK" ]; then
    echo "[ERROR] step: pack-file missing: $PACK" >&2
    exit 1
fi

bash "$ROOT/scripts/prepare_pack.sh" "${ORIG[@]}"
