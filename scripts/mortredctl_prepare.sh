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
  mortredctl prepare --id ID [--force]

--id converts that one product static_bs1 engine on this machine.
It cannot be combined with --pack. Without --id, prepare converts the current pack.
EOF
}

SAW_PACK=0
ID=""
while [ $# -gt 0 ]; do
    case "$1" in
        --pack)
            if [ $# -lt 2 ]; then
                echo "[ERROR] step: args missing value for --pack" >&2
                exit 1
            fi
            SAW_PACK=1
            PACK="$2"
            shift 2
            ;;
        --id)
            if [ $# -lt 2 ]; then
                echo "[ERROR] step: args missing value for --id" >&2
                exit 1
            fi
            ID="$2"
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

if [ -n "$ID" ]; then
    if [ "$SAW_PACK" -eq 1 ]; then
        echo "[ERROR] step: --id and --pack cannot be combined" >&2
        exit 1
    fi
    if ! python3 "$ROOT/scripts/product_static_bs1.py" --lookup-id "$ID" >/dev/null 2>&1; then
        echo "[ERROR] step: unknown product id: $ID" >&2
        exit 1
    fi
elif [ ! -f "$PACK" ]; then
    echo "[ERROR] step: pack-file missing: $PACK" >&2
    exit 1
fi

bash "$ROOT/scripts/prepare_pack.sh" "${ORIG[@]}"
