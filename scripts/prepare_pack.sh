#!/usr/bin/env bash
# prepare_pack.sh - convert TensorRT engines for the active machine pack only.
#
# Does not convert the whole conf/trt_engines.json zoo. Missing ONNX / trtexec
# / engine fails. --check-engines is exist+nonempty; --ready (default when the
# unified server binary exists) starts each TRT id at worker_nums=1 until /ready.
#
# Usage:
#   ./scripts/prepare_pack.sh
#   ./scripts/prepare_pack.sh --pack conf/packs/demo.toml
#   ./scripts/prepare_pack.sh --force          # reconvert even if present
#   ./scripts/prepare_pack.sh --skip-ready     # convert + nonempty only
#   ./scripts/prepare_pack.sh --require-ready  # fail if the server binary is missing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="${MORTRED_PACK:-$ROOT/conf/packs/demo.toml}"
FORCE=0
SKIP_READY=0
REQUIRE_READY=0

usage() {
    sed -n '2,16p' "$0"
    exit 0
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --pack) PACK="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --skip-ready) SKIP_READY=1; shift ;;
        --require-ready) REQUIRE_READY=1; shift ;;
        -h|--help) usage ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done

if [ ! -f "$PACK" ]; then
    fail "pack not found: $PACK"
fi

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

echo "== prepare pack: $PACK =="
LIST="$("$PY" "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --list)"
if [ -z "$LIST" ] || [ "$LIST" = "# no TensorRT backends in pack" ]; then
    echo "  no TensorRT backends in this pack; skip convert and /ready"
    exit 0
fi
echo "$LIST"

FILTERS="$("$PY" "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --convert-filters)"
CONVERT_ARGS=()
[ "$FORCE" -eq 1 ] && CONVERT_ARGS+=(--force)
CONVERT_ARGS+=(--strict)
if [ -n "$FILTERS" ]; then
    while IFS= read -r token; do
        [ -z "$token" ] && continue
        echo "== convert --only $token =="
        bash "$ROOT/scripts/convert_trt_engines.sh" --only "$token" "${CONVERT_ARGS[@]}"
    done <<<"$FILTERS"
fi

echo "== check engines (exist + nonempty) =="
"$PY" "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --check

SERVER_BIN="$ROOT/_bin/mortred-model-server.out"
if [ ! -x "$SERVER_BIN" ]; then
    SERVER_BIN="$ROOT/bin/mortred-model-server.out"
fi

wait_ready() {
    local port="$1" tries=0
    while [ "$tries" -lt 90 ]; do
        if command -v curl >/dev/null 2>&1 && curl -sf --max-time 1 "http://127.0.0.1:${port}/ready" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        tries=$((tries + 1))
    done
    return 1
}

if [ "$SKIP_READY" -eq 1 ]; then
    echo "  --skip-ready: not probing /ready"
    exit 0
fi

if [ ! -x "$SERVER_BIN" ]; then
    if [ "$REQUIRE_READY" -eq 1 ]; then
        fail "mortred-model-server.out not found (needed for /ready)"
    fi
    echo "  server binary missing; skip /ready (pass --require-ready to fail)"
    exit 0
fi

echo "== /ready (worker_nums=1) =="
export LD_LIBRARY_PATH="$ROOT/_lib:$ROOT/3rd_party/libs:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/logs"
PIDS=()

# SIGTERM trips glog's failure handler (stack dump that looks like a crash).
# Supervisor stop uses SIGINT; match that, then SIGKILL if it will not exit.
stop_probe() {
    local pid="$1"
    [ -n "$pid" ] || return 0
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return 0
    fi
    kill -INT "$pid" 2>/dev/null || true
    local i=0
    while [ "$i" -lt 50 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 0.1
        i=$((i + 1))
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        stop_probe "$pid"
    done
}
trap cleanup EXIT

while IFS=$'\t' read -r model_id engine_path; do
    [ -z "${model_id:-}" ] && continue
    case " ${SEEN_IDS:-} " in
        *" $model_id "*) continue ;;
    esac
    SEEN_IDS="${SEEN_IDS:-} $model_id"
    server_toml="$("$PY" - "$ROOT" "$model_id" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from pack_trt import find_server_toml
p = find_server_toml(Path(sys.argv[1]), sys.argv[2])
print("" if p is None else p)
PY
)"
    [ -n "$server_toml" ] || fail "no conf/server mapping for $model_id"
    port="$(awk -F= '/^port=/ { gsub(/[[:space:]]/, "", $2); print $2; exit }' "$server_toml")"
    [ -n "$port" ] || fail "no port in $server_toml"
    echo "  start $model_id on :$port (engine $(basename "$engine_path"))"
    probe_log="$ROOT/logs/prepare-${model_id}.log"
    MORTRED_WORKER_NUMS=1 MORTRED_PROJECT_ROOT="$ROOT" \
        "$SERVER_BIN" --model "$model_id" "$server_toml" >"$probe_log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    if ! wait_ready "$port"; then
        echo "---- $probe_log ----" >&2
        cat "$probe_log" >&2 || true
        fail "$model_id did not become /ready on :$port (engine may not deserialize)"
    fi
    echo "  [ok] $model_id /ready"
    stop_probe "$pid"
    PIDS=()
done <<<"$LIST"

echo "prepare pack done"
