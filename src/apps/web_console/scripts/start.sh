#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/generated_configs"

export PATH="/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="$ROOT/3rd_party/libs"

BIN="$ROOT/_bin/app_server"
if [ ! -x "$BIN" ]; then
    echo "==> building backend (standalone) ..."
    cmake -S "$ROOT/src/apps/web_console/backend" -B "$ROOT/src/apps/web_console/backend/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$ROOT/src/apps/web_console/backend/build" -j"$(nproc)" >/dev/null
    BIN="$ROOT/src/apps/web_console/backend/build/app_server"
fi

export APP_PROJECT_ROOT="$ROOT"
export LD_LIBRARY_PATH="$ROOT/_lib:$ROOT/3rd_party/libs"

if [ -f "$ROOT/logs/app_server.pid" ]; then
    kill "$(cat "$ROOT/logs/app_server.pid")" 2>/dev/null || true
    rm -f "$ROOT/logs/app_server.pid"
fi
pkill -x app_server 2>/dev/null || true
sleep 0.3

nohup "$BIN" > "$ROOT/logs/app_server.log" 2>&1 &
echo $! > "$ROOT/logs/app_server.pid"
sleep 1

echo "==> Mortred Web Console: http://localhost:8787  (pid $(cat "$ROOT/logs/app_server.pid"))"
if command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "http://localhost:8787" >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "http://localhost:8787" >/dev/null 2>&1 || true
fi
