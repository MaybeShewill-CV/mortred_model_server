#!/usr/bin/env bash
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"
if [ -f "$ROOT/logs/app_server.pid" ]; then
    kill "$(cat "$ROOT/logs/app_server.pid")" 2>/dev/null || true
    rm -f "$ROOT/logs/app_server.pid"
fi
pkill -x app_server 2>/dev/null || true
echo "==> web console stopped"
