#!/usr/bin/env bash
# mortredctl_calibrate.sh - `mortredctl calibrate` → pack worker_nums report
# (optional --write-pack updates the pack file, never conf/server).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +e
python3 "$ROOT/scripts/calibrate_pack.py" "$@"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
    echo "next: mortredctl next" >&2
fi
exit "$rc"
