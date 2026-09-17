#!/usr/bin/env bash
# mortredctl_down.sh - `mortredctl down` → stop model servers / the whole
# control plane. Shares the implementation with `mortredctl ps`.
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mortredctl_ps.py" down "$@"
