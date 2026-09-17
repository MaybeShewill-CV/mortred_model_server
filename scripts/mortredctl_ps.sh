#!/usr/bin/env bash
# mortredctl_ps.sh - `mortredctl ps` / `mortredctl down` → control-plane probe
# and shutdown (supervisor, gateway, model servers). See the python source
# for flags; everything degrades gracefully without a token.
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mortredctl_ps.py" "$@"
