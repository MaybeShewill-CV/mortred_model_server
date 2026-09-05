#!/usr/bin/env bash
# mortredctl_prepare.sh - `mortredctl prepare` → pack-scoped TRT convert + /ready.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_pack.sh" "$@"
