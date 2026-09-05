#!/usr/bin/env bash
# verify_deployment.sh - deployment acceptance script (run on the target Linux + GPU machine).
#
# Runs and summarizes, in order:
#   1. Deployment script syntax (bash -n / py_compile)
#   2. Manifest JSON / docker-compose YAML validity
#   3. convert_trt_engines.sh --list (engine manifest completeness)
#   4. fetch_weights.py --dry-run (weights manifest parses, HF path mapping works)
#   5. install_deps.sh --check (3rd_party completeness; enforced only in --full mode)
#   6. fetch_weights.py --check (local weights sha256; missing weights fail only in --full mode)
#   7. live gateway probes (--live mode only: healthz public + inference requires token)
#   8. security_warn.sh --self-test (warning helpers; doctor --strict can fail)
#
# Usage:
#   ./scripts/verify_deployment.sh            # --full: all checks must pass (target machine)
#   ./scripts/verify_deployment.sh --basic    # static checks only (dev box without deps/GPU)
#   ./scripts/verify_deployment.sh --verbose  # print detailed output for each check

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="full"
VERBOSE=0
FAILED=0

for arg in "$@"; do
    case "$arg" in
        --basic) MODE="basic" ;;
        --full) MODE="full" ;;
        --live) MODE="live" ;;
        --verbose) VERBOSE=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Resolve a working python (skip broken PATH stubs like WindowsApps python3 aliases)
resolve_python() {
    local cand p
    for cand in python3 python py; do
        if command -v "$cand" >/dev/null 2>&1; then
            p="$(command -v "$cand")"
            if "$p" -c 'import sys' >/dev/null 2>&1; then
                echo "$p"
                return 0
            fi
        fi
    done
    return 1
}
PY="$(resolve_python)" || { echo "[FAIL] missing a working python3/python"; exit 1; }

check() { # check <name> <cmd...>
    local name="$1"; shift
    if [ "$VERBOSE" -eq 1 ]; then
        echo "==> $name"
        "$@"
    else
        if "$@" >/dev/null 2>&1; then
            echo "  [ok]   $name"
        else
            echo "  [FAIL] $name"
            FAILED=$((FAILED+1))
        fi
    fi
}

echo "== Mortred deployment acceptance (mode=$MODE) =="

# 1) Script syntax
for f in scripts/install_deps.sh scripts/convert_trt_engines.sh \
         scripts/docker_entrypoint.sh scripts/check_repo_clean.sh \
         scripts/clean_artifacts.sh scripts/setup_full_deps.sh \
         scripts/bench_batch.sh scripts/mortredctl_doctor.sh \
         scripts/mortredctl_prepare.sh scripts/prepare_pack.sh \
         scripts/security_warn.sh; do
    check "bash -n $f" bash -n "$ROOT/$f"
done
check "py_compile fetch/gen/check" "$PY" -m py_compile \
    "$ROOT/scripts/fetch_weights.py" "$ROOT/scripts/gen_weights_manifest.py" \
    "$ROOT/scripts/check_consistency.py" "$ROOT/scripts/gen_openapi.py" \
    "$ROOT/scripts/repo_toml.py" "$ROOT/scripts/pack_trt.py"

# 2) Manifest validity
check "JSON: trt_engines/profiles/weights" "$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for f in ["conf/trt_engines.json",
          "conf/trt_profiles/lightglue_extractor.json",
          "conf/trt_profiles/lightglue_matcher.json",
          "conf/weights_manifest.json"]:
    json.loads((root / f).read_text(encoding="utf-8-sig"))
PY
if "$PY" -c "import yaml" >/dev/null 2>&1; then
    # Path passed via argv (MSYS converts standalone args to Windows paths; not inside -c strings)
    check "YAML: docker-compose.yml" "$PY" -c "import yaml,sys; yaml.safe_load(open(sys.argv[1], encoding='utf-8'))" "$ROOT/docker-compose.yml"
else
    echo "  [warn] yaml module missing, skipping docker-compose.yml validation (pip install pyyaml)"
fi

# 3) Engine manifest
check "convert_trt_engines.sh --list" bash "$ROOT/scripts/convert_trt_engines.sh" --list

# 4) Weights manifest dry-run
check "fetch_weights.py --dry-run" "$PY" "$ROOT/scripts/fetch_weights.py" --dry-run
check "security_warn.sh --self-test" bash "$ROOT/scripts/security_warn.sh" --self-test

# 5) 3rd_party completeness (failure is only a warning in --basic mode)
if [ "$MODE" = "full" ]; then
    check "install_deps.sh --check" bash "$ROOT/scripts/install_deps.sh" --check
else
    if bash "$ROOT/scripts/install_deps.sh" --check >/dev/null 2>&1; then
        echo "  [ok]   install_deps.sh --check"
    else
        echo "  [warn] install_deps.sh --check failed (basic mode: missing deps are expected)"
    fi
fi

# 6) Local weights check (--basic mode: sample small files to verify the mechanism, no full hashing)
if [ "$MODE" = "full" ]; then
    check "fetch_weights.py --check" "$PY" "$ROOT/scripts/fetch_weights.py" --check
else
    if "$PY" "$ROOT/scripts/fetch_weights.py" --check --only bpe_simple_vocab >/dev/null 2>&1; then
        echo "  [ok]   fetch_weights.py --check (sampled bpe_simple_vocab)"
    else
        echo "  [warn] fetch_weights.py sampled check failed (run fetch_weights.py to download weights first)"
    fi
fi

# 7) Live gateway probes (--live): /healthz must be public; model routes must
#    reject unauthenticated inference with 401 (fail-closed gateway contract).
if [ "$MODE" = "live" ]; then
    GATEWAY="${MORTRED_GATEWAY_ADDR:-127.0.0.1:8080}"
    route="$(grep -rh -m1 'server_uri' "$ROOT/conf/server" 2>/dev/null | head -1 | sed 's/.*"\(\/[^\"]*\)".*/\1/')"
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://$GATEWAY/healthz" || echo 000)"
    if [ "$code" = "200" ]; then
        echo "  [ok]   gateway /healthz public (200)"
    else
        echo "  [FAIL] gateway /healthz expected 200, got $code ($GATEWAY)"
        FAILED=$((FAILED+1))
    fi
    if [ -n "$route" ]; then
        code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$GATEWAY$route" || echo 000)"
        if [ "$code" = "401" ]; then
            echo "  [ok]   gateway inference requires token (401 on $route)"
        else
            echo "  [FAIL] gateway $route unauthenticated POST expected 401, got $code"
            FAILED=$((FAILED+1))
        fi
    fi
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "== acceptance passed (mode=$MODE) =="
    exit 0
fi
echo "== acceptance failed: $FAILED check(s) failed (see above) =="
exit 1
