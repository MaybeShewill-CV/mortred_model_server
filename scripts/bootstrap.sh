#!/usr/bin/env bash
# bootstrap.sh - the one-line first-hour entry point. Thin by design: it only
# DETECTS the environment and delegates to one of the standard tracks; all
# real logic lives in mortredctl init / install.sh / compose (single core,
# three entries - never three divergent paths).
#
#   curl -fsSL https://raw.githubusercontent.com/MaybeShewill-CV/mortred_model_server/main/scripts/bootstrap.sh | bash
#
# Track selection:
#   docker present -> docker compose --profile <cpu|gpu> up (builds locally)
#   no docker      -> download the latest release tarball + sudo ./install.sh
#   neither possible -> printed manual path (source build)
set -uo pipefail

REPO="MaybeShewill-CV/mortred_model_server"
DL_BASE="https://github.com/$REPO/releases/latest/download"

echo "== Mortred bootstrap =="

# ---- profile detection ----
PROFILE="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    PROFILE="gpu"
fi
echo "  detected profile: $PROFILE"

# ---- track 1: docker ----
if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1; then
        echo "== docker track =="
        cat <<EOF
next:
  1. git clone https://github.com/$REPO.git && cd mortred_model_server
  2. python3 scripts/fetch_weights.py --profile $PROFILE
  3. MORTRED_API_TOKEN=<mgmt> MORTRED_GATEWAY_AUTH_TOKEN=<infer> \\
         docker compose --profile $PROFILE up -d
  4. curl -fs http://localhost:8787/api/v1/health
EOF
        exit 0
    fi
fi

# ---- track 2: release tarball ----
if command -v curl >/dev/null 2>&1; then
    TGZ="mortred_model_server-latest-$PROFILE-linux-x64.tar.gz"
    if curl -fsSL --max-time 15 -o /dev/null "$DL_BASE/$TGZ"; then
        echo "== tarball track =="
        WORK="$(mktemp -d)"
        curl -fSL "$DL_BASE/$TGZ" -o "$WORK/$TGZ"
        curl -fsSL "$DL_BASE/$TGZ.sha256" -o "$WORK/$TGZ.sha256" 2>/dev/null \
            && (cd "$WORK" && sha256sum -c "$TGZ.sha256") \
            || echo "  [WARN] no published sha256; proceeding unverified" >&2
        tar -xzf "$WORK/$TGZ" -C "$WORK"
        cd "$WORK"
        echo "== running installer (needs sudo) =="
        exec sudo ./install.sh
    fi
    echo "  [WARN] no release tarball published yet for profile $PROFILE" >&2
fi

# ---- track 3: manual ----
cat <<EOF
== manual track ==
  1. git clone https://github.com/$REPO.git && cd mortred_model_server
  2. ./scripts/install_deps.sh $( [ "$PROFILE" = "cpu" ] && echo --cpu ) --all
  3. cmake --preset $( [ "$PROFILE" = "cpu" ] && echo full-cpu || echo full ) && cmake --build --preset $( [ "$PROFILE" = "cpu" ] && echo full-cpu || echo full )
  4. mortredctl init --profile $PROFILE
EOF
exit 0