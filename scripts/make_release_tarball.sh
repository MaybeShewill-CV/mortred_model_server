#!/usr/bin/env bash
# make_release_tarball.sh - build a self-contained binary release tarball.
#
# Usage (from the repo root, AFTER a successful full build with
# MORTRED_INSTALL=ON and the wanted profile):
#   ./scripts/make_release_tarball.sh gpu 0.1.0 [build-dir]
#   ./scripts/make_release_tarball.sh cpu 0.1.0 [build-dir]
#
# Produces (into dist/):
#   mortred_model_server-<version>-<profile>-linux-x64.tar.gz
#   mortred_model_server-<version>-<profile>-linux-x64.tar.gz.sha256
#
# The tarball contains the installed tree (bin/lib/conf/docs/scripts/web ui),
# the systemd unit, and install.sh (runtime apt deps + /opt/mortred layout +
# systemd wiring). Weights are NOT bundled: install.sh hints
# `python3 scripts/fetch_weights.py --profile <profile>` instead (a full
# weight set is tens of GB; shipping it in the tarball would be wasteful).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?usage: make_release_tarball.sh <cpu|gpu> <version> [build-dir]}"
VERSION="${2:?usage: make_release_tarball.sh <cpu|gpu> <version> [build-dir]}"
BUILD_DIR="${3:-$ROOT/build/full}"

[ "$PROFILE" = "cpu" ] || [ "$PROFILE" = "gpu" ] || { echo "[ERROR] profile must be cpu or gpu" >&2; exit 1; }

STAGING="$ROOT/dist/staging-$PROFILE"
OUT_TGZ="$ROOT/dist/mortred_model_server-$VERSION-$PROFILE-linux-x64.tar.gz"

command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake not found" >&2; exit 1; }

echo "== install tree -> $STAGING =="
rm -rf "$STAGING"
cmake --install "$BUILD_DIR" --prefix "$STAGING/opt/mortred"

# systemd unit + installer, both outside the /opt/mortred prefix
mkdir -p "$STAGING/deploy/nginx/snippets"
cp "$ROOT/deploy/mortred-supervisor.service" "$STAGING/deploy/"
cp "$ROOT/deploy/nginx/mortred-edge.service" "$STAGING/deploy/nginx/"
cp "$ROOT/deploy/nginx/nginx.conf.skeleton" "$STAGING/deploy/nginx/"
cp "$ROOT/deploy/nginx/snippets/"*.conf "$STAGING/deploy/nginx/snippets/"
cp "$ROOT/scripts/tarball_install.sh" "$STAGING/install.sh"
chmod +x "$STAGING/install.sh"

# profile marker: the installer + supervisor read it for catalog filtering
printf '%s\n' "$PROFILE" > "$STAGING/opt/mortred/PROFILE"

# cpu tarballs must not ship the GPU stack even when the build machine's
# 3rd_party carries it (dev boxes often have both lines installed): prune the
# CUDA/TRT shared libs and the test binaries from the staged lib/bin trees
if [ "$PROFILE" = "cpu" ]; then
    rm -f "$STAGING/opt/mortred/lib/"libMNN_Cuda* \
          "$STAGING/opt/mortred/lib/"libnvinfer* \
          "$STAGING/opt/mortred/lib/"libnvonnxparser* \
          "$STAGING/opt/mortred/lib/"libcudart* \
          "$STAGING/opt/mortred/lib/"libcudnn* 2>/dev/null || true
fi
# release tarballs carry runtime binaries only - never test executables
find "$STAGING/opt/mortred/bin" -maxdepth 1 -type f \
     \( -name '*_unittest' -o -name '*_test' \) -delete 2>/dev/null || true

mkdir -p "$ROOT/dist"
echo "== pack $OUT_TGZ =="
tar -C "$STAGING" -czf "$OUT_TGZ" .
sha256sum "$OUT_TGZ" > "$OUT_TGZ.sha256"
rm -rf "$STAGING"

echo "== done =="
ls -lh "$OUT_TGZ" "$OUT_TGZ.sha256"
