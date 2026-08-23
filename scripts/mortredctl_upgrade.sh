#!/usr/bin/env bash
# mortredctl_upgrade.sh - in-place upgrade between ADJACENT versions (invoked
# by `mortredctl upgrade`). Downloads the requested release tarball, backs up
# conf/, installs over /opt/mortred (weights untouched - they live outside the
# install tree), restarts the service, then runs the doctor.
#
#   mortredctl upgrade                    # latest release, same profile
#   mortredctl upgrade v0.2.0             # specific version
set -euo pipefail

VERSION="${1:-latest}"
PREFIX="${MORTRED_PROJECT_ROOT:-/opt/mortred}"
REPO="MaybeShewill-CV/mortred_model_server"
PROFILE="$(cat "$PREFIX/PROFILE" 2>/dev/null || echo gpu)"

[ -d "$PREFIX" ] || { echo "[ERROR] install tree not found: $PREFIX" >&2; exit 1; }

# ---- resolve version ----
if [ "$VERSION" = "latest" ]; then
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep -oP '"tag_name":\s*"\K[^"]+' | head -n1)"
    [ -n "$VERSION" ] || { echo "[ERROR] cannot resolve latest release (offline?)" >&2; exit 1; }
fi
TGZ="mortred_model_server-${VERSION#v}-$PROFILE-linux-x64.tar.gz"
URL="https://github.com/$REPO/releases/download/$VERSION/$TGZ"

echo "== Mortred upgrade =="
echo "  current: $PREFIX (profile: $PROFILE)"
echo "  target : $VERSION"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "== download $URL =="
curl -fSL "$URL" -o "$WORK/$TGZ"
if [ -f "$URL.sha256" ] || curl -fsSL "$URL.sha256" -o "$WORK/$TGZ.sha256" 2>/dev/null; then
    (cd "$WORK" && sha256sum -c "$TGZ.sha256")
else
    echo "  [WARN] no published sha256; proceeding unverified" >&2
fi
tar -xzf "$WORK/$TGZ" -C "$WORK"

echo "== backup conf + install =="
STAMP="$(date +%Y%m%d-%H%M%S)"
[ -d "$PREFIX/conf" ] && cp -a "$PREFIX/conf" "$PREFIX/conf.backup-$STAMP"
OLD_PROFILE="$(cat "$PREFIX/PROFILE" 2>/dev/null || echo "$PROFILE")"
cp -a "$WORK/opt/mortred/." "$PREFIX/"
printf '%s\n' "$OLD_PROFILE" > "$PREFIX/PROFILE"   # keep the running profile

echo "== restart + doctor =="
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet mortred-supervisor 2>/dev/null; then
    systemctl restart mortred-supervisor
fi
"$PREFIX/scripts/mortredctl_doctor.sh"
echo "== upgrade done: $VERSION (conf backup: conf.backup-$STAMP) =="
