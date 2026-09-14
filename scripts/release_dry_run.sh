#!/usr/bin/env bash
# release_dry_run.sh - local dry-run of release contracts (no GHCR push, no GitHub Release).
#
# Covers SME-04 / P0-3:
#   1) release.yml lowercases github.repository for GHCR IMAGE refs
#   2) tarball + basename .sha256 verify with sha256sum -c
#   3) bootstrap-style mismatch refuses install; missing .sha256 only WARNs
#
# Usage (repo root):
#   bash scripts/release_dry_run.sh
# Exit 0 on all checks pass.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() { echo "[FAIL] $*" >&2; exit 1; }
ok() { echo "[ok] $*"; }

echo "== 1) release.yml GHCR IMAGE lowercase =="
REL="$ROOT/.github/workflows/release.yml"
[ -f "$REL" ] || fail "missing $REL"
grep -Fq "tr '[:upper:]' '[:lower:]'" "$REL" \
  || fail "release.yml must lowercase github.repository for GHCR"
grep -qE "invalid image ref" "$REL" \
  || fail "release.yml must fail-closed on invalid IMAGE ref"
# Simulate mixed-case owner → lowercase image path (same formula as the workflow).
REGISTRY="ghcr.io"
REPO_MIXED="MaybeShewill-CV/mortred_model_server"
IMAGE="$REGISTRY/$(echo "$REPO_MIXED" | tr '[:upper:]' '[:lower:]')"
echo "$IMAGE" | grep -qE '^[a-z0-9._/-]+$' || fail "simulated IMAGE not lowercase-safe: $IMAGE"
case "$IMAGE" in
  *MaybeShewill*|*CV/*) fail "IMAGE still contains uppercase: $IMAGE" ;;
esac
[ "$IMAGE" = "ghcr.io/maybeshewill-cv/mortred_model_server" ] \
  || fail "unexpected IMAGE: $IMAGE"
ok "IMAGE=$IMAGE"

echo "== 2) tarball + basename .sha256 =="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TGZ="mortred_model_server-0.0.0-dryrun-cpu-linux-x64.tar.gz"
mkdir -p "$WORK/stage/opt/mortred"
printf 'dry-run\n' > "$WORK/stage/opt/mortred/PROFILE"
printf '#!/bin/sh\necho install-ok\n' > "$WORK/stage/install.sh"
chmod +x "$WORK/stage/install.sh"
tar -C "$WORK/stage" -czf "$WORK/$TGZ" .
(cd "$WORK" && sha256sum "$TGZ" > "$TGZ.sha256" && sha256sum -c "$TGZ.sha256") \
  || fail "sha256sum -c failed for freshly packed tarball"
# Basename-only checksum file (no path prefix) — required for curl -fLO + verify.
grep -qE "^[0-9a-f]{64}  ${TGZ}\$" "$WORK/$TGZ.sha256" \
  || fail ".sha256 must list basename only, got: $(cat "$WORK/$TGZ.sha256")"
ok "tarball+sha256 basename OK"

echo "== 3) bootstrap checksum: mismatch hard-fail =="
# Mirror scripts/bootstrap.sh track-2 logic without network.
cp "$WORK/$TGZ" "$WORK/bad.tgz"
cp "$WORK/$TGZ.sha256" "$WORK/bad.tgz.sha256"
# Corrupt the recorded digest while keeping the file present.
python3 - <<PY
from pathlib import Path
p = Path("$WORK/bad.tgz.sha256")
line = p.read_text().strip()
digest, name = line.split(None, 1)
bad = ("0" if digest[0] != "0" else "1") + digest[1:]
p.write_text(f"{bad}  {name}\\n")
print("corrupted_sha256")
PY
set +e
out="$(cd "$WORK" && sha256sum -c bad.tgz.sha256 2>&1)"
rc=$?
set -e
[ "$rc" -ne 0 ] || fail "expected sha256 mismatch to fail, got rc=0"
# bootstrap maps this to ERROR + exit 1
ok "mismatch refuses (sha256sum -c rc=$rc)"

echo "== 4) bootstrap checksum: missing .sha256 WARN path =="
# Present tarball, absent .sha256 → bootstrap WARNs and continues (we only assert the branch exists in source).
grep -q '\[WARN\] no published sha256' "$ROOT/scripts/bootstrap.sh" \
  || fail "bootstrap.sh missing WARN-for-absent-.sha256 branch"
grep -q '\[ERROR\] sha256 mismatch' "$ROOT/scripts/bootstrap.sh" \
  || fail "bootstrap.sh missing ERROR-for-mismatch branch"
grep -q 'refusing to install' "$ROOT/scripts/bootstrap.sh" \
  || fail "bootstrap.sh must refuse on mismatch"
ok "bootstrap.sh WARN/ERROR branches present"

echo "== 5) release.yml packs .sha256 beside tarball =="
grep -q 'sha256sum' "$REL" || fail "release.yml must run sha256sum on tarballs"
ok "release.yml sha256sum present"

echo
echo "release_dry_run: ALL OK"
