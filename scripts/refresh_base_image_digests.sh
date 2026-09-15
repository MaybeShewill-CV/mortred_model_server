#!/usr/bin/env bash
# Refresh conf/base_images.lock digests from Docker Hub (manifest list).
# Does NOT rewrite Dockerfile — review the lock diff, then update FROM pins.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
digest_for() {
  local repo="$1" tag="$2"
  local token
  token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')
  curl -fsSI \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
    "https://registry-1.docker.io/v2/${repo}/manifests/${tag}" \
    | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2; exit}'
}
{
  echo "# Locked base-image digests for Dockerfile FROM lines (linux/amd64 index)."
  echo "# Format: reference@digest"
  echo "# Refreshed: $(date -u +%Y-%m-%d) via Docker Hub registry API."
  echo "nvidia/cuda:12.6.2-devel-ubuntu22.04@$(digest_for nvidia/cuda 12.6.2-devel-ubuntu22.04)"
  echo "nvidia/cuda:12.6.2-runtime-ubuntu22.04@$(digest_for nvidia/cuda 12.6.2-runtime-ubuntu22.04)"
  echo "ubuntu:22.04@$(digest_for library/ubuntu 22.04)"
} > "$ROOT/conf/base_images.lock"
echo "wrote $ROOT/conf/base_images.lock"
cat "$ROOT/conf/base_images.lock"
