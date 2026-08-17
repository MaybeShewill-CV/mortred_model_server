#!/usr/bin/env bash
#
# rename_configs_to_toml.sh - migrate all configuration files from .toml to .toml.
#
# This script:
#   1. Renames every *.toml under conf/ to *.toml using git mv when possible.
#   2. Updates source/docs/script references from .toml to .toml.
#
# Usage:
#   ./scripts/rename_configs_to_toml.sh
#
# After running, verify with:
#   find conf -name '*.toml' -print
#   grep -R "\.toml" -n src docs scripts README.md README.zh-cn.md || true

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Rename config files.
while IFS= read -r -d '' f; do
  new="${f%.toml}.toml"
  if [ -e "$new" ]; then
    echo "SKIP: $new already exists"
    continue
  fi
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git mv "$f" "$new"
  else
    mv "$f" "$new"
  fi
  echo "RENAMED: $f -> $new"
done < <(find conf -name '*.toml' -print0)

# 2. Update references in tracked text files.
#    Only replace the .toml extension token, not arbitrary text.
grep -RIl '\.toml' src docs scripts README.md README.zh-cn.md 2>/dev/null | while IFS= read -r file; do
  sed -i 's/\.toml/.toml/g' "$file"
  echo "UPDATED: $file"
done

echo "Configuration migration to .toml completed."
