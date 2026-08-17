#!/usr/bin/env bash
#
# check_repo_clean.sh - repository hygiene gate.
#
# Ensures that paths which should be generated at build/runtime time are not
# present in a clean source tree. It is intended for CI and pre-commit use.
#
# Usage:
#   ./scripts/check_repo_clean.sh
#
# Exit code 0: clean.
# Exit code 1: one or more generated/ignored paths are present.
#
# You can use CHECK_ALLOW_GENERATED=1 to allow the check to pass when those
# paths exist but are ignored by .gitignore (useful in local working trees
# that contain build outputs but do not track them).

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Paths that must never be committed to a clean source tree.
FORBIDDEN_PATHS=(
  "_bin"
  "_lib"
  "build"
  "build-ci"
  "build-werror"
  "build-gate"
  "build-tidy"
  "cmake-build-debug"
  "cmake-build-release"
  "logs"
  "generated_configs"
)

FAILED=0

for p in "${FORBIDDEN_PATHS[@]}"; do
  if [ -e "$p" ]; then
    if [ "${CHECK_ALLOW_GENERATED:-0}" = "1" ]; then
      echo "[WARN] generated path present (allowed by CHECK_ALLOW_GENERATED): $p"
    else
      echo "[ERROR] generated path present in source tree: $p"
      FAILED=1
    fi
  fi
done

# Check that no stale executables with missing source are present in _bin.
# This check only runs if _bin exists and the user explicitly asks for it,
# because _bin is ignored in normal source-control workflows.
if [ "${CHECK_STALE_BINARIES:-0}" = "1" ] && [ -d "_bin" ]; then
  STALE_BINARIES=(
    "llama3_chatbot_server.out"
    "qwen2_vl_chatbot_server.out"
    "ollama_to_llama_cpp_proxy_server.out"
    "jina_embedding_v3_benchmark.out"
    "build_wiki_corpus_index.out"
    "search_wiki_corpus.out"
    "tokenizer_benchmark.out"
    "llm_request_parser_unittest"
    "llm_datatype_unittest"
  )
  for b in "${STALE_BINARIES[@]}"; do
    if [ -e "_bin/$b" ]; then
      echo "[ERROR] stale binary with no source present: _bin/$b"
      FAILED=1
    fi
  done
fi

if [ "$FAILED" -ne 0 ]; then
  echo "Repository hygiene check failed."
  exit 1
fi

echo "Repository hygiene check passed."
exit 0
