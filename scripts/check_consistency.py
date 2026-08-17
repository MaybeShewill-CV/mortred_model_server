#!/usr/bin/env python3
"""Repository consistency checker.

Verifies a few high-signal invariants:

1. Every source path referenced in docs/repository-layout.md exists.
2. Every server executable in docs/repository-layout.md has a matching source file.
3. If _bin exists, stale binaries listed in the layout policy are reported.
4. Every conf/server/*.toml file has at least one matching server source directory.

Exit code 0 means consistent; non-zero means the repository needs attention.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STALE_BINARIES = [
    "llama3_chatbot_server.out",
    "qwen2_vl_chatbot_server.out",
    "ollama_to_llama_cpp_proxy_server.out",
    "jina_embedding_v3_benchmark.out",
    "build_wiki_corpus_index.out",
    "search_wiki_corpus.out",
    "tokenizer_benchmark.out",
    "llm_request_parser_unittest",
    "llm_datatype_unittest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-stale-binaries",
        action="store_true",
        help="Also check _bin for stale executables without source.",
    )
    return parser.parse_args()


def check_layout_references() -> list[str]:
    errors: list[str] = []
    layout = ROOT / "docs" / "repository-layout.md"
    if not layout.exists():
        errors.append("docs/repository-layout.md is missing")
        return errors

    for line in layout.read_text(encoding="utf-8").splitlines():
        # Match inline code paths like `src/...` in the markdown tables/text.
        for match in re.finditer(r"`((?:src|conf|scripts|test|docs)/[^`]+)`", line):
            ref = match.group(1)
            # Strip trailing punctuation that may have been included in inline code.
            ref = ref.rstrip(".,;")
            # Ignore glob-like or placeholder references.
            if any(ch in ref for ch in "*?<>"):
                continue
            if not (ROOT / ref).exists():
                errors.append(f"docs/repository-layout.md references missing path: {ref}")
    return errors


def check_server_source_mapping() -> list[str]:
    errors: list[str] = []
    server_dir = ROOT / "src" / "apps" / "server"
    if not server_dir.exists():
        errors.append("src/apps/server is missing")
        return errors

    # Every top-level task directory should have at least one .cpp file.
    for child in sorted(server_dir.iterdir()):
        if child.is_dir() and not list(child.glob("*.cpp")):
            errors.append(f"server task directory has no .cpp files: {child.relative_to(ROOT)}")

    # Every conf/server subdirectory should have a corresponding server source directory.
    conf_server = ROOT / "conf" / "server"
    if conf_server.exists():
        for child in sorted(conf_server.iterdir()):
            if child.is_dir():
                # conf/server/scene_segmentation -> src/apps/server/scene_segmentation
                src_dir = server_dir / child.name
                if not src_dir.exists() and not list(child.glob("*.toml")) and not list(child.glob("*.toml")):
                    # Some conf dirs contain nested model dirs; only report if no ini at all.
                    errors.append(
                        f"conf/server/{child.name} has no matching src/apps/server/{child.name}"
                    )
    return errors


def check_stale_binaries() -> list[str]:
    errors: list[str] = []
    bin_dir = ROOT / "_bin"
    if not bin_dir.is_dir():
        return errors
    for name in STALE_BINARIES:
        if (bin_dir / name).exists():
            errors.append(f"stale binary without source: _bin/{name}")
    return errors


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    errors.extend(check_layout_references())
    errors.extend(check_server_source_mapping())
    if args.check_stale_binaries:
        errors.extend(check_stale_binaries())

    if errors:
        print("Repository consistency check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Repository consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
