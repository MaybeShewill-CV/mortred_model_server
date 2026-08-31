#!/usr/bin/env python3
"""Contract sync gate: C++ catalogs <-> docs/contract_dump.json <-> OpenAPI.

Chain (each arrow must be reproducible):
  factory catalogs -> contract_dump -> docs/contract_dump.json -> gen_openapi.py

Fails when:
  - the freshly dumped contract differs from the committed dump (a ParamSpec
    or catalog entry changed without regenerating), or
  - docs/openapi.json / src/server/openapi_doc.h differ from regeneration.

Usage:
  python scripts/check_contract_sync.py --dump-bin <build>/bin/contract_dump
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP_PATH = ROOT / "docs" / "contract_dump.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-bin", required=True, help="path to the built contract_dump binary")
    args = parser.parse_args()

    dump_bin = Path(args.dump_bin)
    if not dump_bin.is_file():
        print("ERROR: contract_dump binary not found: %s" % dump_bin)
        print("Build it first: cmake --build <full-build-dir> --target contract_dump")
        return 2

    problems: list[str] = []

    if not DUMP_PATH.exists():
        problems.append("docs/contract_dump.json is missing")
    else:
        fresh = subprocess.run(
            [str(dump_bin)], capture_output=True, text=True, check=False, cwd=str(ROOT)
        )
        if fresh.returncode != 0:
            problems.append("contract_dump exited %d: %s" % (fresh.returncode, fresh.stderr.strip()))
        else:
            # utf-8-sig tolerates a BOM introduced by Windows editors
            committed = json.loads(DUMP_PATH.read_text(encoding="utf-8-sig"))
            try:
                regenerated = json.loads(fresh.stdout)
            except json.JSONDecodeError as error:
                problems.append("contract_dump produced invalid JSON: %s" % error)
                regenerated = None
            if regenerated is not None and regenerated != committed:
                problems.append(
                    "docs/contract_dump.json is out of date with the C++ catalogs "
                    "(regenerate: %s > docs/contract_dump.json)" % dump_bin
                )

    gen = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_openapi.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    if gen.returncode != 0:
        problems.append(gen.stdout.strip() or gen.stderr.strip() or "gen_openapi.py --check failed")

    if problems:
        for problem in problems:
            print("ERROR: %s" % problem)
        return 1
    print("Contract chain is in sync (C++ catalogs -> dump -> OpenAPI).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
