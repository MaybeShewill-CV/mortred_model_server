#!/usr/bin/env python3
"""Golden zero-drift guard.

A green test suite only proves the model still works; identical sha256 hashes
prove the numbers did not change by a single bit. This script records the
golden case names and the hash of every baseline file under test/golden/, and
verifies them later, so a migration can prove it changed nothing.

    python scripts/golden_drift_check.py --record   # write the baseline
    python scripts/golden_drift_check.py --check    # verify against it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test"
DEFAULT_BASELINE = ROOT / "test" / "golden_baseline.json"


def collect() -> dict:
    source = (TEST / "model_golden_test.cc").read_text(encoding="utf-8")
    names = re.findall(r"(?:TEST\(model_golden,|GOLDEN_\w+_CASE\()\s*(\w+)", source)
    golden = {}
    for path in sorted((TEST / "golden").iterdir()):
        if path.is_file():
            golden[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "case_count": len(names),
        "case_names": names,
        "golden_count": len(golden),
        "golden_sha256": golden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="write the baseline file")
    mode.add_argument("--check", action="store_true", help="verify against the baseline file")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="baseline path")
    args = parser.parse_args()

    current = collect()

    if args.record:
        args.baseline.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {current['case_count']} cases and {current['golden_count']} golden files")
        print(f"baseline: {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"error: baseline not found: {args.baseline}", file=sys.stderr)
        return 2
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))

    errors: list[str] = []
    if current["case_names"] != expected["case_names"]:
        missing = [n for n in expected["case_names"] if n not in current["case_names"]]
        added = [n for n in current["case_names"] if n not in expected["case_names"]]
        if missing:
            errors.append(f"cases removed: {missing}")
        if added:
            errors.append(f"cases added: {added}")
        if not missing and not added:
            errors.append("case declaration order changed")

    old_hashes, new_hashes = expected["golden_sha256"], current["golden_sha256"]
    for name in old_hashes:
        if name not in new_hashes:
            errors.append(f"golden file removed: {name}")
        elif old_hashes[name] != new_hashes[name]:
            errors.append(f"golden file modified: {name}")
    for name in new_hashes:
        if name not in old_hashes:
            errors.append(f"golden file added: {name}")

    if errors:
        print("golden drift detected:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"zero drift: {current['case_count']} cases and {current['golden_count']} golden files unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
