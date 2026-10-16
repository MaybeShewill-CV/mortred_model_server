#!/usr/bin/env python3
"""Contract: every buffer.ensure failure in TrtSession::run must set
h2d_slot_guard.abandon = true before returning. Without that the
StagingSlotGuard releases a pinned slot while cudaMemcpyAsync may still
read it (UAF). Live GPU fault-injection is host-dependent; this script is
the always-on gate.

Usage:
  python3 scripts/check_trt_abandon_paths.py
  python3 scripts/check_trt_abandon_paths.py --self-test
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src/models/backend/trt_session.cpp"

ENSURE_FAIL = re.compile(
    r"const auto status = buffer\.ensure\([^)]+\);\s*"
    r"if \(status != StatusCode::OK\) \{([^}]*)\}",
    re.M,
)


def check_text(text: str) -> list[str]:
    errors = []
    blocks = list(ENSURE_FAIL.finditer(text))
    if len(blocks) < 2:
        errors.append(f"expected >=2 buffer.ensure fail blocks, found {len(blocks)}")
    for i, m in enumerate(blocks):
        body = m.group(1)
        if "h2d_slot_guard.abandon = true" not in body:
            errors.append(f"ensure-fail block #{i+1} missing abandon=true before return")
        if "return status" not in body and "return StatusCode" not in body:
            errors.append(f"ensure-fail block #{i+1} has no return")
    return errors


def self_test() -> int:
    bad = (
        "const auto status = buffer.ensure(n);\n"
        "if (status != StatusCode::OK) {\n"
        "            return status;\n"
        "}"
    )
    good = (
        "const auto status = buffer.ensure(n);\n"
        "if (status != StatusCode::OK) {\n"
        "            h2d_slot_guard.abandon = true;\n"
        "            return status;\n"
        "}"
    )
    failed = 0
    if not check_text(bad + "\n" + bad):
        print("FAIL: bad fixture must error")
        failed += 1
    else:
        print("ok: bad fixture errors")
    if check_text(good + "\n" + good):
        print("FAIL: good fixture must pass")
        failed += 1
    else:
        print("ok: good fixture passes")
    print("SELF-TEST", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    errors = check_text(SRC.read_text(encoding="utf-8"))
    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        return 1
    print("TRT buffer.ensure fail paths all set h2d_slot_guard.abandon")
    return 0


if __name__ == "__main__":
    sys.exit(main())
