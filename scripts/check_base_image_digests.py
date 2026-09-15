#!/usr/bin/env python3
"""Fail-closed: every external Dockerfile FROM must be tag@sha256, matching conf/base_images.lock."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
LOCK = ROOT / "conf" / "base_images.lock"

FROM_RE = re.compile(
    r"^FROM\s+(?P<ref>\S+)\s+AS\s+(?P<stage>\S+)\s*$",
    re.MULTILINE,
)
DIGEST_RE = re.compile(r"^([^@\s]+)@(?P<digest>sha256:[0-9a-f]{64})$")


def main() -> int:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lock_lines = [
        ln.strip()
        for ln in LOCK.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    lock_map: dict[str, str] = {}
    for ln in lock_lines:
        m = DIGEST_RE.match(ln)
        if not m:
            print(f"[FAIL] bad lock line: {ln}", file=sys.stderr)
            return 1
        lock_map[m.group(1)] = m.group("digest")

    errors: list[str] = []
    seen_external = 0
    for m in FROM_RE.finditer(text):
        ref = m.group("ref")
        stage = m.group("stage")
        # Stage references (FROM runtime AS mortred-gpu) have no registry slash-or-tag digest form
        if "/" not in ref and ":" not in ref and "@" not in ref:
            # plain stage name
            continue
        if ref.startswith("scratch"):
            continue
        dm = DIGEST_RE.match(ref)
        if not dm:
            errors.append(f"FROM {ref} AS {stage}: missing @sha256 digest pin")
            continue
        name = dm.group(1)
        digest = dm.group("digest")
        seen_external += 1
        expected = lock_map.get(name)
        if expected is None:
            errors.append(f"FROM {ref}: not listed in conf/base_images.lock")
        elif expected != digest:
            errors.append(f"FROM {ref}: digest != lock ({expected})")

    if seen_external < 3:
        errors.append(f"expected >=3 external FROM pins, found {seen_external}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1
    print(f"[ok] Dockerfile digests match conf/base_images.lock ({seen_external} external FROM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
