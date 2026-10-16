#!/usr/bin/env python3
"""SSOT gate: the TSan target list must be identical in all three places that
spell it out.

Sources (all three must expose the same set of test targets):
  1. test/CMakeLists.txt      - the if() condition guarding
                                `set_tests_properties(... LABELS "sanitizer")`
                                (what `ctest -L sanitizer` selects)
  2. CMakePresets.json        - buildPreset "tests-only-tsan" targets
                                (what a local `--preset tests-only-tsan` builds)
  3. .github/workflows/ci.yml - the sanitizers job's
                                `cmake --build build-tsan --target ...` list
                                (what CI actually builds before ctest -L)

A mismatch means one of them drifts: either ctest selects an unbuilt target
(structural red, see the round-12 P1-2 history) or CI builds a target that the
label never runs. Exit 0 iff all three sets are equal and non-empty.

Usage:
  python3 scripts/check_tsan_targets.py            # check the repo
  python3 scripts/check_tsan_targets.py --self-test
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CMAKE_TEST = ROOT / "test" / "CMakeLists.txt"
PRESETS = ROOT / "CMakePresets.json"
CI = ROOT / ".github" / "workflows" / "ci.yml"

LABEL_LINE = 'set_tests_properties(${src} PROPERTIES LABELS "sanitizer")'
PRESET_NAME = "tests-only-tsan"


def label_targets_from_cmake(text: str) -> set[str]:
    """Names inside the if(...) guarding the sanitizer LABELS assignment."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != LABEL_LINE:
            continue
        base_indent = len(line) - len(line.lstrip())
        start = idx - 1
        while start >= 0:
            stripped = lines[start].lstrip()
            indent = len(lines[start]) - len(stripped)
            if stripped.startswith("if(") and indent <= base_indent:
                break
            start -= 1
        else:
            raise ValueError("LABELS line has no opening if() above it")
        block = " ".join(lines[start:idx])
        return set(re.findall(r'STREQUAL\s+"([A-Za-z0-9_]+)"', block))
    raise ValueError(f"no {LABEL_LINE!r} line found in test/CMakeLists.txt")


def targets_from_presets(text: str) -> set[str]:
    data = json.loads(text)
    for preset in data.get("buildPresets", []):
        if preset.get("name") == PRESET_NAME:
            return set(preset.get("targets", []))
    raise ValueError(f"buildPreset {PRESET_NAME!r} not found in CMakePresets.json")


def targets_from_ci(text: str) -> set[str]:
    """Targets on the `cmake --build build-tsan --target ...` command line."""
    for line in text.splitlines():
        stripped = line.strip()
        if "cmake --build build-tsan --target" not in stripped:
            continue
        tail = stripped.split("--target", 1)[1]
        tail = tail.split("-j", 1)[0]
        return set(tok for tok in tail.split() if tok)
    raise ValueError("no `cmake --build build-tsan --target` line in ci.yml")


def run_check(cmake_text: str, presets_text: str, ci_text: str) -> list[str]:
    errors: list[str] = []
    sources = {
        "test/CMakeLists.txt labels": label_targets_from_cmake(cmake_text),
        f"CMakePresets.json {PRESET_NAME}": targets_from_presets(presets_text),
        "ci.yml tsan build list": targets_from_ci(ci_text),
    }
    if not all(sources.values()):
        errors.append(f"a source parsed empty: { {k: sorted(v) for k, v in sources.items()} }")
    union: set[str] = set()
    for name, targets in sources.items():
        union |= targets
        print(f"{name}: {len(targets)} targets -> {sorted(targets)}")
    for name, targets in sources.items():
        for missing in sorted(union - targets):
            errors.append(f"{name} is missing {missing}")
    return errors


def self_test() -> int:
    cmake_ok = (
        "foreach(src ${TEST_LIST})\n"
        "    if((${src} STREQUAL \"async_job_unittest\") OR (${src} STREQUAL \"worker_pool_unittest\"))\n"
        "        set_tests_properties(${src} PROPERTIES LABELS \"sanitizer\")\n"
        "    endif()\n"
        "endforeach()\n"
    )
    presets_ok = json.dumps(
        {"buildPresets": [{"name": PRESET_NAME, "targets": ["async_job_unittest", "worker_pool_unittest"]}]}
    )
    ci_ok = 'cmake --build build-tsan --target async_job_unittest worker_pool_unittest -j"$(nproc)"\n'
    failed = 0
    if run_check(cmake_ok, presets_ok, ci_ok):
        print("FAIL: identical sources must pass")
        failed += 1
    else:
        print("ok: identical sources pass")
    drifted_presets = json.dumps(
        {"buildPresets": [{"name": PRESET_NAME, "targets": ["async_job_unittest"]}]}
    )
    errors = run_check(cmake_ok, drifted_presets, ci_ok)
    if not any("missing worker_pool_unittest" in e for e in errors):
        print("FAIL: drifted presets must be reported:", errors)
        failed += 1
    else:
        print("ok: drift reported")
    print("SELF-TEST", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    errors = run_check(CMAKE_TEST.read_text(encoding="utf-8"),
                       PRESETS.read_text(encoding="utf-8"),
                       CI.read_text(encoding="utf-8"))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("TSan target lists drifted - keep all three sources in sync", file=sys.stderr)
        return 1
    print("TSan target lists agree across labels / preset / CI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
