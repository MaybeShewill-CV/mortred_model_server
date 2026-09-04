#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assert a GoogleTest XML report: the suite actually ran, and (by default)
no test was skipped.

GTest treats an all-skip run as exit 0. Jobs that claim to execute inference
must fail in that case. Local developers keep skip-as-pass; CI writes
--gtest_output=xml:... and runs this script.

Usage:
  python3 scripts/ci_assert_gtest_xml.py report.xml
  python3 scripts/ci_assert_gtest_xml.py report.xml --allow-skips
  python3 scripts/ci_assert_gtest_xml.py report.xml --allow-skips --inventory-out skips.json
  python3 scripts/ci_assert_gtest_xml.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def _int_attr(elem: ET.Element, name: str) -> int:
    raw = elem.get(name)
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def parse_gtest_xml(path: Path) -> tuple[int, int, int, int, list[str]]:
    """Return tests, failures, errors, skipped, and skipped testcase names."""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "testsuites":
        # Some generators emit a bare testsuite as the root.
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        tests = failures = errors = skipped = 0
        names: list[str] = []
        for suite in suites:
            tests += _int_attr(suite, "tests")
            failures += _int_attr(suite, "failures")
            errors += _int_attr(suite, "errors")
            skipped += _int_attr(suite, "skipped")
            names.extend(_skipped_names(suite))
        if skipped == 0 and names:
            skipped = len(names)
        return tests, failures, errors, skipped, names

    tests = _int_attr(root, "tests")
    failures = _int_attr(root, "failures")
    errors = _int_attr(root, "errors")
    skipped = _int_attr(root, "skipped")
    names: list[str] = []
    for suite in root.findall("testsuite"):
        names.extend(_skipped_names(suite))
        if skipped == 0:
            skipped += _int_attr(suite, "skipped")
    if skipped == 0 and names:
        skipped = len(names)
    return tests, failures, errors, skipped, names


def _skipped_names(suite: ET.Element) -> list[str]:
    suite_name = suite.get("name") or ""
    out: list[str] = []
    for case in suite.findall("testcase"):
        result = (case.get("result") or "").lower()
        status = (case.get("status") or "").lower()
        skipped_child = case.find("skipped") is not None
        if result == "skipped" or status == "notrun" or skipped_child:
            case_name = case.get("name") or ""
            out.append(f"{suite_name}.{case_name}" if suite_name else case_name)
    return out


def evaluate(path: Path, allow_skips: bool, inventory_out: Path | None = None) -> int:
    if not path.is_file():
        print(f"[FAIL] gtest xml not found: {path}", file=sys.stderr)
        return 1
    try:
        tests, failures, errors, skipped, names = parse_gtest_xml(path)
    except ET.ParseError as exc:
        print(f"[FAIL] invalid gtest xml {path}: {exc}", file=sys.stderr)
        return 1

    print(f"[gtest] tests={tests} failures={failures} errors={errors} skipped={skipped}")
    for name in names:
        print(f"  [skip] {name}")

    if inventory_out is not None:
        inventory_out.parent.mkdir(parents=True, exist_ok=True)
        inventory_out.write_text(
            json.dumps(
                {
                    "source_xml": str(path),
                    "tests": tests,
                    "failures": failures,
                    "errors": errors,
                    "skipped": skipped,
                    "skipped_names": names,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[gtest] wrote skip inventory: {inventory_out}")

    if tests <= 0:
        print("[FAIL] gtest xml reports zero tests (filter matched nothing?)", file=sys.stderr)
        return 1
    if failures > 0 or errors > 0:
        print("[FAIL] gtest xml reports failures or errors", file=sys.stderr)
        return 1
    if skipped > 0 and not allow_skips:
        print(
            f"[FAIL] {skipped} test(s) skipped; set MORTRED_CI_REQUIRE_WEIGHTS=1 "
            "and provision weights, or pass --allow-skips for an inventory run",
            file=sys.stderr,
        )
        return 1
    return 0


def _self_test() -> int:
    ok_xml = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="2" failures="0" disabled="0" errors="0" skipped="0" name="AllTests">
  <testsuite name="MnnSession" tests="1" failures="0" skipped="0" errors="0">
    <testcase name="InitAndRunMobilenetv2" status="run" result="completed" time="0.1"/>
  </testsuite>
  <testsuite name="BackendConfig" tests="1" failures="0" skipped="0" errors="0">
    <testcase name="ParseValidAndInvalidBlocks" status="run" result="completed" time="0.0"/>
  </testsuite>
</testsuites>
"""
    skip_xml = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="1" failures="0" disabled="0" errors="0" skipped="1" name="AllTests">
  <testsuite name="model_golden" tests="1" failures="0" skipped="1" errors="0">
    <testcase name="yolov8_detection" status="notrun" result="skipped" time="0">
      <skipped message="weights not available"/>
    </testcase>
  </testsuite>
</testsuites>
"""
    empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="0" failures="0" disabled="0" errors="0" skipped="0" name="AllTests">
</testsuites>
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "ok.xml").write_text(ok_xml, encoding="utf-8")
        (tmp_path / "skip.xml").write_text(skip_xml, encoding="utf-8")
        (tmp_path / "empty.xml").write_text(empty_xml, encoding="utf-8")
        if evaluate(tmp_path / "ok.xml", allow_skips=False) != 0:
            print("[FAIL] self-test: passing xml should be ok", file=sys.stderr)
            return 1
        if evaluate(tmp_path / "skip.xml", allow_skips=False) == 0:
            print("[FAIL] self-test: skipped xml must fail without --allow-skips", file=sys.stderr)
            return 1
        if evaluate(tmp_path / "skip.xml", allow_skips=True) != 0:
            print("[FAIL] self-test: skipped xml should pass with --allow-skips", file=sys.stderr)
            return 1
        inv = tmp_path / "inv.json"
        if evaluate(tmp_path / "skip.xml", allow_skips=True, inventory_out=inv) != 0:
            print("[FAIL] self-test: inventory write should still pass with --allow-skips", file=sys.stderr)
            return 1
        payload = json.loads(inv.read_text(encoding="utf-8"))
        if payload.get("skipped") != 1 or "model_golden.yolov8_detection" not in payload.get("skipped_names", []):
            print("[FAIL] self-test: inventory json missing skipped names", file=sys.stderr)
            return 1
        if evaluate(tmp_path / "empty.xml", allow_skips=True) == 0:
            print("[FAIL] self-test: zero tests must fail", file=sys.stderr)
            return 1
        missing = tmp_path / "nope.xml"
        if evaluate(missing, allow_skips=False) == 0:
            print("[FAIL] self-test: missing file must fail", file=sys.stderr)
            return 1
    print("[ok] ci_assert_gtest_xml.py --self-test")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", nargs="?", type=Path, help="gtest XML report")
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="do not fail when skipped>0 (still fail on 0 tests / failures)",
    )
    parser.add_argument(
        "--inventory-out",
        type=Path,
        default=None,
        help="write skipped test names as JSON (used by nightly --allow-skips)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if args.xml is None:
        parser.error("xml path is required unless --self-test")
    return evaluate(args.xml, args.allow_skips, args.inventory_out)


if __name__ == "__main__":
    sys.exit(main())
