#!/usr/bin/env python3
"""Summarise a ctest JUnit report and make skipped tests impossible to miss.

Why this exists: `SKIP_RETURN_CODE 77` makes "no weights here" indistinguishable
from "the check actually ran" in a plain ctest summary. ctest exits 0 when tests
are skipped, so a run that proved nothing looks exactly like a run that proved
everything. This script prints an explicit passed/failed/skipped line and, unless
--allow-skips is given, fails the job when anything was skipped.

Usage:
  python3 scripts/ctest_summary.py --junit build-ci/ctest-junit.xml [--allow-skips]
  python3 scripts/ctest_summary.py --self-test

Exit codes:
  0  no failures (and no skips, or --allow-skips)
  1  at least one failed/errored test
  2  usage / parse error
  3  skipped tests and --allow-skips was not given
"""
import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize(junit_path: Path) -> dict:
    """Return {total, passed, failed, skipped} for a ctest JUnit report."""
    root = ET.parse(str(junit_path)).getroot()
    counts = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    cases = root.iter("testcase")
    for case in cases:
        counts["total"] += 1
        status = (case.get("status") or "").lower()
        has_skip = case.find("skipped") is not None
        has_fail = case.find("failure") is not None or case.find("error") is not None
        if has_skip or status == "disabled":
            counts["skipped"] += 1
        elif has_fail or status in ("fail", "failed"):
            counts["failed"] += 1
        else:
            counts["passed"] += 1
    return counts


def report(counts: dict, allow_skips: bool) -> int:
    print(
        "CTEST SUMMARY total=%d passed=%d failed=%d skipped=%d allow_skips=%s"
        % (counts["total"], counts["passed"], counts["failed"], counts["skipped"],
           "yes" if allow_skips else "no")
    )
    if counts["failed"] > 0:
        print("ctest_summary: %d test(s) failed" % counts["failed"], file=sys.stderr)
        return 1
    if counts["skipped"] > 0 and not allow_skips:
        print(
            "ctest_summary: %d test(s) were SKIPPED and this job claims to prove "
            "them; pass --allow-skips only where skips are expected (e.g. no "
            "weights installed), or install the missing weights." % counts["skipped"],
            file=sys.stderr,
        )
        return 3
    return 0


def _write(path: Path, body: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuites><testsuite name="t" tests="0">' + body + "</testsuite></testsuites>\n",
        encoding="utf-8",
    )


def self_test() -> int:
    fails = 0
    cases = [
        ("all pass", '<testcase name="a"/><testcase name="b" status="run"/>',
         {"total": 2, "passed": 2, "failed": 0, "skipped": 0}, 0),
        ("one failure", '<testcase name="a"><failure message="x"/></testcase>',
         {"total": 1, "passed": 0, "failed": 1, "skipped": 0}, 1),
        ("one skip", '<testcase name="a" status="disabled"><skipped message="weights"/></testcase>',
         {"total": 1, "passed": 0, "failed": 0, "skipped": 1}, 3),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for name, body, want, want_rc in cases:
            p = Path(tmp) / (name.replace(" ", "_") + ".xml")
            _write(p, body)
            got = summarize(p)
            rc = report(got, allow_skips=False)
            if got != want:
                print("FAIL: %s counts %s != %s" % (name, got, want))
                fails += 1
            if rc != want_rc:
                print("FAIL: %s exit %d != %d" % (name, rc, want_rc))
                fails += 1
        # the same skip must pass when skips are explicitly allowed
        p = Path(tmp) / "skip_allowed.xml"
        _write(p, '<testcase name="a" status="disabled"><skipped/></testcase>')
        if report(summarize(p), allow_skips=True) != 0:
            print("FAIL: skip not allowed with --allow-skips")
            fails += 1
        # a missing report is a usage error, not a silent pass
        try:
            summarize(Path(tmp) / "does-not-exist.xml")
            print("FAIL: missing junit accepted")
            fails += 1
        except (OSError, ET.ParseError):
            pass
    print("SUMMARY FAILS=%d" % fails)
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--junit", default="", help="path to the ctest JUnit report")
    parser.add_argument("--allow-skips", action="store_true",
                        help="report skips but do not fail on them")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.junit:
        parser.error("--junit is required (or use --self-test)")
    path = Path(args.junit)
    if not path.is_file():
        print("ctest_summary: no JUnit report at %s "
              "(build the check target with CMake >= 3.21 so it emits one)" % path,
              file=sys.stderr)
        return 2
    try:
        counts = summarize(path)
    except ET.ParseError as exc:
        print("ctest_summary: cannot parse %s: %s" % (path, exc), file=sys.stderr)
        return 2
    return report(counts, args.allow_skips)


if __name__ == "__main__":
    sys.exit(main())
