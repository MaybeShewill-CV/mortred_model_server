#!/usr/bin/env python3
"""Print an explicit T0/T1/T2 verification-tier summary for CI logs.

D3 / P1-2: a green wrapper must never be readable as "full zoo passed" when
T2 (gpu-nightly-full) did not run. This script only prints; it does not
decide job success (callers still exit non-zero on real failures).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID = {"ran", "not_run", "skipped", "failed", "unknown"}


def normalize(value: str) -> str:
    v = value.strip().lower().replace("-", "_")
    aliases = {
        "success": "ran",
        "pass": "ran",
        "passed": "ran",
        "ok": "ran",
        "not_run": "not_run",
        "notrun": "not_run",
        "did_not_run": "not_run",
        "skip": "skipped",
        "failure": "failed",
        "cancelled": "failed",
    }
    v = aliases.get(v, v)
    if v not in VALID:
        raise SystemExit(f"invalid tier status {value!r}; expected one of {sorted(VALID)}")
    return v


def label(status: str) -> str:
    return {
        "ran": "RAN",
        "not_run": "NOT RUN",
        "skipped": "SKIPPED",
        "failed": "FAILED",
        "unknown": "UNKNOWN",
    }[status]


def load_skip_inventory(path: Path | None) -> tuple[int, list[str]]:
    if path is None:
        return 0, []
    if not path.is_file():
        return 0, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = list(payload.get("skipped_names") or [])
    skipped = int(payload.get("skipped") or len(names))
    return skipped, names


def render(
    t0: str,
    t1: str,
    t2: str,
    gpu_smoke: str | None = None,
    t2_skip_inventory: Path | None = None,
) -> str:
    lines = [
        "======== verification tier summary (T0 / T1 / T2) ========",
        f"T0 (weight-free contracts / tests-only): {label(t0)}",
        f"T1 (hosted sha256-locked goldens):        {label(t1)}",
        f"T2 (gpu-nightly-full / full zoo):         {label(t2)}",
    ]
    if gpu_smoke is not None:
        lines.append(f"gpu-smoke (PR gate, not full T2):          {label(gpu_smoke)}")
    if t2 == "not_run":
        lines.append(
            "NOTE: T2 did NOT run this workflow. A green check here is NOT a full-zoo claim."
        )
    elif t2 == "skipped":
        lines.append(
            "NOTE: T2 job was skipped (no GPU runner / gate). Not a full-zoo claim."
        )
    elif t2 == "ran":
        skipped, names = load_skip_inventory(t2_skip_inventory)
        if skipped > 0:
            lines.append(f"T2 rest-of-zoo skips (allow-skips inventory): {skipped}")
            for name in names[:30]:
                lines.append(f"  - {name}")
            if len(names) > 30:
                lines.append(f"  ... {len(names) - 30} more")
        else:
            lines.append("T2 rest-of-zoo skips: 0 (or no inventory provided)")
    lines.append("========================================================")
    return "\n".join(lines) + "\n"


def _self_test() -> int:
    text = render("ran", "ran", "not_run", gpu_smoke="skipped")
    if "T2 (gpu-nightly-full / full zoo):         NOT RUN" not in text:
        print("[FAIL] expected T2 NOT RUN line", file=sys.stderr)
        return 1
    if "NOT a full-zoo claim" not in text:
        print("[FAIL] expected full-zoo disclaimer", file=sys.stderr)
        return 1
    if "gpu-smoke" not in text or "SKIPPED" not in text:
        print("[FAIL] expected gpu-smoke SKIPPED", file=sys.stderr)
        return 1
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        inv = Path(tmp) / "skips.json"
        inv.write_text(
            json.dumps({"skipped": 1, "skipped_names": ["model_golden.foo"]}) + "\n",
            encoding="utf-8",
        )
        text2 = render("ran", "ran", "ran", t2_skip_inventory=inv)
        if "T2 rest-of-zoo skips" not in text2 or "model_golden.foo" not in text2:
            print("[FAIL] expected skip inventory in T2 RAN summary", file=sys.stderr)
            return 1
    print("[ok] ci_print_tier_summary.py --self-test")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0", default="unknown", help="T0 status")
    parser.add_argument("--t1", default="unknown", help="T1 status")
    parser.add_argument("--t2", default="unknown", help="T2 status")
    parser.add_argument("--gpu-smoke", default=None, help="optional PR gpu-smoke status")
    parser.add_argument(
        "--t2-skip-inventory",
        type=Path,
        default=None,
        help="optional gpu-rest-skips.json when T2 ran with --allow-skips",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    print(
        render(
            normalize(args.t0),
            normalize(args.t1),
            normalize(args.t2),
            None if args.gpu_smoke is None else normalize(args.gpu_smoke),
            args.t2_skip_inventory,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
