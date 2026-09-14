# Round-3 P0 evidence — check_consistency align SME-16 TRT10 convert contract

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/r3-p0-consistency-trt10-contract` @ `02f5050`
- **Host:** Mac + WSL (`DESKTOP-B8MJFUS`)

## Problem

After SME-16 removed the `TRT_MAJOR -ge 9` fork, `check_ci_convert_trt_dry_run_contract`
still required that pattern → `check_consistency.py` exit 1 on main (CI deterministic red).

## Fix

Assert: unconditional `BUILD_FLAG="--skipInference"`; no `--buildOnly` / `-ge 9`;
require `-lt 10`; CI step must assert `TRT_VERSION_MAJOR=8` fails.

## WSL results

```text
python3 scripts/check_consistency.py
→ Repository consistency check passed.
→ exit 0

# negative: inject -ge 9 fork into convert_trt_engines.sh
→ must not keep TRT_MAJOR -ge 9 fork ...
→ EXIT:1

git checkout -- scripts/convert_trt_engines.sh
python3 scripts/check_consistency.py
→ passed
```

## Outcome

**PASS.** Round-3 P0 closed.
