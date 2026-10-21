# PF5 retest (coverage set)

PF5 compares a later run on the **same hardware** and the **same command** with the recorded baseline:

- success RPS must be at least **85%** of the recorded value (`RPS_FLOOR = 0.85`)
- p99 must be at most **150%** of the recorded value (`P99_CEIL = 1.50`)

A miss fails the process (exit 1) unless a changelog entry records an intentional regression. A hardware mismatch fails and does **not** rewrite the baseline. `--ids all` is a stub: ids without a PF1 report are not given invented RPS numbers (that gap is P1-FC-1).

The coverage-set nine baselines in `docs/perf/baselines/<id>.json` were imported from `docs/perf/coverage_set_http_rps.md` (2026-10-14 / 2026-10-15, NVIDIA GeForce RTX 2070 SUPER, 8192 MiB). This tree does not add a second live GPU measurement.

## Check the recorded files

```bash
python3 scripts/pf5_retest.py --check-baselines
python3 scripts/pf5_retest.py --ids coverage_set --dry-compare
python3 scripts/pf5_retest.py --self-test
```

`--dry-compare` checks each baseline against itself (ratio 1.0). It does not talk to a GPU.

## Retest one id or the coverage set

Run `scripts/server/http_infer_rps.py --raw --out` with the same `--model`, `-c` (`worker_nums` × 2), `-d`, `--warmup`, and `--image` as that id's `load_command`, on the same hardware string. Then:

```bash
python3 scripts/pf5_retest.py --ids coverage_set \
  --current-dir /path/to/retest-json \
  --hardware "NVIDIA GeForce RTX 2070 SUPER, 8192 MiB"
```

```bash
python3 scripts/pf5_retest.py --ids yolov8s \
  --current /path/to/load.json \
  --hardware "NVIDIA GeForce RTX 2070 SUPER, 8192 MiB"
```

`http_infer_rps.py --out` JSON is read as `rps` and `latency_ms.p99`. A document that already has `success_rps`, `p99_ms`, and `hardware` is accepted too. Exit 0 passes. Exit 1 is a threshold miss, a hardware mismatch, or a command mismatch (`worker_nums`, concurrency, `--raw` / `-c` / `-d` / `--warmup` / `--image`).

## Intentional regression

Add a non-empty reason to `docs/perf/baselines/regressions.json`:

```json
{
  "schema": "pf5-regression-v1",
  "entries": [
    {"id": "yolov8s", "date": "2026-10-21", "reason": "why this RPS or p99 change is accepted"}
  ]
}
```

or pass `--changelog other.json` or `--reason-file notes.txt`. The reason excuses an RPS or p99 miss only. It does not excuse a different GPU or a different load shape.

To replace the recorded numbers after that acceptance (same hardware, same command):

```bash
python3 scripts/pf5_retest.py --adopt-baseline --ids yolov8s \
  --current /path/to/load.json \
  --hardware "NVIDIA GeForce RTX 2070 SUPER, 8192 MiB" \
  --changelog docs/perf/baselines/regressions.json
```

## Where a new baseline comes from

`coverage_http_report.py` creates `docs/perf/baselines/<id>.json` the first time it writes a passing row. On a later run it keeps the recorded success RPS and p99 when the hardware string matches, and it refuses to write when the hardware string differs.

To fill JSON from a markdown report that already has `success_rps` and `p99_ms` (no new numbers):

```bash
python3 scripts/pf5_retest.py --import-markdown
```

The importer writes missing files and refuses to overwrite a baseline whose numbers differ. Do not type RPS values in by hand.
