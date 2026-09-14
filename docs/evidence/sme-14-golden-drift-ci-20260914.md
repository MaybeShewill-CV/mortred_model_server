# SME-14 evidence — golden drift check in CI + baseline reset

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-14-golden-drift-ci` @ `2763686`
- **Host:** Mac + WSL review (`DESKTOP-B8MJFUS`)

## Review before --record

Prior `--check` failure (27/25 vs 30/26) covered intentional case/file adds
(rename dinov2, yolov8_onnx, mixed_batch, ddpm) and
`enlightengan_enhancement.png` refresh (`16ef6dd`). Baseline then recorded.

## Commands / results

```text
python3 scripts/golden_drift_check.py --check
→ zero drift: 30 cases and 26 golden files unchanged
→ exit 0

python3 scripts/check_consistency.py
→ Repository consistency check passed.
→ exit 0

# negative: append byte to enlightengan png
→ golden file modified: enlightengan_enhancement.png
→ exit 1; restored
```

## Outcome

**PASS.** `check_consistency` rule 22 + CI `py_compile` list
`golden_drift_check.py`; refresh docs require `--record` in the same PR.
