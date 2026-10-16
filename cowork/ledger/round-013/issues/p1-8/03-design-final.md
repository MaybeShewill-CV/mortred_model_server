# Design P1-8 v1 (user-locked)
## Goal
Golden skip is not a silent pass locally (exit 77); presets/labels SSOT if needed.
## Fix
1. TSan presets vs labels already aligned in R12 — verify and note SSOT comment in CMakeLists near sanitizer LABELS.
2. model_golden_test already returns 77 when all selected cases skip for missing weights; ensure SKIP_RETURN_CODE 77 remains and add a one-line README/test comment that local ctest without weights must show skip not pass.
