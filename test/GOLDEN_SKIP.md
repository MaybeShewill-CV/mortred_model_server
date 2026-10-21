# Golden skip vs pass

`model_golden_test` exits **77** when every selected case skipped for missing weights.
CTest maps that via `SKIP_RETURN_CODE 77` so an opted-in run without weights is a **skip**, not a green pass.
Hosted CI sets `MORTRED_CI_REQUIRE_WEIGHTS=1` to fail instead of skip (`scripts/ci_assert_gtest_xml.py`).

`model_golden_test`, `Yolov8TrtDetection`, and `yolov8_detection` carry label `heavy-golden`.
Default `check` excludes that label. The name list and alternate gates are `test/CHECK_SSOT.md`.
Exit 77 applies when the binary is run directly or under `MORTRED_CHECK_HEAVY=1`, not as a pass inside default `check` (default `check` does not run these ctests).
