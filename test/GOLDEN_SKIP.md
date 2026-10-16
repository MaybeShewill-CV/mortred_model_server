# Golden skip vs pass

`model_golden_test` exits **77** when every selected case skipped for missing weights.
CTest maps that via `SKIP_RETURN_CODE 77` so local runs without weights are **skips**, not green passes.
Hosted CI sets `MORTRED_CI_REQUIRE_WEIGHTS=1` to fail instead of skip.
