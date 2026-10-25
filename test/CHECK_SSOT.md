# Default `check` exclusion (SSOT)

Default `cmake --build <dir> --target check` runs ctest with
`-LE valgrind|heavy-golden|e2e|stress`. It does not execute the ctest names in
the table. `MORTRED_CHECK_HEAVY=1` runs everything except `valgrind`: it
includes `heavy-golden`, `e2e` and `stress`. A green default `check` is not a
numerical golden proof and is not a full-suite green.

The e2e and stress lanes are opt-in on their own: `ctest -L e2e` and
`ctest -L stress`. They start real gateways, supervisors and listeners, so one
of them aborting used to cut the default suite short (a supervisor multi-instance
abort stopped a full gate at 59/96).

On a full build the heavy binaries are still compiled (`add_dependencies(check …)`),
because hosted and GPU jobs invoke `model_golden_test` directly afterwards.
Default `check` does not run them.

A bare `ctest` without `-LE valgrind|heavy-golden|e2e|stress` is not this target.
`scripts/run_tests.sh` without `-LE` runs every registered ctest, including
`heavy-golden`, `e2e` and `stress`.

`scripts/check_check_ssot.py` (also invoked from `scripts/check_consistency.py`)
fails if this table, the CMake labels, or a README / `docs/ci-golden-regression.md`
claim drifts.

| ctest name | label | why | alternate gate |
| --- | --- | --- | --- |
| gateway_e2e_test | e2e | Starts a real gateway plus fake upstreams and drives the whole proxy path; seconds of wall clock and a listener per suite. | ctest -L e2e (CI runs it right after the default check step). |
| gateway_multiinstance_test | e2e | Two gateways in one process, each binding a port; deterministic only on a quiet machine. | ctest -L e2e. |
| server_e2e_contract_test | e2e | In-process HTTP server covering the full request envelope; the longest suite in the gate and the one whose abort used to hide later tests. | ctest -L e2e. |
| supervisor_malformed_e2e_test | e2e | Spawns the real supervisor binary against malformed configs; needs the built control-plane binary. | ctest -L e2e. |
| supervisor_multiinstance_test | e2e | Two supervisors in one process (ports are per process, so ordering matters); an abort here ended the default run at 59/96. | ctest -L e2e. |
| async_job_stress_test | stress | Deliberate load loop (submits for minutes of wall clock by design), also in the TSan list. | ctest -L stress (and ctest -L sanitizer under TSan). |
| model_golden_test | heavy-golden | Full in-process golden zoo. Historically OOM on about 20GB RAM, so it cannot sit inside default check and still be called an honest green suite. | Hosted cpu-profile runs the binary directly with MORTRED_CI_REQUIRE_WEIGHTS=1 and the filter from scripts/check_hosted_golden.py, then scripts/ci_assert_gtest_xml.py. GPU smoke uses MORTRED_GPU_SMOKE_FILTER on gpu-pr-gate. Nightly is gpu-nightly-full. Opt-in local: MORTRED_CHECK_HEAVY=1 cmake --build <dir> --target check (RUN_SERIAL, SKIP_RETURN_CODE 77 without weights). See docs/ci-golden-regression.md. |
| Yolov8TrtDetection | heavy-golden | CTest alias of model_golden_test filtered to Yolov8TrtDetection.*. Same binary and the same RAM class as the full golden. | Hosted cpu-profile runs the binary directly with MORTRED_CI_REQUIRE_WEIGHTS=1 and the filter from scripts/check_hosted_golden.py, then scripts/ci_assert_gtest_xml.py. GPU smoke uses MORTRED_GPU_SMOKE_FILTER on gpu-pr-gate. Nightly is gpu-nightly-full. Opt-in local: MORTRED_CHECK_HEAVY=1 cmake --build <dir> --target check (RUN_SERIAL, SKIP_RETURN_CODE 77 without weights). See docs/ci-golden-regression.md. |
| yolov8_detection | heavy-golden | CTest alias of model_golden_test filtered to model_golden.yolov8_detection. Same binary; kept with the other golden ctests. | Hosted cpu-profile runs the binary directly with MORTRED_CI_REQUIRE_WEIGHTS=1 and the filter from scripts/check_hosted_golden.py, then scripts/ci_assert_gtest_xml.py. GPU smoke uses MORTRED_GPU_SMOKE_FILTER on gpu-pr-gate. Nightly is gpu-nightly-full. Opt-in local: MORTRED_CHECK_HEAVY=1 cmake --build <dir> --target check (RUN_SERIAL, SKIP_RETURN_CODE 77 without weights). See docs/ci-golden-regression.md. |
| status_code_unittest-memory-check | valgrind | Curated memcheck of a weight-free unittest (A6). Not part of default check. | ctest -L valgrind when MORTRED_ENABLE_VALGRIND_TESTS=ON. See docs/ci-golden-regression.md. |
| base64_unittest-memory-check | valgrind | Curated memcheck of a weight-free unittest (A6). Not part of default check. | ctest -L valgrind when MORTRED_ENABLE_VALGRIND_TESTS=ON. See docs/ci-golden-regression.md. |
| param_spec_unittest-memory-check | valgrind | Curated memcheck of a weight-free unittest (A6). Not part of default check. | ctest -L valgrind when MORTRED_ENABLE_VALGRIND_TESTS=ON. See docs/ci-golden-regression.md. |
| tensor_contract_unittest-memory-check | valgrind | Curated memcheck of a weight-free unittest (A6). Not part of default check. | ctest -L valgrind when MORTRED_ENABLE_VALGRIND_TESTS=ON. See docs/ci-golden-regression.md. |
| session_io_unittest-memory-check | valgrind | Curated memcheck of a weight-free unittest (A6). Not part of default check. | ctest -L valgrind when MORTRED_ENABLE_VALGRIND_TESTS=ON. See docs/ci-golden-regression.md. |
