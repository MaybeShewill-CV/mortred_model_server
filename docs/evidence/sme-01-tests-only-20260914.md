# SME-01 evidence — tests-only archive

- **Date:** 2026-09-14
- **Host:** DESKTOP-B8MJFUS (WSL2), path `/mnt/g/Codex/mortred_model_server`
- **Git:** `d51b0287` — `Stop tracking 3rd_party headers: install_deps.sh is the source of truth.` (`main` / `origin/main`)
- **Toolchain:** cmake 4.4.2, g++ 11.4.0, `nproc=16`
- **Kernel:** Linux 6.18.33.2-microsoft-standard-WSL2 x86_64

## Commands (final successful path)

```bash
cmake --preset tests-only
# configure_exit=0 (vendored workflow + libcrypto.so.3)

# Default preset build only builds non-EXCLUDE_FROM_ALL libs (common/control).
# Full unit suite uses the check target (same as CI):
LD_LIBRARY_PATH="$PWD/_lib:${LD_LIBRARY_PATH:-}" \
  CTEST_OUTPUT_ON_FAILURE=1 \
  cmake --build build/tests-only --target check -j"$(nproc)"
```

## Result

| Item | Value |
|---|---|
| `check_exit` | **0** |
| ctest | **100% tests passed out of 53** |
| Wall time | **60.18 s** |
| sanitizer label | 5 tests, 8.69 sec*proc |

Notable passed tests include: `batch_collector_unittest`, `do_work_lifetime_unittest`, `async_job_*`, `server_e2e_contract_test`, `supervisor_*`, `gateway_multiinstance_test`, `occupancy_gate_unittest`, `trt_spawn_gate_unittest`.

## Blockers encountered (resolved for this archive)

1. **Default `cmake --build --preset tests-only`** only built `common` / `control` / `control_workflow` because tests are `EXCLUDE_FROM_ALL`. Must use `--target check`.
2. **Truncated `3rd_party/include/toml/toml.hpp`** (~13332 lines) caused cascading glog/OpenCV/`namespace` errors. Complete toml++ v3.4.0 single header is ~17700+ lines.
3. **`install_deps.sh` toml install** can hang (bare `curl` without timeouts) and skips re-download if a truncated file already exists (`[ -f ] \|\| curl`).
4. **Browser-fetched full header without `TOML_EXCEPTIONS=0` prepend** broke unittests (`.table()` / `!parsed`). Project requires exceptions-off API. After prepend, build+ctest succeeded.
5. **GPU `install_deps.sh --all` / MNN CUDA+Cutlass** was NOT required for this archive; deferred (Cutlass header / fetch issues on this machine).

## GPU smoke

**Skipped** for SME-01 (optional). Reason: full GPU `--all` blocked on MNN `MNN_Cuda_Main` / Cutlass; tests-only path does not need it.

## SME-01 acceptance

**Met** for Linux tests-only on this host at `d51b0287`: archived commands, exit code 0, 53/53 passed. Follow-ups: harden `install_deps.sh` toml one-shot install; optional GPU smoke under SME-02 / later.
