# SME-06 evidence — start/stop/restart HTTP status (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-06-action-http-status` @ `eb9ef5d`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`

## Goal

`POST /api/v1/servers/{id}/start|stop|restart` must not stay HTTP 200 on business failure. Align with `graceful_restart` (200 / 500 + body `ok`).

## Unit / multiinstance

```text
cmake --preset tests-only
cmake --build --preset tests-only --target supervisor_multiinstance_test -j$(nproc)
./_bin/supervisor_multiinstance_test --gtest_filter='*start_stop_restart*:*failed_start_stop*'
→ start_stop_restart_return_json_ok PASSED
→ failed_start_stop_return_http_500 PASSED
→ ut_exit=0
```

Fixture notes locked in with this SME: `profile="any"`, copy `fake_model_server` into instance `_bin/`, CMake `MORTRED_FAKE_BIN_DEFAULT` for this target.

## Outcome

**PASS.** Failure path returns HTTP 500 + `ok:false`; success path HTTP 200 + `ok:true`. Docs (`api-contract`) and UI updated.
