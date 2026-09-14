# SME-09 evidence — batch path honors request deadline (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-09-batch-deadline` @ `6667e8b`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`
- **Constraint:** no new mutex; filter expired before/after checkout; checkout wait capped by min remaining deadline

## Commands / results

```text
./_bin/batch_collector_unittest --gtest_filter='*expired_deadline*:*submit*'
→ expired_deadline_skips_worker_run PASSED
→ submit_inline_go_runs_batch_and_notifies PASSED
→ submit_when_stopped_timeouts_all_slots PASSED
→ (related submit filters green)
→ ut_exit=0
```

User confirmed full filtered suite OK on WSL after rebuild.

## Outcome

**PASS.** Past-deadline entries TIMEOUT without `run_batch` / worker hold; checkout wait respects remaining deadline; empty post-checkout batch checkins without model run.
