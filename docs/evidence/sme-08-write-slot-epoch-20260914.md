# SME-08 evidence — lock-free write_slot + submit/stop epoch (WSL)

- **Date:** 2026-09-14
- **Branch / HEAD:** `fix/sme-08-write-slot-epoch` @ `e6492fe`
- **Host:** WSL (`DESKTOP-B8MJFUS`), `/mnt/g/Codex/mortred_model_server`
- **Constraint:** no new mutex; atomic claim→write→publish + `_accepting`/epoch

## Commands / results

```text
./_bin/batch_collector_unittest --gtest_filter='*write_slot*:*submit*'
→ write_slot_notifies_once_on_last PASSED
→ write_slot_concurrent_same_index_publishes_once PASSED
→ submit_stop_race_all_slots_publish PASSED
→ submit_inline_go_runs_batch_and_notifies PASSED
→ submit_when_stopped_timeouts_all_slots PASSED
→ ut_exit=0
```

## Outcome

**PASS.** Concurrent same-index writers publish once; submit/stop races leave all slots published (TIMEOUT fixup / drain) without new locks.
