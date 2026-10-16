# Round 9 live progress board

- 队列: `02-p0-p1-final.md`（**v1 已锁定**）：4 × P0；0 × P1；0 × P2。
- progress.html — clickable issue cards; detail via hash #issue/<id>. Meta refresh **60**s while running; **stops** at 100%.
- progress_data.json — machine-readable status from ledger artifacts.
- update_round009_progress.py — regenerates JSON+HTML; prints DONE when 100%.
- progress_watch.sh — every 60s; exits and removes pid when percent==100.
- issue.html — deep-link ?id=p0-1 detail view.

## Start watcher (WSL)

```bash
cd /mnt/g/Codex/mortred_model_server/cowork/ledger/round-009
nohup bash progress_watch.sh >/dev/null 2>&1 &
```

Open progress.html (file://). Cards open detail with design/rd/qa/merge status and merge SHA.
