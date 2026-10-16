# Round 6 live progress board

- progress.html — clickable issue cards; detail via hash #issue/<id> (in-page panel). Meta refresh **60**s while running; **stops** when all merged.
- progress_data.json — machine-readable status from ledger artifacts.
- update_round006_progress.py — regenerates JSON+HTML; exits 0 always; prints DONE when 100%.
- progress_watch.sh — every 60s; exits and removes pid when percent==100.
- issue.html — deep-link ?id=p1-1 detail view (also reachable from cards).

## Start watcher (WSL)



Open progress.html (file://). Cards open detail with design/rd/qa/merge status, artifact paths, merge SHA, and current step text.
