# Round 8 live progress board

- 队列: `02-p0-p1-final.md`（v1，待第 2 步三方确认）。6 条 P0、7 条 P1、9 条 P2。
- progress.html — clickable issue cards; detail via hash #issue/<id> (in-page panel). Meta refresh **60**s while running; **stops** when all merged.
- progress_data.json — machine-readable status from ledger artifacts.
- update_round008_progress.py — regenerates JSON+HTML; exits 0 always; prints DONE when 100%.
- progress_watch.sh — every 60s; exits and removes pid when percent==100.
- issue.html — deep-link ?id=p0-1 detail view (also reachable from cards).

## Start watcher (WSL)

```bash
cd /mnt/g/Codex/mortred_model_server/cowork/ledger/round-008
nohup bash progress_watch.sh >/dev/null 2>&1 &
```

## Round 8 起点

- 审计基线: `10d3a764`（Round 7 已全部合入并关闭）。
- 队列来源: 外部只读源码审计（未读 docs/README）；每条"现在的行为"都在该基线上复验过，其中 P0-1 的两条命令为本机实测。
- 未生成 `issues/`：第 3 步（一问题一份方案）尚未开始。卡片在 `03-design-final.md` 出现前显示"待开始 · Step 3 方案"。
- 第 1 步（九份模块记录 + 评分复核）尚未跑；本目录不含 `01-modules/`、`01-score.md`、`01-score-check.md`。

Open progress.html (file://). Cards open detail with design/rd/qa/merge status, artifact paths, and merge SHA once the loop starts merging.
