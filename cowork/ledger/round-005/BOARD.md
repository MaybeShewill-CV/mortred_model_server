# Round 5 live progress board

- `progress.html` — static snapshot (meta refresh **60**s). Prefer regenerating full HTML via the watcher (file:// fetch of JSON is unreliable).
- `progress_data.json` — machine-readable status derived from `issues/*/05-merge.md`.
- `update_round005_progress.py` — regenerates both files from the ledger.
- `progress_watch.sh` — loops every 60s calling the updater.

## Start watcher (WSL)

```bash
cd /mnt/g/Codex/mortred_model_server/cowork/ledger/round-005
nohup bash progress_watch.sh >/dev/null 2>&1 &
echo $! > progress_watch.pid
```

Open `progress.html` in a browser (file:// or any static server). The page reloads every 60s; the watcher refreshes file contents each minute as merges land.
