# Round 13 full report (P0-2 → P1-8)

- **Board:** `cowork/ledger/round-013/progress.html` (+ `progress_data.json`)
- **Final HEAD:** `e200ab9998355eb4a75869a3c36d42896c31f685` (`e200ab99`) on `sub-agent-co-work`
- **Status:** 11/11 done (P0-1 skipped — already closed at tip `fb982277`)
- **Generated:** 2026-10-16 21:46:01 CST

## Merge table

| Order | ID | Branch SHA | Merge SHA | One-liner |
| --- | --- | --- | --- | --- |
| 1 | P0-2 | 8453c81a | 514b044d | TRT Staging/D2H slot sync-then-release on error abandon |
| 2 | P1-5 | 0a36f1c7 | f0bfef79 | SuperPoint sort comparator `>=` → `>` |
| 3 | P0-3 | aad2bc46 | e1550d2f | `http_infer_rps` allow warmup 0; parse before self-test |
| 4 | P1-1 | f915f595 | a9ac51c2 | `max_queue_depth` default 32; fill 5 tomls; 429+Retry-After kept |
| 5 | P1-4 | ab777c17 | e804f862 | Resolve relative `MORTRED_CONTROL_CONFIG` via project_root |
| 6 | P1-6 | 0409d986 | c14255a9 | JPEG SOF + PNG IHDR size probe before full imdecode |
| 7 | P1-7 | 11bf407f | 7bf9755a | systemd `KillMode=control-group` + PDEATHSIG orphan note |
| 8 | P0-4 | 0ede69ea | 2a1f070d | `handler_threads` 8 default / tomls 16 + capacity sheet |
| 9 | P1-2 | 09b0ed2b | fec578f3 | YOLOv8 unmap-then-NMS (align v5/v6/v7) |
| 10 | P1-3 | cc3a8ff8 | cf322469 | Accept `MORTRED_METRICS_TOKEN` on model `/metrics` |
| 11 | P1-8 | c363ea97 | e200ab99 | Golden skip exit-77 visibility + sanitizer label SSOT comment |

## Residuals (honest)

- No GPU live fault-injection for TRT mid-H2D failure (P0-2).
- Process RSS not sampled inside occupancy gate — documented out-of-band only (P0-4).
- Broader matting/seg/libface family audit parked (P1-2).
- Non-JPEG/PNG codecs still size-check after imdecode (P1-6).
- P1-8 was mostly docs; exit-77 behavior already existed.
- Packing/Docker still out of scope (except systemd KillMode under deploy/).
