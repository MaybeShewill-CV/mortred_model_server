# Single-GPU capacity sheet (Round 13 P0-4)

Evidence basis: idle mobilenetv2 with `worker_nums=1` historically ≈ 61 threads / hundreds MB RSS when `handler_threads=50`. Defaults are now lower so a mid-enterprise 16 GB host can answer “how many ids?”.

## Defaults (code + shipped tomls)

| Knob | Old common | New default / shipped |
| --- | --- | --- |
| `handler_threads` (in-class) | 50 | **8** |
| `handler_threads` (shipped server tomls that were 50) | 50 | **16** |
| `worker_nums` | 1 | 1 (unchanged) |
| `max_queue_depth` | 32 (after P1-1) | 32 |

Rule of thumb: prefer `handler_threads ≈ max(8, worker_nums * 8)` on a single GPU node. Workflow library threads dominate; raising handler_threads past ~16 for `worker_nums=1` rarely improves inference RPS and multiplies process RSS/fd.

## Rough residency budget (order-of-magnitude)

| Id class | Threads (handler=16, workers=1) | RSS (CPU path) | GPU mem |
| --- | --- | --- | --- |
| Small classifier (mobilenet) | ~25–35 | ~0.3–0.5 GiB | engine-dependent |
| YOLO-class detector | ~25–40 | ~0.5–1.5 GiB | see `gpu_mem_mib` stamp |
| Heavy diffusion / SAM | higher | higher | stamp required |

44 ids × ~30 threads ≈ **1.3k threads**; 44 × ~0.5 GiB ≈ **22 GiB RSS** if everything is resident — still too much for 16 GB RAM. Practical packing: autostart a **pack** subset, not the whole catalog; use `mortredctl calibrate` / occupancy stamps for VRAM.

## RSS dimension in occupancy

Round 14: `mortredctl calibrate --write-pack` records per-id `rss_mib` / `thread_count` / `fd_count` floors and pack `host_memory_total_mib`. `occupancy_ready_for_spawn` enforces the host RAM joint budget (and live MemAvailable) when those stamps are present — not docs-only.

## How to size a host

1. Pick the pack ids that must be hot.
2. Sum `gpu_mem_mib` stamps + reserve (occupancy policy).
3. Budget ~0.4–1.0 GiB RSS × id count under `handler_threads<=16`.
4. Cap concurrent autostart so threads stay under ~1–2k on a 16-core box.
