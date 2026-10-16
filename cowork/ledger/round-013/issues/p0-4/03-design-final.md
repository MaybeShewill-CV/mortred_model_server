# Design P0-4 v1 (user-locked)
## Goal
Lower handler_threads defaults tied to workers; capacity sheet; RSS note in occupancy if feasible.
## Fix
1. In-class default handler_threads 50 → 8 (single-GPU friendly).
2. After load: if handler_threads > max(8, worker_nums*8), WARNING clamp? Prefer: if omitted use default 8; leave explicit 50 alone but rewrite shipped tomls that still say 50 down to 16 when worker_nums==1.
3. Add docs/capacity-single-gpu.md sheet (threads/RSS/GPU sketch).
4. Occupancy: document optional measured RSS residual — add comment in occupancy_gate that process RSS is out-of-band via pack_occupancy measured-json (already has measured_gpu_mem); no fake RSS sampler.
