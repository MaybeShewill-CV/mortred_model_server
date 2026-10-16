# Design P1-1 v1 (user-locked)
## Goal
Positive max_queue_depth by default; five missing tomls filled; full queue still 429+Retry-After.
## Fix
1. In-class default 0 → 32.
2. Load clamp: <=0 → default (WARNING), no silent unbounded.
3. Add max_queue_depth=32 to 4 diffusion + sam_amg server tomls.
4. 429+Retry-After path already in base_server_impl (unchanged).
