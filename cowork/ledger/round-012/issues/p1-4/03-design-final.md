# Design P1-4 v1 (user-locked)
## Goal
Dual-upstream isolation: two model ids; stop one; other still 200; dead returns 503.
## Bounds
No packing/Docker; reuse fake_model_server + gateway.
## Fix
Extend gateway_e2e_test with dual fake upstreams catalog; assert isolation after stop.
