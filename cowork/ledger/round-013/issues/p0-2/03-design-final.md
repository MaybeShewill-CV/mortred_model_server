# Design P0-2 v1 (user-locked)
## Goal
TRT H2D/D2H pinned staging slots must not be pool-released while the session stream may still touch them on error returns.
## Bounds
No packing/Docker. TensorRT path only.
## Root cause
StagingSlotGuard.abandon never set; D2HSlotGuard had no abandon; cudaFreeHost-without-sync also unsafe.
## Fix
Both guards hold cudaStream_t; abandon => sync-then-release to pool. Error returns set abandon before matching successful sync.
