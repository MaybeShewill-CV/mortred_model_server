# Design P1-2 v1 (user-locked)
## Goal
check and CI TSan structurally green.
## Bounds
No packing/Docker; do not widen sanitizer label set.
## Root cause
1. packed_batch_nvi_unittest / cv_image_input_unittest subclass BackendCvModel needing GpuDecodeSlot but must not link libmodels (own InferenceSession::create).
2. TSan builds 6 targets; ctest -L sanitizer also selects unbuilt rate_limit_gcra_unittest.
## Fix
1. EXTRA_SRCS gpu_jpeg_decoder.cpp for those two unittests.
2. Add rate_limit_gcra_unittest to CI TSan --target list.
