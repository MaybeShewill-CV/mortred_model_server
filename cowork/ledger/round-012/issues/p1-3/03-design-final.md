# Design P1-3 v1 (user-locked)
## Goal
enlightengan.inl compiles named_tensor_as; contract UT stays green.
## Bounds
Include only; no inference logic change.
## Fix
Add #include "models/backend/model_runtime.h" to enlightengan.inl.
