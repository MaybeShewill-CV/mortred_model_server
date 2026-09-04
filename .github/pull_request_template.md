## Summary

<!-- What changed and why. -->

## Inference path

- [ ] I did **not** treat a green fork `cpu-profile` as TensorRT/CUDA proof.
- [ ] HTTP catalog / golden changes update `conf/ci_hosted_golden.json` when needed.
- [ ] TensorRT-only edits expect maintainer `gpu-pr-gate` (`MORTRED_HAS_GPU_RUNNER=true`).

## Exposure

- [ ] External access stays on a TLS reverse proxy; compose still binds `127.0.0.1`.
- [ ] I did not publish model ports (`9001+`).

## Test plan

<!-- How you verified. -->
