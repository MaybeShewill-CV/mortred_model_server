# 第3步签字

- 问题: P1-6
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收第 7 条改为对照 `.github/workflows/ci.yml` 原文，判定步把 `needs.cpu-profile.result` 与 `needs.gpu-pr-gate.result` 传给 `--cpu` 与 `--gpu`，不以 `--check-workflow` 退出码 0 代替。第 10 条要求 `docs/ci-golden-regression.md` 改为退出码 2 与 `NOT_A_TRT_PROOF`，并写明这不是 TensorRT 或 CUDA 证明。`cpu=success` 且 `gpu=skipped` 仍是退出码 2，`NOT_A_TRT_PROOF` 与 `TRT_PROOF=yes` 未改。
