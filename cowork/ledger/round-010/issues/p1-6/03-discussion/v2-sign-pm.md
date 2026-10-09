# 第3步签字

- 问题: P1-6
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收 1–6、8、9 用退出码和声明字符串判定。第 7 条对照 `.github/workflows/ci.yml` 原文：判定步调用 `scripts/ci_eval_inference_paths.py`，并把 `needs.cpu-profile.result` 与 `needs.gpu-pr-gate.result` 传给 `--cpu` 与 `--gpu`；`always()`、无 `continue-on-error`、该 job 不再把 `skipped` 与 `success` 一起 `exit 0`、`gpu-pr-gate` 仍含 `vars.MORTRED_HAS_GPU_RUNNER == 'true'`，都绑在自测检索的退出码上，`--check-workflow` 退出码 0 不算通过。第 10 条用检索退出码判定 `docs/ci-golden-regression.md` 去掉「skip 算成功」，并写上退出码 2、`NOT_A_TRT_PROOF` 和不是 TensorRT 或 CUDA 证明。模型目录、HTTP/tensor 契约和不支持边界不在改动面里。
