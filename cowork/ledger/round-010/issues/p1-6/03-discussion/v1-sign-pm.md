# 第3步签字

- 问题: P1-6
- 方案版本: v1
- 结论: 不同意
- 条件: 验收第 7 条改为对照 `.github/workflows/ci.yml` 原文判定，不得只凭 `--check-workflow` 退出码 0：`inference-paths` 的判定步调用 `scripts/ci_eval_inference_paths.py`，并把 `needs.cpu-profile.result` 与 `needs.gpu-pr-gate.result` 传给 `--cpu` 与 `--gpu`；该 job 的 `if` 含 `always()`；该步没有 `continue-on-error`；该 job 中不存在把 `skipped` 与 `success` 一起 `exit 0` 的分支；`gpu-pr-gate` 的 `if` 仍含 `vars.MORTRED_HAS_GPU_RUNNER == 'true'`。验收增加一条：`docs/ci-golden-regression.md` 不再写无 `MORTRED_HAS_GPU_RUNNER` 时 wrapper 把 skip 当成成功，改为退出码 2 与 `NOT_A_TRT_PROOF`，并写明这不是 TensorRT 或 CUDA 证明。
- 理由: 验收 1–6、8、9 能用退出码和输出判定。第 7 条把「skipped 不再与 success 一起退出 0」收成脚本退出码，现场看不到 needs 结果有没有传入脚本。文档里「skip 仍算成功」也没有编号验收。模型目录、HTTP/tensor 契约和不支持边界与这条 CI 收口一致。
