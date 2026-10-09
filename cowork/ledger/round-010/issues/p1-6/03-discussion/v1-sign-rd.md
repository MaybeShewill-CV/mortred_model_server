# 第3步签字

- 问题: P1-6
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `inference-paths` 已在 `ubuntu-22.04`、`if: always()` 下读取 `cpu-profile` 与 `gpu-pr-gate` 的 `needs` 结果，并把 `success|skipped` 一起 `exit 0`。该分支可抽成脚本，在 `cpu=success` 且 `gpu=skipped` 时改为退出码 2。`gpu-pr-gate` 的 `if` 已是 `vars.MORTRED_HAS_GPU_RUNNER == 'true'` 且同源 push/PR，`runs-on` 仍是 `[self-hosted, X64, gpu]`；fork 与未设变量的结果都是 `skipped`。`ci_print_tier_summary.py` 只打印且退出码为 0，PR/push 的 T2 已是 NOT RUN，gpu-smoke skipped 已印 SKIPPED，补一句证明声明不影响不传 `--gpu-smoke` 的 nightly 与 `t2-tier-notice`。`docs/ci-golden-regression.md` 第 21–25 行现写 skip 在 cpu-profile 成功时算成功，可改成退出码 2 与 `NOT_A_TRT_PROOF`。不改一 catalog 一进程，也不改 MNN、ONNX Runtime、TensorRT。
