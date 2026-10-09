# 第3步签字

- 问题: P1-6
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `inference-paths` 已在 `ubuntu-22.04`、`if: ${{ always() && ... }}` 下用 python3 读取 `needs.cpu-profile.result` 与 `needs.gpu-pr-gate.result`，并把 `success|skipped` 一起 `exit 0`。该步可改成只调用 `scripts/ci_eval_inference_paths.py`，实参写成 `'${{ needs.cpu-profile.result }}'` 与 `'${{ needs.gpu-pr-gate.result }}'`。WSL 的 awk 按两个空格的 job 名切片：`inference-paths` 是最后一个 job，切片含现有 `if: ${{ always()`；`gpu-pr-gate` 在 `gpu-nightly-full` 之前截断，切片含 `vars.MORTRED_HAS_GPU_RUNNER == 'true'`。这两条 `rg -F` 现在退出码都是 0，判定步按方案改完后这两行保持原样即过。`continue-on-error` 在该 job 切片里现在无命中，退出码 1。`success…skipped…exit 0` 现在命中内联分支；删掉该分支后同一条 rg 退出码为 1。`ci_print_tier_summary.py` 在 `--gpu-smoke skipped` 时已打印 `SKIPPED` 与 T2 `NOT RUN` 且退出码 0，`--self-test` 只断言这些子串存在，补上 `This result is not a TensorRT or CUDA proof.` 后仍为 0。`docs/ci-golden-regression.md` 里 `skip as success` 只在第 24 行；第 68 行的 `skip-as-pass` 对不上 `-F "skip as success"`。换成同时含 `exits 2`、`NOT_A_TRT_PROOF` 与 `This result is not a TensorRT or CUDA proof.` 的那一句后，第 10 条五条检索的退出码与方案一致。`deploy-tools` 已有 `py_compile` 与 `--self-test` 列表，挂上新脚本即可。`gpu-pr-gate` 的 `runs-on: [self-hosted, X64, gpu]` 与变量条件保持不动。一个 catalog id 仍是一个 OS 进程，后端仍是 MNN、ONNX Runtime、TensorRT。
