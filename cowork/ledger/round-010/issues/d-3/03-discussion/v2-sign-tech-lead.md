# 第3步签字

- 问题: D-3
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收第 8 条对两份文档分开判定，各自都要有 `NUMERICAL_NOT_RUN`、`77`，以及 `单条缺权重仍可以跳过` 或 `case skips`；自测按文件单独检索，缺单条跳过原文则不通过。`test/ci_require_weights.h` 不含 `stays green` 单独判定。验收第 9 条只看 `gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步：第一次 `unset MORTRED_CI_REQUIRE_WEIGHTS` 之前仍有 `MORTRED_CI_REQUIRE_WEIGHTS=1`，从这句 `unset` 到该步第一次 `gpu-rest-skips.json` 之前同时有 `77`、`gpu-rest.xml` 和 `--allow-skips`。同文件更早的 `MORTRED_CI_REQUIRE_WEIGHTS=1` 不能算通过。验收第 1 至第 7 条仍能用退出码、标准输出和 ctest 结果文字分开「未跑数值」与「已通过」，范围停在已冻结的 D-3。
