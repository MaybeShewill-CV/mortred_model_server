# 第3步签字

- 问题: D-3
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的两处条件已经写进验收第 8 条、第 9 条和对应自测。第 8 条对 `docs/ci-golden-regression.md` 与 `docs/model-developer-guide.md` 分开检索，每份都要同时有 `NUMERICAL_NOT_RUN`、`77`，以及「单条缺权重仍可以跳过」或 `case skips`；缺单条跳过原文时该条退出码不是 0。`test/ci_require_weights.h` 不含 `stays green` 单独看退出码 1，不替代两份文档。第 9 条只取 `gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步：`MORTRED_CI_REQUIRE_WEIGHTS=1` 只在该步第一次 `unset` 之前查找；`77`、`gpu-rest.xml`、`--allow-skips` 只在这句 `unset` 到该步第一次 `gpu-rest-skips.json` 之前查找。更早位置的 `MORTRED_CI_REQUIRE_WEIGHTS=1` 不能让这条通过。验收 1 至 7 仍只看退出码、标准输出和 ctest 结果文字。这些句子都可以黑盒判定。
