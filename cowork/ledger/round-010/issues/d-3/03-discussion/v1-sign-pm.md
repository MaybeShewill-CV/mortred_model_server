# 第3步签字

- 问题: D-3
- 方案版本: v1
- 结论: 不同意
- 条件: 验收第 8 条改为两份文档分开判定：`docs/ci-golden-regression.md` 与 `docs/model-developer-guide.md` 各自都含 `NUMERICAL_NOT_RUN` 和 `77`，并且各自含原文 `单条缺权重仍可以跳过` 或 `case skips`。自测对每个文件单独检索；只命中 `NUMERICAL_NOT_RUN` 和 `77`、没有上述单条跳过原文时，第 8 条不通过。`test/ci_require_weights.h` 不含 `stays green` 单独判定。验收第 9 条改为只看 `gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步：该步在 `unset MORTRED_CI_REQUIRE_WEIGHTS` 之前仍含 `MORTRED_CI_REQUIRE_WEIGHTS=1`；从这句 `unset` 到该步第一次 `gpu-rest-skips.json` 之前，同时含 `77`、`gpu-rest.xml` 和 `--allow-skips`。同文件更早位置的 `MORTRED_CI_REQUIRE_WEIGHTS=1` 不能算这一步通过。
- 理由: 验收 1 至 7 可以看退出码、标准输出和 ctest 的结果文字判定。第 8 条要求两份文档写明单条缺权重仍可以跳过，自测只检索 `NUMERICAL_NOT_RUN` 和 `77`。只写退出码 77、不写单条仍可跳过时，按句子失败，按命令通过。第 9 条要求 smoke-8 那段仍含 `MORTRED_CI_REQUIRE_WEIGHTS=1`，自测用的是全文里 `unset` 之前最后一次出现。删掉 smoke-8 这一步里的这句、更早 job 仍留着时，按句子失败，按命令通过。这两条不能唯一判定。
