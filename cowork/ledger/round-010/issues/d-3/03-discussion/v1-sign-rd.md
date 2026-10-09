# 第3步签字

- 问题: D-3
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 缺权重和 `golden updated` 已经是两条不同的 `GTEST_SKIP` 文案，`RUN_ALL_TESTS` 返回后仍能按条区分跳过、失败和做完比对，xml 也在这之前写完。`model_golden_test` 现在只是多链了 `gtest_main`，可以改成自己的 `main`，只在全部选中用例都是 `weights not available` 时退出 77 并打一行 `NUMERICAL_NOT_RUN`。根构建要求 CMake ≥ 3.18，`SKIP_RETURN_CODE` 77 只加在测试名 `model_golden_test` 上，混合结果仍是退出码 0、CTest Passed；`backend_unittest` 继续走 `gtest_main`。`gpu-nightly-full` 里剩余 golden 现在在 `set -e` 下直接跑，按方案只接住 77 再进现有 `--allow-skips` 即可，smoke-8 仍是 `MORTRED_CI_REQUIRE_WEIGHTS=1`。
