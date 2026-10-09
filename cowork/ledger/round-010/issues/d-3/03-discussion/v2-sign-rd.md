# 第3步签字

- 问题: D-3 缺权重时 golden 默认仍 SKIP
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `model_golden_test` 现在没有自己的 `main`，由 `test/CMakeLists.txt` 的 `GTEST_LIBS` 带上 `gtest_main`。改成自己的 `main` 并去掉 `gtest_main` 后，`RUN_ALL_TESTS()` 返回时 xml 已经写完，并能按条区分失败、`weights not available`、`golden updated` 和做完比对。`mobilenetv2_classification` 在 `run_case` 里先查权重，缺 `weights/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn` 时不会走到 `golden updated`。`--gtest_list_tests` 和零命中过滤都不是「全部缺权重」，可以保持 gtest 原退出码且不打两个标记。根构建要求 CMake ≥ 3.18，`SKIP_RETURN_CODE` 可用；该项是单个 CTest 测试，只给它设 77。混合结果退出码 0 时仍是 Passed。`backend_unittest` 继续用 `gtest_main`，同一文件缺失时该过滤仍是 `GTEST_SKIP`、退出码 0。`gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步，`MORTRED_CI_REQUIRE_WEIGHTS=1` 在 `unset` 之前，`gpu-rest.xml` 与 `--allow-skips` 已在 `unset` 和 `gpu-rest-skips.json` 之间；现在 `set -e` 会在 77 停住，只在这段接住 77 再进现有 `--allow-skips` 即可。两份文档各自写上 `NUMERICAL_NOT_RUN`、`77` 和单条仍可跳过的原文，并去掉 `test/ci_require_weights.h` 里的 `stays green`，宏分支不动。
