# 第3步签字

- 问题: P1-3 已有 fail-closed 黑盒脚本未进默认 ctest
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 推荐做法只动 `test/CMakeLists.txt` 注册既有脚本，用例名、`WORKING_DIRECTORY`、`RUN_SERIAL`、不打 valgrind label、不改脚本退出码/AC8 skip 与 `src/` 均写死；AC1–AC6 可黑盒判定，自测命令可执行且退出码约定清楚，否决项合理，不扩大到其他列表项。
