# 第3步签字

- 问题: P1-3 已有 fail-closed 黑盒脚本未进默认 ctest
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `test/CMakeLists.txt` 已有 `WORKING_DIRECTORY ${PROJECT_ROOT_DIR}` 的 `add_test`，根 CMake 无条件 `add_subdirectory(test)`，可在 foreach 外无条件再挂 `p0_1_engine_fail_closed`（bash + 脚本路径）并 `set_tests_properties(... RUN_SERIAL TRUE)`，不打 valgrind label 即可进默认 `check`/`-LE valgrind`；`scripts/test_p0_1_engine_fail_closed.sh` 中 `skip` 不增 `FAILS`，AC8 无 GPU/`trtexec`/onnx 走 skip，`FAILS=0` 时 `exit 0`，无需改脚本或 `src/`。
