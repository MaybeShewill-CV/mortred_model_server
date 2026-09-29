# 合入

- 问题: P1-2
- 分支: cowork/r04-p1-2-depth-contract
- 工作树: /tmp/mortred-cowork-r04-p1-2
- 问题分支提交: 118e548c
- QA 结论: pass
- 重跑: `scripts/run_tests.sh build/full -R 'mono_depth_output_contract'` 退出码 0（1/1 Passed）；`rg -n GTEST_SKIP test/mono_depth_output_contract_unittest.cc` 退出码 1；`rg -n mono_depth_output_contract_unittest test/CMakeLists.txt` 退出码 0
- merge commit: a78c3149
