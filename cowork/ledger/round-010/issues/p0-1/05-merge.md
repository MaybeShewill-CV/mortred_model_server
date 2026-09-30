# 合入

- 问题: P0-1
- 分支: cowork/r10-p0-1-sam-status
- 工作树: /tmp/mortred-cowork-r10-p0-1
- 问题分支提交: e90fda45c49fd2fe85cd8a826555d06cf52812dc
- QA 结论: pass
- 重跑: `scripts/run_tests.sh build/full -R SamOutputContract --output-on-failure` 退出码 0；`ctest --test-dir build/full -R '^SamOutputContract$' --output-on-failure` 退出码 0
- 命令: `git merge --no-ff cowork/r10-p0-1-sam-status`
- merge commit: 08b107f810470edb49d1e5bde0a63b44727f8d99
