# 合入

- 问题: P1-1
- 分支: cowork/r04-p1-1-drain-timeout
- 工作树: /tmp/mortred-cowork-r04-p1-1
- 问题分支提交: 193bc362eba0bb66565c5054c22d7f0396a2e11a
- QA 结论: pass
- 重跑: `worker_pool` 退出码 0；`async_job_unittest` 退出码 0
- 命令: `scripts/run_tests.sh build/tests-only -R 'worker_pool'`
- merge commit: e7b63399
