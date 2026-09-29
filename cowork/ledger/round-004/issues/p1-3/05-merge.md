# 合入

- 问题: P1-3
- 分支: cowork/r04-p1-3-fail-closed-ctest
- 工作树: /tmp/mortred-cowork-r04-p1-3
- 问题分支提交: 04c76439
- QA 结论: pass
- 重跑: tests-only/full `ctest -N -R` 退出码 0；valgrind 列表退出码 0 且 0 tests；脚本退出码 0；`-LE valgrind` ctest 退出码 0；破坏配置后 ctest 退出码 8，恢复后 0
- merge commit: 973ebabbacf8eed798bc8b421d473867a8e97f14
