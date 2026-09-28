# 合入

- 问题: P0-1
- 分支: cowork/r01-p0-1-engine-fail-closed
- 工作树: /tmp/mortred-cowork-r01-p0-1
- 问题分支提交: fa0a8003、7c594c68、b85dcfea
- QA 结论: blocked
- 重跑: `bash scripts/test_p0_1_engine_fail_closed.sh` 退出码 0；AC8 在脚本中为 SKIP
- 命令: 未执行 merge
- merge commit: 无

未合入。QA 结论是 blocked，AC8 需要 GPU、trtexec 和 onnx。
