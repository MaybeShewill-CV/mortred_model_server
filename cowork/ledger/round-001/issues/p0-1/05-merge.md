# 合入

- 问题: P0-1
- 分支: cowork/r01-p0-1-engine-fail-closed
- 工作树: /tmp/mortred-cowork-r01-p0-1
- 问题分支提交: fa0a8003、7c594c68、b85dcfea
- QA 结论: pass
- 重跑: `pack_trt --check` 退出码 0；临时 stamp pack 的 occupancy `--check` 退出码 0；`mortredctl_doctor.sh` 退出码 1，engine/stamp 两段为 [ok]，失败句是 `live checks skipped`
- 命令: `git merge --no-ff cowork/r01-p0-1-engine-fail-closed`
- merge commit: 4f0d6af9
