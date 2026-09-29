# 合入

- 问题: P1-3
- 分支: cowork/r01-p1-3-install-host
- 工作树: /tmp/mortred-cowork-r01-p1-3
- 问题分支提交: d9bc3db639208c935015e85e0509e094face776b
- QA 结论: blocked
- 重跑: `--help` 退出码 0；`systemctl is-system-running` 为 offline
- 命令: 未执行 merge
- merge commit: 无

未合入。AC1–AC4 需要一次真实的 systemd enable，当前环境做不到。
