# 合入

- 问题: P1-3
- 分支: cowork/r02-p1-3-prepare-entry
- 工作树: /tmp/mortred-cowork-r02-p1-3
- 问题分支提交: be6d692a
- QA 结论: pass
- 重跑: 缺 pack 退出码 1；`--help` 0；未知参数 1；`rg exec .*prepare_pack` 1
- 命令: `bash scripts/mortredctl_prepare.sh --pack /tmp/mortred-p1-3-missing-pack.toml`
- merge commit: 8c166d0b
