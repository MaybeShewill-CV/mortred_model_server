# 合入

- 问题: P1-3
- 分支: cowork/r01-p1-3-install-host
- 工作树: /tmp/mortred-cowork-r01-p1-3
- 问题分支提交: d9bc3db639208c935015e85e0509e094face776b
- QA 结论: pass
- 重跑: 与 `05-qa.md` 命令表一致。安装退出码 0，`systemctl is-enabled mortred-supervisor` 退出码 0（enabled），非 root 退出码 1，重复安装退出码 0，`--with-edge` 退出码 1，`is-active` 退出码 3（inactive）
- 命令: `/mnt/c/Windows/System32/wsl.exe -u root -- bash /tmp/mortred-cowork-r01-p1-3/scripts/mortredctl_install-host.sh --from-tarball-root /tmp/p13-unpack`
- merge commit: 985682ee
