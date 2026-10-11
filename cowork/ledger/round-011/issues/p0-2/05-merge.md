# 合入

- 问题: P0-2
- 分支: cowork/r11-p0-2-doctor-id
- 工作树: G:\Codex\mortred-cowork-r11-p0-2
- 问题分支提交: dd1699492955de96347726a5c8b9c914d1109251
- QA 结论: pass
- 重跑: 工作目录 `/mnt/g/Codex/mortred-cowork-r11-p0-2`，`LD_LIBRARY_PATH` 含该工作树 `3rd_party/libs`。冻结名单 39 个 id 各执行 `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id <id>`，退出码都是 0。验收 6、7、8 的 `yolov8s`、`metric3d_512`、`yolov8n` 含在这 39 次里。
- 命令: `git merge --no-ff cowork/r11-p0-2-doctor-id`
- merge commit: 6389504638dccc00dbae60987c1c4b7fcfb55350
