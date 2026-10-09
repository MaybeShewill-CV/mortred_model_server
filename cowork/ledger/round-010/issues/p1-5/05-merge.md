# 合入

- 问题: P1-5
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: G:\Codex\mortred-cowork-r10-p1-5
- 问题分支提交: 55f812fca57f2744d3350c9018cf38e7c30d2a4c
- QA 结论: pass
- 重跑: 验收 1 退出码 0，2 为 0，4 为 0，5 为 0，6 为 0、8、0。与 `05-qa-replay.md` 一致。engine 已改回原名。
- 命令: `git merge --no-ff cowork/r10-p1-5-trt-golden`
- merge commit: 1f267ea20c125f40e674242a5908a2583a18e331

先前直接合入冲突并已中止。问题分支先并入基线 `afff3131`，保留 `Yolov8TrtDetection` 与 D-3 的 `main`。这次 `merge-tree` 退出码 0，合入成功。
