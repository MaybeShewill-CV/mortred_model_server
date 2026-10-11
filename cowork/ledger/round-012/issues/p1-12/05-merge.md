# 合入

- 问题: P1-12
- 分支: cowork/r12-p1-12-trt-product
- 工作树: G:\Codex\mortred-cowork-r12-p1-12
- 问题分支提交: 611ecea4
- QA 结论: pass
- 重跑: 验收 1 至 4 退出码 0、0、0、0、0。验收 5 至 13 九条退出码都是 1。验收 14 十条退出码都是 0。验收 15 的 18 条更新和 18 条重跑退出码都是 0。验收 16 退出码 1，`mobilenetv2_classification.json` 的 sha256 前后都是 `91afb45fb017c97c631ed652853164b160c4635bd27bbcd4e2ce5dd59a2d7dc0`。验收 17 退出码 77。验收 18 退出码 1，engine sha256 前后都是 `244bb7fbbfd66eade8e0900023ace6ea02f587c74d1ef784ca72ac9af51a90ba`。验收 19 至 30 破坏退出码都是 1，恢复后退出码都是 0。`GOLDEN_BYTES_MATCH`。与 `05-qa.md` 一致。
- 命令: `git merge --no-ff cowork/r12-p1-12-trt-product`
- merge commit: 0b2a7598
