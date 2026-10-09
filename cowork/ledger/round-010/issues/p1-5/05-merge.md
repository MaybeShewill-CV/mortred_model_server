# 合入

- 问题: P1-5
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: G:\Codex\mortred-cowork-r10-p1-5
- 问题分支提交: 3f3f607a9b096e33004dcc9d3e3c5194ef0d0610
- QA 结论: pass
- 重跑: 验收 1 退出码 0，2 为 0，4 为 0，5 为 0，白盒为 0，6 为 0、8、0。与 `05-qa-pass.md` 一致。engine 已改回原名。
- 命令: `git merge --no-ff cowork/r10-p1-5-trt-golden`
- merge commit: 无

不合入。`test/CMakeLists.txt` 可以自动合并。`test/model_golden_test.cc` 内容冲突：基线一侧是 D-3 的 `golden_process_exit` 与自定义 `main`，问题分支一侧是 `Yolov8TrtDetection`。已执行 `git merge --abort`。基线工作区恢复干净。
