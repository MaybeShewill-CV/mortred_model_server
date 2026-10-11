# 合入

- 问题: P1-8
- 分支: cowork/r11-p1-8-cpu-golden
- 工作树: G:\Codex\mortred-cowork-r11-p1-8
- 问题分支提交: 42cf0f45
- QA 结论: 18 条 CPU 数值用例此前已通过。`msocrnet` 按用户决定未跑。
- 重跑: 冲突已在问题分支解开后再合入，基线合并没有冲突。`model_golden_test` 在工作树 `build/full` 下编译退出码 0。没有跑 `model_golden.msocrnet`，没有跑 19 条放进同一进程的过滤。
- 命令: `git merge --no-ff cowork/r11-p1-8-cpu-golden`
- merge commit: c6f164dd59796536ee03135fd74966d466751b0f

合入的是 18 份 CPU 黄金和对应 CI toml：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。`conf/ci/msocrnet_onnx_hosted.toml` 和 `model_golden.msocrnet` 登记一并进入基线。`test/golden/msocrnet.json` 没有生成。这不是 `NUMERICAL_PASSED`。

用户于 2026-10-11 决定不再在这台 WSL 上测试 `msocrnet`，因为机器内存不够。
