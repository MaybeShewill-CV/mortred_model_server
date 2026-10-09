# 挂起

- 问题: P0-2
- 方案版本: v2
- 分支: cowork/r11-p0-2-doctor-id
- 工作树: G:\Codex\mortred-cowork-r11-p0-2
- 问题分支提交: dd1699492955de96347726a5c8b9c914d1109251
- QA 结论: blocked
- 缺的环境: 工作树里 `conf/product_static_bs1.json` 的 39 个 `onnx` 路径都没有文件。工作树 `3rd_party/bin` 不存在，WSL 的 PATH、`/usr/src/tensorrt/bin` 和有限 `find` 都没有 `trtexec`。主仓库 `G:\Codex\mortred_model_server\3rd_party\bin\trtexec` 的横幅是 TensorRT v100300。GPU 为 NVIDIA GeForce RTX 2070 SUPER。
- 恢复条件: 把这 39 个非空 onnx 放到工作树表内 `onnx` 路径，并使 `trtexec` 位于工作树 `3rd_party/bin`、PATH 或 `/usr/src/tensorrt/bin`，或把 `TRTEXEC` 指到已发现的 TensorRT 10.3 二进制。然后只重跑第 6、7、8、12 条。
- 未跑的验收: 6、7、8、12
- 重跑: 未合入。第 1 到第 5 条和第 9 到第 11 条的退出码以 QA 报告为准，调度者未再跑。
