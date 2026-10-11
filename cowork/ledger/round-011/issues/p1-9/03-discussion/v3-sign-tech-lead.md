# 第3步签字

- 问题: P1-9
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: v3 把 centerface 的 `--shapes` 写成 `input.1:1x3x640x640`。本机这份 engine 经 `trtexec --dumpOptimizationProfile` 打印的输入是 `input.1`，opt 是 `1x3x640x640`。v2 使用 profile 名 `input` 时，标准错误是 `Cannot find input tensor with name "input"`，进程退出码 1，没有 `Throughput:`。libface、enlightengan、realesrgan 的 shapes 仍是各自 profile 的 name 和 opt。39 个 id、表 5 的列、指标只取本次最后一条 `Throughput:`，以及两条自测命令的期望退出码都可以在黑盒下判定。范围仍是冻结句 H3。
