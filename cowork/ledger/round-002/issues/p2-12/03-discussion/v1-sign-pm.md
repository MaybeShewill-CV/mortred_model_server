# 第3步确认

- 问题: P2-12
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 四条验收标准都已编号。一次打开两份教程，就能判定分类命令的 `--model` 是 `mobilenetv2`、特征点命令的 `--model` 是 `superpoint`，且这两处不再是 `--model MOBILENETV2`、`--model SUPERPOINT`。同一轮打开还能判定另外三条命令、`worker_nums` 原句、节标题和教程里没有 `densenet`；再打开现有 catalog 头文件、两份 server toml 和 `model_server_main.cpp`，就能判定 id、`model=` 和大小写规则仍是现在的样子。不需要拉起进程。
