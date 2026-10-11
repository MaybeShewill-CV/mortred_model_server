# 第3步签字

- 问题: D-8
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 用默认镜像交付的人，在仓库根目录执行 `docker build -t mortred_model_server:gpu .` 且退出码为 0 之后，用冻结命令 `docker run --rm --entrypoint id mortred_model_server:gpu -u` 验收。验收 1 规定该命令退出码 0，标准输出去掉末尾唯一换行后原文等于 `10001`，没有第二行，标准错误为空。同一次命令的退出码和输出原文就能判定这条 `id -u` 不再打印 `0`。
