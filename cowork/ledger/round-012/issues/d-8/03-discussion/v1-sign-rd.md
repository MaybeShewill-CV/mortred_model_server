# 第3步签字

- 问题: D-8
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `docker build` 的默认目标就是 Dockerfile 最后阶段 `mortred-gpu`，该阶段只有 `FROM runtime`。`runtime` 与 `mortred-gpu` 都没有 `USER`，未指定用户时是 uid 0。`conf/mortred.toml` 的 `log_dir` 是相对路径 `logs`，容器项目根是 `/opt/mortred`，supervisor 启动时对这个目录做 `create_directories` 并忽略错误码。只在 `mortred-gpu` 里以 root 创建 uid 与 gid 均为 `10001` 的用户 `mortred`，把 `/opt/mortred/logs` 交给该用户，必要时只改 `/opt/mortred` 已有文件的所有者，再把该阶段默认用户设为这个账号，即可一次完成。入口脚本只 `exec` 监督进程，仓库里没有 uid 判断，8080 和 8787 都高于 1024。验收命令用 `--entrypoint id`，镜像名后的 `-u` 是 `id` 的参数，读到的就是镜像默认用户，退出码 0，标准输出一行 `10001`，标准错误为空。
