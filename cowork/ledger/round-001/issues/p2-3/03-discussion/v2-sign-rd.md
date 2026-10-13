# 第3步确认

- 问题: P2-3
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 锁定句要求的三处都在现有进程里：`spawn_locked` 在关掉 fd 3 及以上之后设置 `PR_SET_PDEATHSIG`，`query_nvidia_smi` 与 `probe_nvidia_smi` 在 `fork` 之后调用 `execvp`。子进程改用 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)`，父进程按 `PATH` 解析出绝对路径后再 `execve`，父进程死亡信号仍留在子进程里。v1 的跟踪缺口已补上：每次记录重新打开日志，`execve` 钩子在真正替换映像前写下终点，关 fd 之后该行仍在，并以 `inh=closed` 证明继承 fd 已是 `EBADF`。窗口切在这条绝对路径 `execve` 上，之后的调用不算进窗口；探测有输出时，另一个直接子进程用同一段规则，路径不必以 `/nvidia-smi` 结尾。自测命令使用本机 WSL 路径、`build/full` 和现有的启动、状态接口；`libface` 引擎文件已在，`/tmp` 可执行，挂起用例能跑到 300 毫秒截止。
