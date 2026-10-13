# 第3步确认

- 问题: P2-3
- 方案版本: v1
- 结论: 不同意
- 条件: 模型子进程仍先关掉 fd 3 及以上，随后同一次启动的跟踪日志新增一行：pid 为该 libface 子进程，ppid 为 supervisor，func=execve，arg0 以 / 开头。
- 理由: v1 用 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)` 替换子进程里的 `prctl`，并在父进程按 PATH 解析出 `nvidia-smi` 绝对路径后只调用 `execve`，这与锁定句一致，关 fd 的循环也保留。第三条自测只在 constructor 里打开一次 `/tmp/p2-3-libtrace.log`，`execve` 钩子再 `write` 这个 fd。`supervisor.cpp` 在 `execve` 之前执行 `for (int fd = 3; fd < max_fd; ++fd) close(fd)`，该 fd 已被关闭，`write` 失败。脚本只跟踪 `prctl`、`execvp`、`setenv`、`execve`，`syscall` 留不下一行，这四个函数在 close 循环之前也没有被调用，日志里没有 libface 的 pid。验收标准 2 规定没有这行就不通过。若 exec 之后的新映像再调用 `setenv`，这条 `setenv` 会落在窗口内，因为切开窗口的那次绝对路径 `execve` 没有写入。按「脚本原样执行」和「保留这段 close」写出的自测会以非 0 退出。
