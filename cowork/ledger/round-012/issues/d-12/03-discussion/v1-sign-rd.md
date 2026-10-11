# 第3步签字

- 问题: D-12
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `spawn_locked` 已在 `fork` 前备好可执行路径、token、管道和 `bin_dir`。现在落在 `fork` 与 `execl` 之间的是 `setenv`、`std::to_string`、`signal`，以及 `close_all_fds_except_stdio` 里的 `opendir`。这些都可以留在父进程：`fork` 前用 `getrlimit` 得到 `max_fd`，从当时的 `environ` 复制并按现有条件覆盖环境块，组好三套 `argv` 指针。子进程只做 `dup2`、`close`、`prctl`、`sigaction`、`sigprocmask`、`chdir`、`execve`、`_exit`。`fork` 复制地址空间，这些指针在子进程里仍然有效，监督进程自己的环境不必 `setenv`。`MORTRED_LISTEN_HOST=127.0.0.1` 和内部 token 写入 `MORTRED_AUTH_TOKEN` 的行为不变。验收命令先扫 `if (pid == 0)` 窗口，再在临时端口拉起 id 为 `probe` 的进程。无 pack 时占用门直接放过，这份 toml 没有模型配置，TensorRT 门也放过。一次可以做完。
