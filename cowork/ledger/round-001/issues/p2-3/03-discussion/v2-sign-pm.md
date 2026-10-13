# 第3步确认

- 问题: P2-3
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 4 已编号，同一次启动就能判定。标准 1、3 核对三份现有配置，以及 `GET /api/v1/servers/libface` 的 `server.state=running`、`server.pid>0` 且进程仍在。标准 2 用同一次库函数跟踪：supervisor 的 `func=init` 且 `watchfd>=3` 证明挂钩在 fork 之前；libface 这条直接子进程必须有 `func=execve`、arg0 以 `/` 开头、`inh=closed` 的终点，表示该行是关掉 fd 3 及以上之后追加的；这行之前没有 `prctl`、`execvp`、`setenv`。没有这行终点，或该 pid 只出现在 exec 之后，不通过。标准 4 在 `gpu sampler online (` 或 `gpu sampler offline (utilization N/A)` 时，对另一个直接子进程用同一段判定，不读取路径是否以 `/nvidia-smi` 结尾；出现 `gpu sampler offline (no nvidia-smi / no GPU)` 时不要求探测子进程。自测脚本按这四条给出退出码，对照源码不能代替这次跟踪。这覆盖锁定句，也覆盖上一版写进正文的两条条件。
