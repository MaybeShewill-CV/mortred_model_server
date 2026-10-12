# 第2步确认

- 轮次: 1
- 针对版本: v3
- 结论: 不同意
- 理由: P2-3 的可观察行为是这一次启动的库函数跟踪：`fork` 返回之后、随后的 `exec` 之前，没有 `prctl`、`execvp`、`setenv`。`gpu_sampler.h` 与 `occupancy_gate.h` 的 `nvidia-smi` 探测在 `fork` 之后调用的 `execvp` 就是随后那次 `exec`，落在这段窗口外面。只去掉模型子进程里的 `prctl` 时，这一次跟踪已经符合写出的通过条件，确认不了这两处探测是否还在调用 `execvp`。后一句只允许带绝对路径的 `execve` 作为这次 `exec` 出现，没有把「随后那次 `exec` 必须是它」写成通过条件。
