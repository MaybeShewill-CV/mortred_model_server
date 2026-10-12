# 第2步确认

- 轮次: 1
- 针对版本: v2
- 结论: 不同意
- 理由: P2-3 仍标着 ST5，通过条件确认不了这条完成标准。模型子进程在 `fork` 之后、`execve` 之前调用 `prctl`；`src/control/gpu_sampler.h` 与 `src/control/occupancy_gate.h` 的 `nvidia-smi` 探测在 `fork` 之后调用 `execvp`。v2 把通过条件收成一次系统调用跟踪，并写明对照这三处源码、以及把 `nvidia-smi` 改成带绝对路径的 `execve`，都不算通过。`execvp` 与 `setenv` 不是系统调用。跟踪里这次探测只会显示成随后的 `execve`，落在「exec 之前」这段窗口外面。只去掉 `prctl` 时，跟踪里仍然可以没有 `prctl`、`execvp`、`setenv` 这三个名字，两处探测却仍在 `fork` 之后调用 `execvp`。ST5 因此关不掉。
