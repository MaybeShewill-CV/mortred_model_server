# 第2步确认

- 轮次: 1
- 针对版本: v3
- 结论: 不同意
- 理由: P2-3 仍标着 ST5，通过条件确认不了这条完成标准。模型子进程在 `fork` 之后、`execve` 之前调用 `prctl`；`src/control/gpu_sampler.h` 与 `src/control/occupancy_gate.h` 的 `nvidia-smi` 探测在 `fork` 之后调用 `execvp`。v3 把通过条件收成一次库函数跟踪，检查的是 `fork` 返回之后、随后那次 `exec` 之前的库函数名，并写明带绝对路径的 `execve` 作为这次 `exec` 本身出现不算失败。`execvp` 就是这两处探测随后的那次 `exec`，落在窗口外面。只去掉 `prctl` 时，窗口里仍然可以没有 `prctl`、`execvp`、`setenv`，两处探测却仍在 `fork` 之后调用 `execvp`。ST5 因此关不掉。
