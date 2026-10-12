# 第2步确认

- 轮次: 1
- 针对版本: v3
- 结论: 不同意
- 理由: P2-3 交付不了。通过条件确认不了 ST5。`src/control/gpu_sampler.h` 与 `src/control/occupancy_gate.h` 在 `fork` 之后调用 `execvp`。库函数跟踪里这次调用就是随后那次 exec，落在「fork 返回之后、exec 之前」的窗口外面。只去掉 `src/control/supervisor.cpp` 里的 `prctl` 后，该窗口里的库函数名可以没有 `prctl`、`execvp`、`setenv`，两处探测仍调用 `execvp`。附录把「改为带绝对路径的 execve」排除在通过条件之外。ST5 关不掉。
