# 第3步确认

- 问题: P2-6
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `deploy/mortred-supervisor.service` 现为 `Type=simple`，`ExecStart` 就是 `mortred-supervisor.out`，已有 `KillSignal=SIGTERM` 与 `TimeoutStopSec=120`，没有 `KillMode=`。补上 `KillMode=process` 后，systemd 的停止信号只打到这个主进程。`install_host_common.sh` 拷贝单元时只替换 `/opt/mortred` 路径，cpu profile 只在 `[Service]` 后插入 `MORTRED_PROFILE`，不会改掉这一行。主进程收到 SIGTERM 后，现有 `signal_loop()` 与 SIGINT 走同一条 `request_shutdown()`，`perform_ordered_shutdown()` 先对已有 pid 的模型发 SIGINT 并等待（5000 毫秒后再 SIGKILL，随后再等 2000 毫秒），全部模型等待结束后才对 gateway 发 SIGINT 并等待；`_shutdown_done` 在这之后才置位，主进程才退出。子进程是 supervisor 直接 `fork`/`execve`，ppid 就是 supervisor，模型与 gateway 都不另起守护进程。两个模型的停止不要求彼此顺序。临时 pack 经现有 `MORTRED_PACK` 只拉起 `yolov8s` 与 `libface`，`worker_nums=1` 仍是每个模型一个进程、worker 为线程并各持 execution context。状态接口已有 `gateway` 与 `servers`，`ready`、`state`、`pid` 可用；8787、8080、9051、9056 与现有 autostart 预算（gateway 15 秒、模型 60 秒）都对得上。验收脚本和 `stop_order_record_rules` 可以按已有 `resident_pair_record_rules` 的方式加上，不必改有序退出、token 判定或 TensorRT context 归属。
