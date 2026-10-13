# 第3步确认

- 问题: P2-6
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `deploy/mortred-supervisor.service` 是 `Type=simple`，`ExecStart` 为 `mortred-supervisor.out`，已有等号两侧无空格的 `KillSignal=SIGTERM` 与 `TimeoutStopSec=120`，没有 `KillMode=` 和 `ExecStop=`。在 `[Service]`、`[Install]` 之前加一行 `KillMode=process` 后，停止信号只打到这个主进程。`install_host_common.sh` 只替换 `/opt/mortred` 路径，cpu profile 只在 `[Service]` 后插入 `MORTRED_PROFILE`，这一行会留在拷贝结果里。主进程的 `signal_loop()` 对 SIGTERM 与 SIGINT 都调用 `request_shutdown()`，`monitor_loop()` 进入现有 `perform_ordered_shutdown()`：先对已有 pid 的模型 `kill(pid, SIGINT)` 并 `wait_child_exit`，全部模型等待返回后才对 gateway 发 SIGINT 并等待，然后才置位 `_shutdown_done`。`SupervisorApp::run()` 要在 `wait_shutdown()` 看到该标志后才返回，主进程在 gateway 退出之后才结束。子进程由 supervisor 直接 `fork`/`execve`，ppid 就是 supervisor，模型与 gateway 都不守护化。`worker_nums=1` 仍是每个模型一个进程，worker 是线程并各自持有 execution context。`GET /api/v1/status` 已把 gateway 放在键 `gateway` 且 `id` 为 `__gateway`，模型在 `servers` 里，`ready`、`state`、`pid` 都有。临时 pack 走现有 `MORTRED_PACK`，只让 `yolov8s` 与 `libface` autostart。8787、8080、9051、9056 与 autostart 默认的 gateway 15 秒、模型 60 秒都对得上。`--check` 只读这份单元文件并写出验收标准 1 的七行，不拉起进程、不创建 json；实跑用同一套单元判定，并以 50 毫秒采样写下 `models_gone`、`gateway_gone`、`after_supervisor`。同一拍里 gateway 与 supervisor 都已不在时，再取下一拍作为 `after_supervisor`，其 `sample_index` 更大。`stop_order_record_rules` 可以按 `resident_pair_record_rules` 注册。有序退出、token 判定和每个 worker 的 TensorRT execution context 都不用改。
