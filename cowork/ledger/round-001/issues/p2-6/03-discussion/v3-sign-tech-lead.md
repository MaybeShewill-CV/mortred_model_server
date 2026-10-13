# 第3步确认

- 问题: P2-6
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1、5、7 沿用已写入正文的三句，同一次外部验收可以判定。标准 1 只跑交付后的 `python3 scripts/stop_order_record.py --check`，退出码为 0，标准输出恰好七行且第七行后没有空行；`logs/stop_order_yolov8s_libface.json` 在命令前后都不存在；同一次打开的 `deploy/mortred-supervisor.service` 在 `[Install]` 之前只有一行 `KillMode=process`，等号两侧没有空格，`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 保持原文，没有 `ExecStop=`。后四行是这次标准输出里的固定文字。`--check` 退出码 0 单独不算驻留进程已经按顺序退出。标准 5 只看这份 json 里的 `signal`、`signal_pid`、`supervisor_pid` 和三份 `ppid`，全文不出现 `systemctl`，不再另取进程号。标准 7 只看 `gateway_gone` 与 `after_supervisor` 的布尔值和 `sample_index`；没有这两次之间的采样，并且这两个对象没有 pid 键，即按该条通过。`KillMode=process` 加在现有 `Type=simple` 的 `[Service]` 上，`ExecStart` 仍是 `mortred-supervisor.out`。`install_host` 的路径替换和 cpu profile 插入不改这一行。主进程收到 SIGTERM 后仍走现有 `perform_ordered_shutdown()`：先向已有 pid 的模型发 SIGINT 并等待，再停 gateway；5000 毫秒宽限和随后 2000 毫秒保持不变。不持连接时，两个模型消失到 gateway 消失约 2.2 毫秒。采样间隔定为 1 毫秒，`models_gone` 就是这一间隔上两个模型已不在、gateway 与 supervisor 仍在的那一拍；没有这一拍就是失败。实跑不向 `127.0.0.1:8080` 保持连接，也不把通过改成两个时间戳小于等于。临时 pack、端口和 75 秒 autostart 预算与已经拉起过的 `yolov8s`/`libface` 驻留方式一致。`stop_order_record_rules` 只跑 `--check`，实跑不进默认 `check`。有序退出、token 判定和每个 worker 的 TensorRT execution context 都不改。两条自测命令的工作目录和期望退出码已经写明。没有还要另加的条件。
