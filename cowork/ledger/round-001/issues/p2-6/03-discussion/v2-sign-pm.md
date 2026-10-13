# 第3步确认

- 问题: P2-6
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 7 已编号。同一次先跑交付后的 `python3 scripts/stop_order_record.py --check`，再跑不带 `--check` 的那条，只打开 `deploy/mortred-supervisor.service` 和这一份 `logs/stop_order_yolov8s_libface.json`。标准 1 看这次标准输出的七行、退出码 0、该 json 在命令前后都不存在，以及单元里 `[Service]` 在 `[Install]` 之前只有一行 `KillMode=process`、`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 保持原文、没有 `ExecStop=`。后四行就是这次标准输出里的四句固定文字。`--check` 退出码 0 单独不表示驻留进程已经按顺序退出。标准 5 只看这份 json：`signal` 为 `SIGTERM`，`signal_pid` 与 `supervisor_pid` 是同一个正整数，三处 `ppid` 都等于它，全文没有 `systemctl`，不再另取进程号。标准 7 只看 `gateway_gone` 与 `after_supervisor`：两个模型布尔值为 false，后者四个布尔值都是 false 且 `sample_index` 更大；这四个布尔值表示标准 3 记下的四个 pid，对象里不写 pid；没有这两次之间的采样，按本条通过。标准 2、3、4、6 仍是这一份 json：退出码 0 且不出现三个 token；`before` 里 `yolov8s`、`libface` 与键 `gateway`（`id` 为 `__gateway`）的 `ready`、`state`、`pid`；pack 只有这两节；`models_gone` 时两个模型已不在而 gateway 与 supervisor 仍在，随后 `gateway_gone` 的序号更大。这覆盖锁定句里模型先退出、gateway 随后退出、gateway 退出后模型进程不再存在。现有 `GET /api/v1/status`、有序退出、token 和每个 worker 的 TensorRT context 不在这七条的通过条件里。上一版要写入正文的三句已经落在标准 1、5、7。
