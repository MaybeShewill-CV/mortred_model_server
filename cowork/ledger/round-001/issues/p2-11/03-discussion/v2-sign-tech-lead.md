# 第3步确认

- 问题: P2-11
- 方案版本: v2
- 结论: 不同意
- 条件: `mortred_workers_available` 与 `mortred_workers_busy` 只在推理完成路径里写入。进程刚 ready、还没有完成一次推理时，两只 gauge 仍是初值 0，相加不等于 `worker_nums`。进程中的 worker 数必须取官方压测已经产生完成响应、模型进程仍在时的刮取；压测前那一次刮取只用来做直方图增量，不能当作 worker 数。
- 理由: v1 的条件已经写进本版。`run_one_point` 改传 `server_listen` 的 `uri`，九个 `server_uri` 与 `conf/server` 一致，`_metric_series` 的 `{model="...",le="..."}` 和 `{model="..."}` 对得上已渲染的排队直方图与推理直方图，`set_model(_m_server_uri)` 保持不动。四步里的第 3 步把 worker gauge 相加放在第 4 步官方压测之前。`/metrics` 只渲染当前值，`/ready` 不更新这两只 gauge，所以这次相加在正确部署上仍是 0。环境变量 `MORTRED_WORKER_NUMS` 与 `MORTRED_DECODE_AUTO` 在拉起后即可读，不受这一处影响。
