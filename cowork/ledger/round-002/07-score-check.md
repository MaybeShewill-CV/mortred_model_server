# 评分复核

- 轮次: 2
- 原始分: 8.9
- 加权分: 8.9
- 上限规则: 无
- 相对本轮起点: 第 2 轮起点 8.9。关闭 P2-14 不重算模块分。
- 终态: 继续
- 开放项: 相对本轮起点减少
- 阻塞: 无
- 与角色自评: 一致

原始分仍是 (15×9.0 + 20×9.4 + 15×9.0 + 10×9.2 + 5×9.0 + 15×8.8 + 10×8.8 + 5×7.4 + 5×8.0) / 100 = 8.9。没有未关闭且未阻塞的 P0 或 P1，不套上限。P2-14 已从开放项删除。仍开放 P2-16、P2-17、P2-19、P2-20、P2-21。完成标准 27 行都是已满足，开放项不是空，终态不是结束。

## 完成标准

| 编号 | 状态: 已满足 / 未满足 / 阻塞 | 证据或队列 id |
| --- | --- | --- |
| PF1 | 已满足 | `docs/perf/coverage_set_http_rps.md` 九行都有硬件、日期、pack 的 `worker_nums` 与 `decode_auto`、`--raw` 命令、成功 RPS、p50、p99、错误率 0，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和比值。`-c` 等于该行 `worker_nums` 的两倍 |
| PF2 | 已满足 | 同一份清单九行的排队计数与推理计数分开，模型标签是该 id 的 `server_uri` |
| PF3 | 已满足 | 九份覆盖集 server toml 的 `max_batch_size=1`。`on_sync_item_done` 对每张执行完的图记 1。`test/server_e2e_contract_test.cc` 的 `batch_size_one_counts_each_executed_image` 与 `multi_image_failure_is_isolated_per_item` |
| PF4 | 已满足 | 同一份清单九行都有 gateway 五段 p50 和 `server_time_ms_p50`，五段之和大于模型 p50 的一半，并写了 `dominant_stage=upstream`。`scripts/coverage_http_report.py` 用五段直方图 p50 写出 `dominant_stage`。P2-19 是超过 4096 次后 `/metrics` 不再写 `mortred_gateway_stage_dominant`，不把这一行改成未满足 |
| PF5 | 已满足 | 已有记录值在 `docs/perf/coverage_set_http_rps.md`。没有看到同命令复测的成功 RPS 低于该值 85%，或 p99 高于该值 150% |
| AV1 | 已满足 | 覆盖集九个产品 toml 是 `type=tensorrt`。cpu 侧 `mobilenetv2_cpu.toml` 是 `type=onnx`，不把缺少 HTTP RPS 写成未满足。这次没有把未跑的拉起写成故障 |
| ST1 | 已满足 | 未知字段 422，空图不成功，错误 token 401，失败项 `data` 为 null。`test/server_e2e_contract_test.cc`、`test/request_envelope_unittest.cc` |
| ST2 | 已满足 | `wrong_token_preserves_full_burst` 与 `unauthenticated_stays_401_under_rate_limit`。401 在转发前退回准入 |
| ST3 | 已满足 | `queue_limit_returns_429_with_retry_after`。拒绝不把占着槽的请求打成失败 |
| ST4 | 已满足 | `dead_upstream_maps_to_503`。停一个 id 不改其他 child 的 `wanted` |
| ST5 | 已满足 | 环境块在 `fork` 之前写入。子进程路径是 `dup2`、`close`、`syscall`、`getppid`、`sigaction`、`sigprocmask`、`chdir`、`execve`。`nvidia-smi` 使用绝对路径 `execve` |
| ST6 | 已满足 | `nvidia-smi` 无输出或利用率为 N/A 时 `available=false`，日志写明 offline |
| TP1 | 已满足 | 九次官方压测都带 `--raw`，时长不少于 60 秒，错误率 0，429 单独为 0。并发是该行已部署 `worker_nums` 的两倍。引擎保持 static shape、batch=1 |
| TP2 | 已满足 | `cowork/ledger/round-001/issues/p2-4/05-qa.md` 第 3、5、6 条：`yolov8s` 与 `libface` 同时 ready，两条请求时间重叠，两边 HTTP 200 且根上 `status` 为 0。记录文件在 gitignore 的 `logs/`，验收正文在这份已提交的测试报告里 |
| TP3 | 已满足 | `scripts/mortredctl_doctor.sh` 与 `scripts/pack_occupancy.py`：缺 engine、缺占用戳，或占用门闩关闭时非 0 退出，并打印下一条命令 |
| UX1 | 已满足 | 中英文 README 与两份 deployment 的构建命令与 `CMakePresets.json` 的 `full` / `full-cpu` 一致 |
| UX2 | 已满足 | 第一小时从 `mortredctl next` 走到 `doctor --strict`。`scripts/mortredctl_next.sh` 最后一条是该命令 |
| UX3 | 已满足 | `quality-gates` 与 `cpu-profile` 检查 OpenAPI 与仓库一致。GPU 作业被跳过时 `inference-paths` 打印 `NOT_A_TRT_PROOF` 并返回 2。P2-20 是成功路径把四个 ONNX 用例写成 `TRT_PROOF=yes`，不把跳过路径改成未满足 |
| UX4 | 已满足 | `docs/openapi.json` 为覆盖集九个 id 各有请求和响应示例。`img_data` 的 422 与迁移说明在 `docs/api-contract.md` 和 `legacy_img_data_returns_422_with_migration_hint` |
| OP1 | 已满足 | `mortredctl prepare --id` 经 `scripts/prepare_pack.sh` 为覆盖集每个 id 生成 static batch=1 engine |
| OP2 | 已满足 | `/api/v1/status` 按 id 返回 `state`，有退出记录时写入 `last_exit_status` |
| OP3 | 已满足 | `restart_server` 只停再起这一个 id |
| OP4 | 已满足 | `tokens_usable_for_listen` 在推理、管理、metrics token 缺失或彼此相同时拒绝监听 |
| OP5 | 已满足 | `perform_ordered_shutdown` 先向模型发 SIGINT 并等待，再停 gateway。`deploy/mortred-supervisor.service` 为 `KillMode=process` |
| EX1 | 已满足 | `docs/how_to_add_new_server.md` 与 `docs/model-developer-guide.md` 的示例 id 是 `densenet`，不要求每个模型一个服务器二进制。中文示例是 P2-16，不在 EX1 点名的两份英文文件里 |
| EX2 | 已满足 | `project_http` 从 `catalog()` 投影。gateway 按 id 查找转发，没有按模型写死的路由 |
| EX3 | 已满足 | 覆盖集九个产品 toml 均为 `type=tensorrt`。这次没有观察到 cpu profile 上 `type=tensorrt` 进入 ready |
