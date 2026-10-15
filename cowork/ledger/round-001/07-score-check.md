# 评分复核

- 轮次: 1
- 原始分: 7.1
- 加权分: 7.1
- 上限规则: 无
- 相对本轮起点: 无变化
- 终态: 结束
- 开放项: 相对本轮起点减少
- 阻塞: 无
- 与角色自评: 一致

原始分 = (15×6.0 + 20×7.3 + 15×7.6 + 10×9.0 + 5×6.0 + 15×6.5 + 10×6.8 + 5×6.5 + 5×9.0) / 100 = 713.0 / 100 = 7.1。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11 已关闭。未关闭且未阻塞的开放项为无，阻塞项为空。不再套 7.9 或 8.9。模块分数沿用 `01-modules` 正文，与 `07-close-p211-2x.md` 一致。test 正文仍是 6.8。

本轮起点未关闭且未阻塞的 id 共 13 条。关闭后为无。当前评分标准要求官方压测并发等于已部署 `worker_nums` 的两倍。已合入 `9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32` 的 `docs/perf/coverage_set_http_rps.md` 九行 `-c` 都是该行 pack `worker_nums` 的两倍。

## 完成标准
| 编号 | 状态: 已满足 / 未满足 / 阻塞 | 证据或队列 id |
| --- | --- | --- |
| PF1 | 已满足 | P2-11 已合入 `9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32`。九行都有硬件、日期、pack 的 `worker_nums` 与 `decode_auto`、`--raw` 命令、成功 RPS、p50、p99、错误率 0，以及对应 `docs/benchmark-h3/<id>.txt` 的 Throughput 和比值。`-c` 等于该行 `worker_nums` 的两倍 |
| PF2 | 已满足 | 同一张清单每行都有分开的 `mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` count 增量 |
| PF3 | 已满足 | P2-1 已合入 `d5c9dffc522802a7e29292c7d8bd90f71fcaa837`。新拉起的 `yolov8s` 提交 2 张合法图片后 `results` 长度为 2；`/metrics` 里 `mortred_batch_size` 的样本都是 1，count 与 sum 都是 2，`le="1"` 的桶等于该 count。重跑打印 `p2-1-ok`，退出码 0 |
| PF4 | 已满足 | 同一张清单每行都有 gateway 五段和 `server_time_ms_p50`。五段之和大于模型 p50 的一半，`dominant_stage=upstream`。P1-1 已合入 `6803520d9dbf526170cb8c70e77d4b3e33834e01`，不创建 `/tmp/mortred_stage_timing` 时 gateway `/metrics` 能读到五段 |
| PF5 | 已满足 | 已有按当前性能约束完成的 HTTP RPS 记录。没有一次同命令复测低于已记录成功 RPS 的 85%，或 p99 高于已记录值的 150% |
| AV1 | 已满足 | `conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml` 为 `type=onnx`。缺少 HTTP RPS 不使本行未满足。本次没有观察到该配置拉起失败，或失败被报成成功 |
| ST1 | 已满足 | `test/server_e2e_contract_test.cc`：未知字段与 `img_data` 为 422，模型失败时 `results[].data` 为 null |
| ST2 | 已满足 | P2-2 已合入 `b44f9cb2afd06269f446c9e4dfe0e04a202dcbd0`。`enabled=true` 与 `MORTRED_RATE_LIMIT=enforce` 下，同一 IP 先 3 次错误 token 为 401，紧接着 3 次合法推理为 200，再一次才是 429。重跑退出码 0 |
| ST3 | 已满足 | `test/server_e2e_contract_test.cc` 的 `queue_limit_returns_429_with_retry_after`：队列打满返回 429 和 `Retry-After`，占槽请求仍成功 |
| ST4 | 已满足 | `test/gateway_e2e_test.cc` 的 `dead_upstream_maps_to_503`：上游拒绝连接时 gateway 返回 503 |
| ST5 | 已满足 | P2-3 已合入 `6031d5fdbe66b217762f25ba51f9a9027cfb8e6e`。同一次启动里，libface 直接子进程在关掉 fd 3 及以上之后写下绝对路径 `execve`，`inh=closed`，终点前没有 `prctl`、`execvp`、`setenv`。进程 `state=running`。采样日志为 `gpu sampler online`。重跑退出码 0 |
| ST6 | 已满足 | `src/control/gpu_sampler.h`：`nvidia-smi` 失败或 utilization 为 N/A 时 `available` 为 false，利用率记为 -1，日志写 offline |
| TP1 | 已满足 | 九次官方压测都带 `--raw`，时长不少于 30 秒，错误率是 0，429 单独为 0。并发数是该行已部署 `worker_nums` 的两倍。引擎保持 static shape、batch=1 |
| TP2 | 已满足 | P2-4 已合入 `eddcc454a20293ccbfbef0717cb6453032109cb5`。同一份记录里 `yolov8s` 与 `libface` 同时 ready，两条请求时间重叠且都成功。重跑 `--check`、实跑与判定脚本退出码都是 0 |
| TP3 | 已满足 | `scripts/mortredctl_doctor.sh` 与 `scripts/pack_occupancy.py`：缺 engine、缺占用戳或占用门闩关闭时非 0 退出，并打印下一条命令 |
| UX1 | 已满足 | `README.md`、`README.zh-cn.md`、`docs/deployment.md` 与 `CMakePresets.json` 的 `full` / `full-cpu` 命令一致 |
| UX2 | 已满足 | `README.md` 与 `docs/deployment.md` 从 `mortredctl next` 走到 `doctor --strict`；`scripts/mortredctl_next.sh` 的最后一条是该命令 |
| UX3 | 已满足 | `.github/workflows/ci.yml` 运行 `gen_openapi.py --check`；GPU 作业跳过时 `ci_eval_inference_paths.py` 返回 2 并打印 `NOT_A_TRT_PROOF` |
| UX4 | 已满足 | `docs/openapi.json` 含覆盖集九个 id 的请求和响应示例；`test/server_e2e_contract_test.cc` 对 `img_data` 返回 422 并指向改用 `images` |
| OP1 | 已满足 | `scripts/mortredctl_prepare.sh` 接受 `prepare --id`，`scripts/prepare_pack.sh` 按产品清单转换该 id 的 static batch=1 engine |
| OP2 | 已满足 | `src/control/supervisor/supervisor_app.cpp` 的状态接口按 id 返回 `state`，并在有退出信息时写入 `last_exit_status` |
| OP3 | 已满足 | `restart_server` 只停再起这一个 id |
| OP4 | 已满足 | P2-5 已合入 `525dd8353a3c5056a209d8a6c60e5b0d13cad96d`。推理 token 缺失、管理 token 缺失，或二者相同而 metrics token 不同时，gateway 退出码 1 且不监听。三个彼此不同的短 token 可以监听，`GET /healthz` 为 200，正文为 `ok`。重跑白盒与外部脚本退出码都是 0 |
| OP5 | 已满足 | P2-6 已合入 `1175a0cab213499034f33b3af376f6b108fadc68`。一次外部跟踪里，supervisor 先向 yolov8s 和 libface 发 SIGINT 并等到它们退出，再向 gateway 发 SIGINT。gateway 退出后这两个模型进程不再存在。重跑 `--check` 与实跑退出码都是 0 |
| EX1 | 已满足 | P1-2 已合入 `259c11a006826f6a65b6ad24caded3c6c6a12d1d`。`docs/how_to_add_new_server.md` 示例 id、family 与 `model =` 为 `densenet`，并写明不为每个模型新增服务器二进制。`README.zh-cn.md` 的 `--model` 为 `mobilenetv2`。两份 README 的 cpu 精选为 `resnet`。`docs/model-developer-guide.md` 使用 `catalog()` 与 `mortred-model-server.out --list`，没有要求单独的服务器二进制 |
| EX2 | 已满足 | `src/apps/product_index.cpp` 从 `catalog()` 投影；`src/control/gateway/gateway_app.cpp` 按 id 查找转发，没有按模型写死的路由 |
| EX3 | 已满足 | 覆盖集九个产品 toml 均为 `type=tensorrt`。本次没有观察到 cpu profile 上 `type=tensorrt` 进入 ready |
