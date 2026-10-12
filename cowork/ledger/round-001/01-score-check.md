# 评分复核

- 轮次: 1
- 原始分: 7.1
- 加权分: 7.1
- 上限规则: P0 上限 7.9
- 相对本轮起点: 无
- 终态: 继续
- 开放项: 没有减少
- 阻塞: 无
- 与角色自评: 一致

原始分 = (15×6.0 + 20×7.3 + 15×7.6 + 10×9.0 + 5×6.0 + 15×6.5 + 10×6.8 + 5×6.5 + 5×9.0) / 100 = 713.0 / 100 = 7.1。存在未关闭且未阻塞的 P0-1，上限 7.9。7.1 低于上限，加权分保持 7.1。test 模块返回摘要曾写 6.4，记录正文是 6.8，复核采用正文 6.8。九份「未覆盖」都是「无」。分数高于 8.0 的 src-factory-common 与 ci 没有未覆盖路径。每条问题都有出处、现在的行为和一次外部验收即可确认的变化。

## 完成标准
| 编号 | 状态: 已满足 / 未满足 / 阻塞 | 证据或队列 id |
| --- | --- | --- |
| PF1 | 未满足 | P1-1 |
| PF2 | 未满足 | P1-1 |
| PF3 | 未满足 | P2-1 |
| PF4 | 未满足 | P1-1 |
| PF5 | 未满足 | P1-1 |
| AV1 | 已满足 | `conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml` 为 `type=onnx`。缺少 HTTP RPS 不使本行未满足。本次没有观察到该配置拉起失败，或失败被报成成功 |
| ST1 | 已满足 | `test/server_e2e_contract_test.cc`：未知字段与 `img_data` 为 422，模型失败时 `results[].data` 为 null |
| ST2 | 未满足 | P2-2 |
| ST3 | 已满足 | `test/server_e2e_contract_test.cc` 的 `queue_limit_returns_429_with_retry_after`：队列打满返回 429 和 `Retry-After`，占槽请求仍成功 |
| ST4 | 已满足 | `test/gateway_e2e_test.cc` 的 `dead_upstream_maps_to_503`：上游拒绝连接时 gateway 返回 503 |
| ST5 | 未满足 | P2-3 |
| ST6 | 已满足 | `src/control/gpu_sampler.h`：`nvidia-smi` 失败或 utilization 为 N/A 时 `available` 为 false，利用率记为 -1，日志写 offline |
| TP1 | 未满足 | P1-1 |
| TP2 | 未满足 | P2-4 |
| TP3 | 已满足 | `scripts/mortredctl_doctor.sh` 与 `scripts/pack_occupancy.py`：缺 engine、缺占用戳或占用门闩关闭时非 0 退出，并打印下一条命令 |
| UX1 | 已满足 | `README.md`、`README.zh-cn.md`、`docs/deployment.md` 与 `CMakePresets.json` 的 `full` / `full-cpu` 命令一致 |
| UX2 | 已满足 | `README.md` 与 `docs/deployment.md` 从 `mortredctl next` 走到 `doctor --strict`；`scripts/mortredctl_next.sh` 的最后一条是该命令 |
| UX3 | 已满足 | `.github/workflows/ci.yml` 运行 `gen_openapi.py --check`；GPU 作业跳过时 `ci_eval_inference_paths.py` 返回 2 并打印 `NOT_A_TRT_PROOF` |
| UX4 | 已满足 | `docs/openapi.json` 含覆盖集九个 id 的请求和响应示例；`test/server_e2e_contract_test.cc` 对 `img_data` 返回 422 并指向改用 `images` |
| OP1 | 已满足 | `scripts/mortredctl_prepare.sh` 接受 `prepare --id`，`scripts/prepare_pack.sh` 按产品清单转换该 id 的 static batch=1 engine |
| OP2 | 已满足 | `src/control/supervisor/supervisor_app.cpp` 的状态接口按 id 返回 `state`，并在有退出信息时写入 `last_exit_status` |
| OP3 | 已满足 | `restart_server` 只停再起这一个 id |
| OP4 | 未满足 | P2-5 |
| OP5 | 未满足 | P2-6 |
| EX1 | 未满足 | P1-2 |
| EX2 | 已满足 | `src/apps/product_index.cpp` 从 `catalog()` 投影；`src/control/gateway/gateway_app.cpp` 按 id 查找转发，没有按模型写死的路由 |
| EX3 | 已满足 | 覆盖集九个产品 toml 均为 `type=tensorrt`。本次没有观察到 cpu profile 上 `type=tensorrt` 进入 ready |
