# 模块评分

- id: ops
- 权重: 15
- 分数: 6.5
- 读过的路径: `cowork/ledger/rubric.md`、`cowork/ledger/index.md`、`scripts/mortredctl_init.sh`、`scripts/mortredctl_prepare.sh`、`scripts/mortredctl_doctor.sh`、`scripts/mortredctl_next.sh`、`scripts/mortredctl_calibrate.sh`、`scripts/prepare_pack.sh`、`scripts/pack_occupancy.py`、`scripts/product_static_bs1.py`、`scripts/calibrate_pack.py`、`scripts/security_warn.sh`、`scripts/server/http_infer_rps.py`、`scripts/bench_set_a/http_raw.py`、`docker-compose.yml`、`deploy/`（`mortred-supervisor.service`、`nginx/mortred-edge.service`、`prometheus.compose.yml`、`docker-compose.monitoring.yml`、`prometheus.yml`、`alert-rules.yml`、`grafana-dashboard.json`、`grafana/provisioning/`、`nginx/nginx.conf.skeleton`、`nginx/snippets/`）、`src/apps/control/mortredctl.cpp`、`src/control/cli/cli_app.cpp`、`src/control/supervisor.cpp`、`src/control/supervisor/supervisor_app.cpp`、`src/control/gateway/gateway_app.cpp`、`src/control/trust_tokens.h`、`conf/packs/demo.toml`、`docs/benchmark-set-a.md`、`docs/benchmark-h3/yolov8s.txt`、`docs/perf/yolov8s_decode_backend_rps.md`

## 已具备

- 覆盖集九个 id 都在 `scripts/product_static_bs1.py` 的产品清单里，且模型 toml 为 `type = "tensorrt"`。`scripts/mortredctl_prepare.sh` 接受 `prepare --id ID`，`scripts/prepare_pack.sh` 用产品 manifest 只转换这一个 static batch=1 engine。
- `scripts/mortredctl_doctor.sh` 在 pack 缺 TensorRT engine 时打印 `next: mortredctl prepare --pack`，缺占用戳时打印 `next: mortredctl calibrate --pack … --write-pack`，然后以非 0 退出。`scripts/pack_occupancy.py` 的 `check_occupancy()` 在 `occupancy_policy=off` 或 `MORTRED_OCCUPANCY_ENFORCE=0` 时直接返回错误，错误文本里带有 `mortredctl calibrate --pack … --write-pack`。
- `scripts/mortredctl_next.sh` 按 trust、prepare、calibrate、启动 supervisor、`mortredctl doctor --strict` 的顺序每次只打印一条下一条命令。
- `mortredctl status` 读取 `/api/v1/status`。`SupervisorApp::handle_status` 为每个模型写入 `id`、`state`，并在 `has_last_exit` 时写入 `last_exit_status`。`mortredctl restart <id>` 只调用该 id 的 `restart_server()`。
- supervisor 收到 SIGINT 或 SIGTERM 后，`perform_ordered_shutdown()` 先向模型进程发 SIGINT 并等待，再向 gateway 发 SIGINT。
- `scripts/calibrate_pack.py` 的 worker 扫描和 JPEG 解码投票都用 `LoadConfig(raw=True)`，`--write-pack` 只写 pack 的 `worker_nums` 与 `decode_auto`。`scripts/server/http_infer_rps.py` 支持 `--raw`，输出成功 RPS、p50、p99，并把 HTTP 429 记为 `overload`。
- `docker-compose.yml` 用 `${MORTRED_API_TOKEN:?…}`、`${MORTRED_GATEWAY_AUTH_TOKEN:?…}`、`${MORTRED_METRICS_TOKEN:?…}` 要求三个变量已设置，宿主端口绑在 `127.0.0.1`。

## 问题

1. 覆盖集没有一份按本次性能约束完成的 HTTP 报告（PF1、TP1、PF5；同一次压测因此也没有 PF2、PF4 所要的读数）。出处：`docs/benchmark-set-a.md` 表 4、`scripts/bench_set_a/http_raw.py`、`conf/packs/`、`docs/benchmark-h3/yolov8s.txt`。现在的行为：九个 `type=tensorrt` 的覆盖集 id 都没有同时写明硬件、日期、pack 里的 `worker_nums` 与 `decode_auto`、`--raw` 压测命令、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值的报告。表 4 把 `yolov8s` 记成 w\* = 8、conc = 32；`http_raw.py` 把并发设为 `max(8, min(64, 4 * wstar))`，`duration_s=15.0`。`conf/packs/` 里的 pack 只有 `worker_nums = 1`，没有 `decode_auto`。表 4 引用的 `logs/bench/set_a/` 不在仓库中。改完后一次外部验收即可确认的变化：每个覆盖集 id 都有一份报告，其中 `http_infer_rps.py --raw` 的并发等于该 pack 已部署的 `worker_nums`，时长不少于 30 秒，并含上述字段；抽查 `yolov8s` 时，成功 RPS 与 `docs/benchmark-h3/yolov8s.txt` 里的 `Throughput: 400.677 qps` 的比值写在同一份报告里，且该次 `/metrics` 的 `mortred_jpeg_decode_total` 落在报告里的 `decode_auto` 上。

2. 没有 `yolov8s` 与 `libface` 同时 ready 的并发压测记录（TP2）。出处：`scripts/bench_set_a/http_raw.py`、`scripts/calibrate_pack.py` 的 `run_one_point()`、`docs/benchmark-set-a.md`。现在的行为：这两处脚本每次只拉起一个模型进程；仓库里没有一份记录写明两个 id 同时 ready、对两者并发请求、且两者都有成功响应。改完后一次外部验收即可确认的变化：同一份记录里，`yolov8s` 与 `libface` 同时为 ready，并发请求两者，两者都有成功响应。

3. 推理 token 与管理 token 缺一，或二者相同而 metrics token 不同时，gateway 仍会启动（OP4）。出处：`src/control/gateway/gateway_app.cpp`（`has_static_token` 为两个静态 token 的或）、`src/control/trust_tokens.h` 的 `scrape_token_usable()`、`scripts/security_warn.sh`、`docker-compose.yml`。现在的行为：metrics token 为空，或与推理 token、管理 token 之一相同，gateway 会拒绝启动。只设置管理 token 和一份不同的 metrics token、不设置推理 token 时，`has_static_token` 为真，gateway 继续初始化。推理 token 与管理 token 相同、metrics token 不同时，`scrape_token_usable()` 返回成功，gateway 继续初始化；`security_warn.sh` 只打印警告。`docker-compose.yml` 的 `${VAR:?}` 只拒绝未设置，三个变量设成相同非空值时 compose 仍会启动容器。改完后一次外部验收即可确认的变化：上述两种环境启动 gateway 时进程以非 0 退出，日志写明拒绝原因；三个 token 都已设置且彼此不同时，gateway 可以进入监听。

4. systemd 单元没有把停止信号限制在 supervisor 主进程（OP5）。出处：`deploy/mortred-supervisor.service`、`src/control/supervisor.cpp` 的 `perform_ordered_shutdown()`。现在的行为：单元设置了 `KillSignal=SIGTERM` 和 `TimeoutStopSec=120`，没有 `KillMode=`。supervisor 进程自己收到 SIGINT 或 SIGTERM 后，会先让模型进程退出，再让 gateway 退出。单元没有把 systemd 的停止信号收成只发给主进程。改完后一次外部验收即可确认的变化：对已安装的 `mortred-supervisor.service` 执行停止后，驻留集模型进程先退出，gateway 随后退出，gateway 退出后这些模型进程不再存在。

## 范围外

- Docker 镜像与默认运行用户。`docker-compose.yml` 只作为宿主编排入口阅读，不把「没有镜像测试环境」写成问题。
- 多机调度、Kubernetes、覆盖集以外的 catalog id、TensorRT 动态 batch、MNN 与 ONNX Runtime 的吞吐优化。

## 依据

分数 6.5。`prepare --id`、doctor 的 engine 与占用门闩、`mortredctl status` 的逐 id 状态和 `last_exit_status`、单 id `restart`、calibrate 的 `--raw` 与 `--write-pack` 都在当前脚本和 supervisor 状态接口里。覆盖集 HTTP 报告、双模型同时压测、token 启动条件和 systemd 停止范围仍未满足完成标准，所以不到可以按本次约束给出 HTTP 吞吐并放上这台机器的程度。TP3 按 `mortredctl_doctor.sh` 与 `pack_occupancy.py` 的退出码和下一条命令视为已满足。OP1、OP2、OP3 按 `prepare --id`、`/api/v1/status` 和 `restart_server()` 视为已满足。本次没有跑压测；报告缺失按仓库里的表现成未满足，没有标成阻塞。

## 未覆盖

无
