# 模块评分

- id: ops
- 权重: 15
- 分数: 8.8
- 读过的路径: `cowork/ledger/rubric.md`、`scripts/mortredctl_init.sh`、`scripts/mortredctl_prepare.sh`、`scripts/mortredctl_doctor.sh`、`scripts/mortredctl_next.sh`、`scripts/mortredctl_calibrate.sh`、`scripts/prepare_pack.sh`、`scripts/pack_occupancy.py`、`scripts/product_static_bs1.py`（覆盖集九个 id 的产品行）、`scripts/calibrate_pack.py`（worker 扫描并发与 `--raw`）、`scripts/resident_pair_record.py`、`scripts/stop_order_record.py`、`docker-compose.yml`、`deploy/`（目录清单及 `deploy/mortred-supervisor.service`）、`src/control/cli/cli_app.cpp`、`src/control/supervisor.cpp`、`src/control/supervisor/supervisor_app.cpp`、`src/control/gateway/gateway_app.cpp`、`src/control/trust_tokens.h`、`docs/perf/coverage_set_http_rps.md`、`docs/deployment.md`（首小时与 §10）、`README.md`、`README.zh-cn.md`、`docs/benchmark-h3/` 下覆盖集九个 `<id>.txt`

## 已具备

- 覆盖集九个 id 都在 `scripts/product_static_bs1.py` 的产品表里。`mortredctl prepare --id <id>` 经 `scripts/mortredctl_prepare.sh` 进入 `scripts/prepare_pack.sh`，用产品 static batch=1 manifest 只转换这一个 engine（OP1）。
- `mortredctl status` 读取 `/api/v1/status`。`handle_status` 为每个模型写入 `id`、`state`、`ready`，并在 `has_last_exit` 时写入 `last_exit_status`（OP2）。`mortredctl restart <id>` 只对该 id 调用 `restart_server()`：先停这一个进程，再拉起这一个进程（OP3）。
- `deploy/mortred-supervisor.service` 为 `KillMode=process`、`KillSignal=SIGTERM`、`TimeoutStopSec=120`。supervisor 收到 SIGINT 或 SIGTERM 后，`perform_ordered_shutdown()` 先向模型发 SIGINT 并等待，再向 gateway 发 SIGINT（OP5）。
- `tokens_usable_for_listen()` 在推理 token 缺失、管理 token 缺失，或二者相同，或 metrics token 缺失、与前两者之一相同时拒绝监听。gateway 初始化失败则进程以非 0 退出（OP4）。
- `scripts/mortredctl_doctor.sh` 在 pack 缺 engine 时打印 `next: mortredctl prepare --pack`，缺占用戳时打印 `next: mortredctl calibrate --pack … --write-pack`，然后非 0 退出。`MORTRED_PACK` 未设置时，缺产品 engine 的每一行打印 `next: mortredctl prepare --id <id>`。`pack_occupancy.py` 在 `occupancy_policy=off` 或 `MORTRED_OCCUPANCY_ENFORCE=0` 时直接返回错误，错误文本带 `mortredctl calibrate --pack … --write-pack`。`--strict` 再把安全警告并入失败（TP3）。
- `scripts/mortredctl_next.sh` 的最后一条是 `mortredctl doctor --strict`。`README.md`、`README.zh-cn.md` 和 `docs/deployment.md` 的首小时都从 `mortredctl next` 走到 `doctor` / `doctor --strict`。这些命令在 `src/control/cli/cli_app.cpp` 里分发到对应的 `scripts/mortredctl_*.sh`（UX2）。
- `docs/perf/coverage_set_http_rps.md` 九行都是 `checked=yes`，硬件为 NVIDIA GeForce RTX 2070 SUPER、8192 MiB。每行都有 `mortredctl prepare --id`、`mortredctl calibrate --pack … --write-pack`、`calibrate_raw` 写明 worker 扫描和 JPEG 投票走 `--raw`，`conf_server_worker_nums=1`，`MORTRED_WORKER_NUMS` 与 `process_workers` 等于 pack 的 `worker_nums`，`decode_auto` 为 `cpu` 或 `gpu`。官方压测是 `scripts/server/http_infer_rps.py --raw`，`-c` 等于该行 `worker_nums` 的两倍，`wall_s` 约 60 秒，`error_rate=0.0`，`http_429=0`。九行对应关系是：`mobilenetv2` 4/8、`yolov8s` 4/8、`libface` 4/8、`pphuman_mobile` 2/4、`dbnet` 2/4、`dinov2_vits14` 2/4、`superpoint` 1/2、`realesrgan` 4/8、`metric3d_512` 1/2。每行有成功 RPS、p50、p99、`docs/benchmark-h3/<id>.txt` 的 Throughput 和比值；排队直方图与推理直方图分开计数；`jpeg_landing` 与 `decode_auto` 一致（`cpu` 走 CPU 计数，`gpu` 的 `jpeg_jpeggpu` 增加）。gateway 五段与 `server_time_ms_p50` 同在，并写了 `dominant_stage`（PF1、PF2、PF4、TP1）。`scripts/calibrate_pack.py` 的 worker 扫描并发是该点 `worker_nums` 的两倍，请求走 `raw=True`。
- `docker-compose.yml` 用三个 `${…:?}` 要求推理、管理、metrics token 已设置，宿主端口绑在 `127.0.0.1`。

## 问题

无

## 范围外

- Docker 镜像与默认运行用户。`docker-compose.yml` 只作宿主编排入口。
- 多机、Kubernetes、覆盖集以外的 catalog id、TensorRT 动态 batch、MNN 与 ONNX Runtime 的吞吐优化。
- P2-8 已在范围外。交换目录与产品 toml 的对照不进入本模块问题。
- 重复描述（已关闭，不进入问题）：P2-4，没有 `yolov8s` 与 `libface` 同时 ready 的并发成功记录（TP2）。P2-5，推理 token 或管理 token 缺失或二者相同仍启动 gateway（OP4）。P2-6，停止信号没有收成只发给 supervisor 主进程（OP5）。P2-11，覆盖集九个 TensorRT id 没有按本次性能约束完成的 calibrate 与 `--raw` HTTP 清单（PF1、PF2、PF5、TP1）。

## 依据

分数 8.8。当前文件里，换机可以用 `mortredctl prepare --id` 为覆盖集每个 id 生成 static batch=1 engine；`/api/v1/status` 能区分每个 id 的状态并带最近一次退出码；单 id 重启不带动 gateway 或其他 id；doctor 在缺 engine、缺占用戳、占用门闩关闭时非 0 退出并给出下一条命令；首小时从 `mortredctl next` 走到 `doctor --strict`，所列命令在仓库里有对应脚本。覆盖集 HTTP 清单的 `-c` 是该行已部署 `worker_nums` 的两倍，按 rubric 满足 TP1，不因此把 PF1 或 TP1 写成未满足。错误率均为 0，429 单独计数。本次没有在 WSL 里重跑 prepare、doctor 或压测；判断对的是仓库里的脚本、单元文件和这份清单，没有标成阻塞。

## 未覆盖

无
