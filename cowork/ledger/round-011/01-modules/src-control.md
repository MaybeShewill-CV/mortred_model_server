# 模块评分

- id: src-control
- 权重: 10
- 分数: 7.4
- 读过的路径:
  - `src/control/supervisor.h`
  - `src/control/supervisor.cpp`
  - `src/control/catalog.h`
  - `src/control/gpu_sampler.h`
  - `src/control/gateway/gateway_app.h`
  - `src/control/gateway/gateway_app.cpp`
  - `src/control/api_key_manager.h`
  - `src/control/rate_limit/gcra.h`
  - `src/control/trt_spawn_gate.h`
  - `src/control/supervisor/supervisor_app.cpp`
  - `src/control/cli/cli_app.cpp`
  - `src/control/control_config.h`
  - `src/control/control_config.cpp`
  - `conf/mortred.toml`

## 优势
- 进程树监督闭环仍在：fork/exec、PDEATHSIG、SIGCHLD reaper、backoff 与崩环放弃、就绪探测、网关先起模型后起、模型先停网关后停（`supervisor.h` 31–36、94–104 行；`supervisor.cpp` `spawn_locked`、`reaper_loop`、`autostart_all`、`perform_ordered_shutdown`）。
- Autostart 失败可定位：网关二进制缺失、start 失败或 deadline 会跳过模型波；模型波 spawn 失败与 models deadline 写入 `models_failed` 和 `reason`（`supervisor.h` 55–60、99 行；`supervisor.cpp` `mark_gateway_autostart_blocked_locked`、`mark_models_failed_locked`、`autostart_all`）。deadline 分支对仍在 starting 的已拉起 child 调用 `note_permanent_failure`（`supervisor.cpp` 1131–1138 行）。
- 空 `server_uri` 在 readiness 路径永久失败，并在模型波同步 `models_failed`（`supervisor.cpp` `fail_missing_server_uri_locked`、`probe_readiness`）。
- 拉起前有硬门禁：空 scrape token 拒启；TensorRT engine 缺失或空文件拒启，错误串含下一行 `run mortredctl prepare (scripts/prepare_pack.sh) on this GPU`（`trt_spawn_gate.h` 26–31 行；`supervisor.cpp` 261–264 行）。
- Catalog 契约清楚：port 与 `server_uri` 必填，重复 id、port、uri 失败，cpu/gpu profile 过滤后再查重（`catalog.h` 37–57 行）。
- 推理路由未鉴权不会转发。网关无静态 token 且没有可用 API key 时拒绝监听；监督面无 `MORTRED_API_TOKEN` 时拒绝启动（`gateway_app.cpp` 553–557、664–683 行；`supervisor_app.cpp` 765–771 行）。`/healthz` 在鉴权之前返回。

## 劣势
- supervisor 在 `fork` 之后、`exec` 之前调用 `setenv` 和其他非 async-signal-safe 函数。`spawn_locked` 于 `supervisor.cpp` 364 行 `fork`，子进程从 375 行到 `execl`（433、436、439 行）之间调用 `setenv`（397–412、417–426、435 行），同段还有 `close_all_fds_except_stdio` 里的 `opendir`（42–56、382 行）、`signal`（389–390 行）和 `std::to_string`（403 行）。此时 reaper、signal、monitor 线程已经由 `start_threads`（208–210 行）启动。现在的行为：多线程监督进程每次拉起网关或模型，子进程都在 exec 前改环境并可能分配内存。改完后一次可判定的变化：同一段子进程在 exec 前不再出现这些调用，环境改由 `exec` 的环境块带入；模型仍监听 `127.0.0.1`，并带上 `MORTRED_AUTH_TOKEN` 或网关的 `MORTRED_METRICS_TOKEN`。
- 推理请求在鉴权失败时仍消耗限流配额。`GatewayApp::process` 在路由和鉴权之前调用 `check_ip_rate_limit`（`gateway_app.cpp` 399–402、496–498、509–557 行），`admit` 在 432 行。`ip_limiter_` 按 `rate_per_sec` 和 `burst` 构造（615–618 行）；`GcraPolicy::enabled()` 只看这两个数是否大于 0（`gcra.h` 63–67、169–170 行）。`rate_limit_enabled_` 只用于启动日志（619–624 行）。仓库配置是 `enabled = false`、`rate_per_sec = 100000`、`burst = 200000`（`conf/mortred.toml` 20–24 行），策略仍然计数。已识别但没有 inference/admin scope 的 key 会在 `allow_rate_limit` 加一之后才因 scope 得到 401（`api_key_manager.h` 219–239 行；`gateway_app.cpp` 518–557 行）。现在的行为：把该节改成 `rate_per_sec = 1`、`burst = 1` 且保持 `enabled = false`，同一来源先对推理路由发无 Bearer 请求得到 401，同一秒内再带正确 token 得到 429；`rate_limit_qps = 1` 且 scope 不是 inference/admin 的 key，第一次推理是 401，第二次是 429。改完后：上述 401 不再占配额，紧接着的合法推理进入转发；`enabled = false` 时这道门不计数也不回 429；无 inference 权限的 key 连续两次都是 401。
- `nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率，日志和状态不标明 offline。`parse_i32_or` 把 `[N/A]` 和 `N/A` 收成 -1（`gpu_sampler.h` 55–57、181 行）。CSV 非空时 `available_` 仍设为 true，sample 进入 ring，并打印 `gpu sampler online`（201–214 行）。`/api/v1/gpu` 原样输出 `available: true` 和 `util`（`supervisor_app.cpp` 379、387 行）。`mortredctl ps` 在 `samples` 非空时用绿色打印 `util -1%`，不读 `available`（`cli_app.cpp` 238–251 行）。现在的行为：utilization 为 N/A 的一次成功探测后，`GET /api/v1/gpu` 为 `available: true` 且样本 `util` 为 -1，stderr 是 online。改完后：同一次探测的 `available` 为 false，或日志/状态写出 offline，CLI 不再把 -1 印成利用率百分比。

## 不计毛病
- 一个 catalog id 一个 OS 进程（`catalog.h` 的 `ServerEntry`；`supervisor.cpp` `set_catalog` 与 `spawn_locked` 一 id 一 child、一次 fork）。
- 停服先 SIGINT，宽限后再 SIGKILL（`supervisor.cpp` `perform_ordered_shutdown` 855–861、879–883 行）。

## 依据
- S2、S4、S6 仍在上述行号，三条都写进劣势。未鉴权调用推理路由这一半不成立：401 发生在 `forward_to_model` 之前。`nvidia-smi` 整次失败且输出为空时，`available_` 为 false，并打印一次 `gpu sampler offline`（`gpu_sampler.h` 216–223 行）；未满足的是 N/A 利用率仍进入在线采样。
- H4 在控制面的部分：缺 engine 或空文件时 spawn 返回失败，错误含 `mortredctl prepare`（`trt_spawn_gate.h` 26–31 行，`supervisor.cpp` 261–264 行）。`mortredctl doctor` 与 `prepare` 把脚本退出码原样返回（`cli_app.cpp` 938–983 行）。按 id 生成 engine，以及 `doctor --strict` 在缺 engine 时的退出码，落在脚本侧，本模块不另开一句。
- 第 10 轮已关闭的原句不送回。deadline 路径已把仍在 starting 的 child 收成 failed。
- 监督、catalog、鉴权拒启和 engine 拒启足够小团队起停和定位。spawn 子进程不安全、鉴权失败仍扣配额、N/A 利用率显示为在线，所以不到 8.0，记 7.4。

## 未读
无
