# 模块评分

- id: src-control
- 权重: 10
- 分数: 8.7
- 读过的路径:
  - `src/control/supervisor.h`
  - `src/control/supervisor.cpp`
  - `src/control/catalog.h`

## 优势
- 进程树监督闭环完整：fork/exec、PDEATHSIG、SIGCHLD reaper、backoff/崩环放弃、就绪探测、网关先起模型后起、模型先停网关后停（`supervisor.h` 31–36、94–104；`supervisor.cpp` `spawn_locked`/`reaper_loop`/`autostart_all`/`perform_ordered_shutdown`）。
- AutostartGate 可定位：网关二进制缺失/start 失败/deadline 跳过模型波；模型波 spawn 失败与 models deadline 写入 `models_failed`/`reason`（`supervisor.h` 55–60、99；`supervisor.cpp` `mark_gateway_autostart_blocked_locked`/`mark_models_failed_locked`/`autostart_all`）。
- 空 `server_uri` 在 readiness 路径立刻永久失败，并在模型波同步 `models_failed`（`supervisor.cpp` `fail_missing_server_uri_locked`/`probe_readiness`）。
- 拉起前硬门禁：scrape token、TRT engine、GPU occupancy（`supervisor.cpp` `spawn_locked` 252–296）。
- Catalog 契约清楚：port+server_uri 必填，重复 id/port/uri 失败，cpu/gpu profile 过滤后再查重（`catalog.h` 37–57）。
- Status/日志面够用：state/pid/ready/restart_count/last_exit/error + 按 id 的 LogBuffer（`supervisor.h` 44–53、89–92；`supervisor.cpp` `status`/`statuses`/`logs`）。

## 劣势
- occupancy 依赖策略声明的 `gpu_mem_mib` 与 spawn 时 `nvidia-smi`；声明不准会拒启或仍有共卡风险，这三份文件看不到完整 pack/校准闭环（`supervisor.cpp` 266–295）。
- models deadline 到期只写 `models_failed` 并 return，未把仍未 `running` 的子进程置为终态；gate 已失败而 child 可能仍停在 `starting`（`supervisor.cpp` 1109–1128）。

## 不计毛病
- 一个 catalog id 对应一个 OS 进程（`catalog.h` `ServerEntry`；`supervisor.cpp` `set_catalog`/`spawn_locked` 一 id 一 child、一 fork）。
- 停服先 SIGINT 宽限再 SIGKILL，属超时 drain 后外层强杀路径（`supervisor.cpp` `stop_server`/`perform_ordered_shutdown`；rubric 不计毛病）。

## 依据
- 相对早期「模型波只打 stderr / 空 uri 长期 starting」已在当前代码关闭，监督与门闩可支撑小团队日常起停与定位。
- 多模型共卡时 occupancy 声明显存仍易错；models deadline 与 child 状态可能不一致，故不到 9.0，取 8.7。

## 未读
无
