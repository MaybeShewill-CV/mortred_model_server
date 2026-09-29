# 模块评分

- id: src-control
- 权重: 10
- 分数: 8.3
- 读过的路径:
  - `src/control/supervisor.h`
  - `src/control/supervisor.cpp`
  - `src/control/catalog.h`
  - `src/control/supervisor/supervisor_app.h`
  - `src/control/supervisor/supervisor_app.cpp`（`handle_status` / `autostart_gate`）

## 优势
- 进程树监督闭环完整：fork/exec、PDEATHSIG、SIGCHLD reaper、backoff/崩环放弃、就绪探测、网关先起/模型先停（`supervisor.h` 31–36、94–98；`supervisor.cpp` `spawn_locked`/`reaper_loop`/`perform_ordered_shutdown`）。
- 网关 autostart 门闩可定位：二进制缺失、start 失败或 deadline 未到 `running` 时 `skip_models` 直接 return，不拉起模型；`AutostartGate`（blocked/models_skipped/reason）写入 gateway error，并经 `/api/v1/status` 的 `autostart` 字段暴露（`supervisor.h` 55–60、92、94–98；`supervisor.cpp` 631–650、940–1015；`supervisor_app.cpp` 248–254）。
- 状态与日志面够用：`Status`（state/pid/ready/restart_count/last_exit/error）+ 按 id 的 `LogBuffer`，崩环有明确文案（`supervisor.h` 44–53、89–91；`supervisor.cpp` 233–234、582–598）。
- 拉起前硬门禁：scrape token、TRT engine、GPU occupancy（含 nvidia-smi 与兄弟进程显存）（`supervisor.cpp` 252–295）。
- Catalog 契约清楚：port+server_uri 必填，重复 id/port/uri 失败，cpu/gpu profile 过滤后再查重（`catalog.h` 37–57）。

## 劣势
- 网关 metrics token 为空仍永久失败 spawn；小团队配错时入口起不来，需懂「永久失败 / 手工再 start」（`supervisor.h` 72–74；`supervisor.cpp` 252–260、502–508）。
- 模型波 autostart 失败与 models deadline 仍主要 `fprintf(stderr)`，未并入与 gateway 同级的结构化 gate（`supervisor.cpp` 1049–1068）。
- 无 `uri` 的 child 不做 readiness probe，可能长期停在 starting，直到 models deadline（`supervisor.cpp` 656–666、1033–1061）。
- occupancy 依赖策略里声明的 `gpu_mem_mib` 与 spawn 时 nvidia-smi；声明不准会拒启或仍有共卡风险，这几份文件看不到完整 pack/校准闭环（`supervisor.cpp` 266–294）。

## 依据
- 分数 8.3：相对上轮 7.7，P0-2 对应的「网关未就绪仍起模型、仅 stderr」已在核心与 status 面关闭，交付时可从 `GET /api/v1/status.autostart` 直接看到 blocked/reason；监督/Catalog/门禁优势仍在。token 永久失败、无 uri 就绪、occupancy 声明与模型波 stderr 仍会卡交付或误判，故不到 9.0。

## 未读
无
