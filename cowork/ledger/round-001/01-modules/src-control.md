# 模块评分

- id: src-control
- 权重: 10
- 分数: 7.7
- 读过的路径:
  - `src/control/supervisor.h`
  - `src/control/supervisor.cpp`
  - `src/control/catalog.h`

## 优势
- 进程树监督闭环完整：fork/exec、PDEATHSIG、SIGCHLD reaper、backoff/崩环放弃、就绪探测、网关先起/模型先停，小团队可按模型粒度启停整树（`supervisor.h` 31–36、86–94；`supervisor.cpp` `spawn_locked`/`reaper_loop`/`autostart_all`/`perform_ordered_shutdown`）。
- 出错可定位面够用：`Status` 含 state/pid/ready/restart_count/last_exit/error，另有按 id 的 `LogBuffer`，崩环有明确文案（`supervisor.h` 44–53、84；`supervisor.cpp` 233–234、582–598）。
- 拉起前有硬门禁：网关 scrape token、TRT engine、GPU occupancy（含 nvidia-smi 与已跑兄弟进程显存），能挡住多客户共卡时的盲目 spawn（`supervisor.cpp` 252–295、83–85）。
- Catalog 契约清楚：与 gateway 共用一套 registry，要求 port+server_uri，重复 id/port/uri 加载失败，cpu/gpu profile 过滤后再查重（`catalog.h` 37–57）。

## 劣势
- 网关 metrics token 为空会永久失败 spawn；小团队配错 token 时整棵入口起不来，且需懂「永久失败 / 手工再 start」（`supervisor.h` 65–67；`supervisor.cpp` 252–260、502–508）。
- autostart 失败与 deadline 只 `fprintf(stderr)`；网关失败后仍继续起 model，无路由时头少运维不易当场发现（`supervisor.cpp` 929–1004）。
- 无 `uri` 的 child 不做 readiness probe，可能长期停在 starting，直到 models deadline（`supervisor.cpp` 630–647、978–996）。
- occupancy 依赖策略里声明的 `gpu_mem_mib` 与 spawn 时 nvidia-smi；声明不准会拒启或仍有共卡风险，这三份文件看不到完整 pack/校准闭环（`supervisor.cpp` 266–294）。

## 依据
- 分数 7.7：监督与状态/日志足以支撑一两台机、多模型交付的启停与多数进程故障定位（`supervisor.h`/`supervisor.cpp`），Catalog 契约也利于 gateway 与 supervisor 对齐（`catalog.h`）；但 token/autostart 可观测性与 readiness/占用配置踩坑仍会卡交付，故不到 9.0，也未到「可独立交付且出错必能定位」的 10。

## 未读
无
