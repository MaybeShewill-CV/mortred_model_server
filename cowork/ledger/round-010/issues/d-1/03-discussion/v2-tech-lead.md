# 技术方案

- 问题: D-1 models deadline 到期后 child 可能仍停在 starting
- 版本: v2
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只收口模型波 `models deadline` 返回之后、supervisor 尚未 shutdown 时的 child 状态。`autostart_all` 在 `supervisor.cpp` 的 models deadline 分支今天只调用 `mark_models_failed_locked` 然后 return。已 launch、仍为 `starting` 的模型 child 要在同一把锁里变成 `failed`。`autostart_all` 返回时，以及再过至少一个就绪探测间隔之后，该 child 的 `state` 都是 `failed`。

一个 catalog id 仍是一个 OS 进程。deadline 分支不 fork 替换进程，也不向该进程发信号。进程继续占用原 pid 和原端口，直到既有 stop 或有序 shutdown。停服顺序保持：SIGINT、等待 `kStopGraceMs`、仍存活则外层 SIGKILL。超时 drain 之后析构成员、再由外层 SIGKILL，维持这条路径。

已是 `running` 的模型保持 `running`。尚未 `start_server` 的模型保持 `stopped`。已经是 `failed` 的模型（空 `server_uri`）不改写其 `error`。`__gateway` 不进这条终态。`models_failed`、`models_skipped`、`blocked` 和 `reason` 的既有极性不变。

不涉及 P1-5、P1-6、D-2 至 D-6，也不涉及 occupancy、网关 deadline、spawn 失败路径。

## 做法

### 推荐

在 models deadline 已成立、持有 `_mu` 的同一临界区内，先按 child 当前 `engine.state()` 处理，再调用现有的 `mark_models_failed_locked`。判定用锁内状态，不用临界区外的 `status()` 快照。

对本次模型波已 launch、且锁内状态仍为 `starting` 的非网关 child：

- `note_permanent_failure()`，状态变为 `failed`。
- `wanted = false`，`ready = false`，`backoff_due_ms = 0`。
- `stopping` 保持 false。
- `error` 为空时写入含字面 `deadline` 和该 catalog id 的句子。
- 子进程日志追加一行同等说明。
- 不发送 SIGINT 或 SIGKILL，不调用 `stop_server`。

锁内已经是 `running` 的 child 保持 `running`。尚未 launch、`pid < 0` 且为 `stopped` 的 child 保持 `stopped`。随后 `mark_models_failed_locked` 仍写入现有 deadline `reason`（含 `deadline` 与未就绪 id），`models_skipped` 仍为 false，然后 return。

探测线程若已在途：`probe_readiness` 只有在锁内仍为 `starting` 时才 `note_ready`。本分支先把状态写成 `failed` 后，这次探测不能把它改回 `running`。反过来，探测已先写成 `running` 的，本分支不能再写成 `failed`。

代价：改 `src/control/supervisor.cpp` 的 models deadline 分支，并改 `test/supervisor_unittest.cc` 的 `autostart_models_respects_deadline`，另增 `autostart_running_sibling_stays_running_on_peer_deadline`。未就绪 child 仍占一个 OS 进程和其端口，直到既有 stop 或 shutdown。不新增常驻进程，不新增显存或 engine。

### 否决

在 deadline 分支里对该 child 发 SIGINT 或 SIGKILL，或删掉 `stop_server`、`perform_ordered_shutdown`、析构 shutdown 上的外层 SIGKILL。波内发信号会把「宽限后再外层强杀」收成模型波内强杀，deadline 返回时 pid 不再保持为正。去掉外层 SIGKILL 就是把不计毛病的强杀路径改掉。

只用 `note_cancel()` 把状态写成 `stopped`，或只写 `models_failed`、不改 child。`stopped` 且 `stopping == false` 时，进程一旦退出，`note_exit` 仍可能再拉起。只写 gate 就是当前行为：gate 已失败，child 长期 `starting`。

不把 spawn 失败 child 的 `state == failed`、以及网关 deadline 时模型「不是 `starting`」写进验收。本版不改这两条用例去补断言。

## 改动面

- `src/control/supervisor.cpp`：`autostart_all` 的 models deadline 分支。需要私有辅助函数时，声明放在 `src/control/supervisor.h` 的现有 private 区。
- `test/supervisor_unittest.cc`：`SupervisorTest.autostart_models_respects_deadline`。
- `test/supervisor_unittest.cc`：新增 `SupervisorTest.autostart_running_sibling_stays_running_on_peer_deadline`。该用例只在测试配置里把 `start_concurrency` 设为 2。产品默认 `start_concurrency` 仍是 1。

## 明确不改

- `restart_policy.h` 的状态枚举、`note_exit`、backoff 和崩环阈值。终态复用已有 `note_permanent_failure()`。
- `stop_server`、`perform_ordered_shutdown`、析构里的 `request_shutdown`：SIGINT、`kStopGraceMs`、其后的 SIGKILL。
- 一个 catalog id 一次 `spawn_locked`、一个 OS 进程。
- 网关 deadline、空 `server_uri`、spawn 失败分支、occupancy 与 TRT 门禁。
- `autostart_models_failed_on_spawn_failure`、`empty_server_uri_fails_readiness_immediately`、`autostart_skips_models_when_gateway_deadline_missed` 的既有断言。不在这三条里新增 `status().state` 断言。
- HTTP 路径与字段名、`AutostartGate` 字段名、产品 toml、tensor 与模型后端。
- P1-5、P1-6、D-2 至 D-6。

## 契约

`servers[].state` 仍使用已有字符串：`stopped`、`starting`、`running`、`backoff`、`failed`。模型波超时且该 child 在 deadline 锁内仍为 `starting` 时，`state` 为 `failed`，`error` 含 `deadline` 与该 id。不新增 JSON 字段，不新增状态名。

`autostart.models_failed` 仍为 true。`models_skipped` 仍为 false。`blocked` 仍为 false。`reason` 仍含 `deadline` 与未就绪 id。

同一 id 的 `pid` 在 `autostart_all` 返回时仍大于 0。已 `running` 的兄弟 child 由验收第 6 条判定。

## 性能与资源

deadline 分支不增加 sleep，不增加进程，不加载 engine，不分配显存。未就绪 child 仍是原来的那一个 OS 进程。shutdown 仍先 SIGINT，宽限 `kStopGraceMs`（5000 ms），仍存活再 SIGKILL。

## 验收标准

1. `autostart_models_respects_deadline`：`autostart_all` 返回后，`fake_model_server` 的 `state` 为 `failed`，不是 `starting`，也不是 `running`。`error` 含 `deadline` 与 `fake_model_server`。`pid > 0`。`models_failed` 为 true，`models_skipped` 为 false，`blocked` 为 false，`reason` 含 `deadline` 与 `fake_model_server`。
2. 同一用例里再等待至少 1000 ms 后，该 child 的 `state` 仍为 `failed`，`pid` 与返回当时相同。
3. `autostart_healthy_models_not_failed`：deadline 内就绪的模型 `state` 为 `running`，`models_failed` 为 false。
4. 三条既有用例仍通过，不新增 `status().state` 断言。`autostart_models_failed_on_spawn_failure` 只判定 `missing_model` 的 `pid <= 0`、`reason` 含 `missing_model` 与 `failed`、`models_failed` 为 true。`empty_server_uri_fails_readiness_immediately` 判定 `fake_model_server` 的 `state` 为 `failed` 且 `error` 含 `server_uri`。`autostart_skips_models_when_gateway_deadline_missed` 判定 `models_skipped` 为 true 且 `fake_model_server` 的 `pid <= 0`。
5. 验收第 1 条和第 2 条里该 catalog id 只有一条 status，pid 不换成另一个正数。
6. `autostart_running_sibling_stays_running_on_peer_deadline`：网关先就绪。本波两个模型 child，该用例的 `start_concurrency` 为 2。`ready_sibling` 在 models deadline 前 `status("ready_sibling").state` 已是 `running`。`fake_model_server` 为 never-ready，并因此触发 models deadline。`gateway_deadline_ms` 为 3000，`models_deadline_ms` 为 2000。`autostart_all` 返回后，`status("ready_sibling").state` 仍为 `running`。`models_failed` 为 true，`models_skipped` 为 false，`reason` 含 `deadline` 与 `fake_model_server`。该用例期望退出码 0。

## 测试义务

QA 以 `supervisor_unittest` 判定第 1 条到第 6 条。第 4 条只按这三条用例的现有断言记录通过与否。第 6 条只认 `status("ready_sibling").state`。白盒核对 models deadline 临界区：只对锁内仍为 `starting` 的已 launch 模型调用 `note_permanent_failure()`；该分支没有 `kill`。`stop_server` 与 `perform_ordered_shutdown` 仍保留 SIGINT、`kStopGraceMs`、SIGKILL。不测真 GPU，不测 D-2 至 D-6。

## 自测命令

在 WSL、构建目录 `build/full` 执行。两条命令期望退出码都是 0。ctest 选中 0 条算失败。

```bash
cmake --build build/full --target supervisor_unittest -j
bash scripts/run_tests.sh build/full -R '^supervisor_unittest$' --output-on-failure -- --gtest_filter='SupervisorTest.autostart_models_respects_deadline:SupervisorTest.autostart_healthy_models_not_failed:SupervisorTest.autostart_models_failed_on_spawn_failure:SupervisorTest.empty_server_uri_fails_readiness_immediately:SupervisorTest.autostart_skips_models_when_gateway_deadline_missed:SupervisorTest.autostart_running_sibling_stays_running_on_peer_deadline'
```

```bash
bash scripts/run_tests.sh build/full -R '^supervisor_unittest$' --output-on-failure
```

## RD 不得自行决定

- 终态字符串固定为 `failed`。不得用 `stopped`、`backoff` 或新状态名代替。
- deadline 分支不得发 SIGINT 或 SIGKILL，不得调用 `stop_server`，不得把 `stopping` 设为 true。
- 不得删除或绕过外层 SIGKILL，也不得改 `kStopGraceMs`。
- 不得为同一 catalog id 再 spawn 一个进程。
- 不得把锁内已是 `running` 的 child 写成 `failed`。
- 不得把尚未 launch 的 `stopped` child 写成 `failed`。
- 不得改网关 deadline、空 `server_uri`、spawn 失败、`models_failed` 与 `models_skipped` 的极性。
- 不得给第 4 条的三条既有用例新增 `status().state` 断言。
- 第 6 条的测试名固定为 `autostart_running_sibling_stays_running_on_peer_deadline`。兄弟 id 固定为 `ready_sibling`。不得改产品默认 `start_concurrency`。
- 不得新增 HTTP 字段或 StatusCode。
- 不得改 P1-5、P1-6、D-2 至 D-6。
