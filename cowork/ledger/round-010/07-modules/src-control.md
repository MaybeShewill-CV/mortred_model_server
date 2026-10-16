# 模块：src-control

- 路径: `src/control`
- 权重: 15
- 分数: 7.5
- 基线: sub-agent-co-work @ `05c74314`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **规模与分层**：28 文件 / **9,161 行**，四个子域清晰——`supervisor/`（进程树与生命周期）、`gateway/`（数据面、鉴权、限流、路由）、`cli/`（`mortredctl` 的 11 个子命令）、`rate_limit/`（GCRA 与 subject 归类）。最大的四个文件：`supervisor.cpp`(1251)、`cli/cli_app.cpp`(1129)、`gateway/gateway_app.cpp`(1066)、`supervisor/supervisor_app.cpp`(1007)。
- **本轮在此模块完成的两项改造（均已合入并复跑）**：
  - **P1-2**（`19eb0cb2`）：删除全局 `setenv("MORTRED_API_TOKEN")`，在 `spawn_locked` 的 `environ` 拷贝之后、`fork()` 之前按 `key + "="` 精确删除四个凭据键，再由分支注入按角色恢复。**QA 为真盘验收**（真 supervisor + 真 gateway + 真 mortredctl）：模型子进程四键计数 **0**、gateway 子进程四键各 1、`mortredctl status` 退出 0；实现者另做**变异验证**（注释掉一处 drop 后新用例立即失败并打印 `model child inherited MORTRED_INTERNAL_TOKEN`），证明断言不是空的。
  - **P1-3**（`9478e8f0`）：每个被管子进程自成进程组，7 处投递点改整组投递，新增 `process_group_gone`/`signal_process_group`/`reap_group_with_grace`；`supervisor_unittest` **26/26**，含"整组回收"正例与"孙进程忽略 SIGINT 触发升级"反例。
- **鉴权与限流的判定面集中在网关**：`gateway_app.cpp` 的守卫链有明确顺序（`:869-885` 非环回 + 限流关闭 → 拒绝启动；`:923` expose 守卫；`:933` token 可用性；`:941-946` metrics token 与推理/管理 token 相同 → 拒绝启动），并且 401 路径会 `ip_limiter_->release(...)` 归还已 admit 的配额（`:767-768`）——完成标准 ST2/OP4 的证据都在这里，且都有 e2e 用例（`gateway_e2e_test.cc:969/1012/1020`、`server_e2e_contract_test.cc:733`）。
- **GPU 采样与占用门可插拔且有 fail-closed 测试**：`gpu_probe.h`（NVML 优先 → nvidia-smi 兜底 → 否则 fail-closed）、`gpu_sampler.h`（`[N/A]`/空串一律当无效读数）、`occupancy_gate.h`；`test/gpu_probe_unittest.cc` 用 `FakeGpuProbe`/`ChainGpuProbe` 覆盖"不可用时 fail-closed"（ST6 证据）。
- **停止顺序有可执行规则检查**：`python3 scripts/stop_order_record.py --check` 在 P1-3 合入前后**均为退出码 0**（编排方事前在 `ae6c5bac` 实测 0、事后复核；脚本输出 4 行「为通过」+ 3 行自带负例自检）；且 `deploy/mortred-supervisor.service` 的最后一次提交仍是 `28e365eb`（2026-10-13），即 P1-3 未动 systemd 单元。

## 问题

1. **supervisor 自身被 SIGKILL 时孙进程仍可能存活（P1-3 声明的范围边界）**：`PR_SET_PDEATHSIG` 只覆盖直接子进程；supervisor 被 `SIGKILL` 时进程组机制没有机会执行。彻底覆盖需要 `KillMode=control-group`，而方案的 `stop_order_record.py` 规则检查明确要求单元保持 `KillMode=process`。当前缓解是孙进程随 PDEATHSIG 级联（组长死 → 组长的子进程收 SIGINT），但"组长先被外部 SIGKILL、孙进程忽略 SIGINT"这一组合仍会留下孤儿。
2. **P0-4 残留的日志线程风险仍未收口**：`handle_exit` 现在是 close fd 后 **detach** joiner，阻塞在 `read()` 的线程能否被唤醒在 Linux 上未定义（fd 号可能被复用），因此仍可能按停顿子进程泄漏线程、该 id 重启需等 `spawn_locked` 的 10s 上界；`LogBuffer` 仍是阻塞写在单锁下（只是不再每行 flush）。第 9 轮 P0-4 的关闭记录已写明，本轮未动。
3. **同 uid 的 `/proc/<pid>/environ` 通道未关闭**：P1-2 只关闭了"继承"面；模型子进程与 supervisor 同 uid，理论上可直接读 supervisor 的 `environ` 取走管理 bearer。要关闭需改运行用户/权限模型（范围外）。
4. **凭据擦除是 drop-list 而非 allowlist**：将来新增凭据型环境变量若未登记进 `spawn_locked` 的 scrub 列表，P1-2 修好的那类泄漏会以新名字复活（已记入 `discovered.md` 并给出可验收的变化）。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| ST2（鉴权失败不得消耗限流配额；未鉴权不得调推理路由） | **已满足** | 401 路径归还配额：`gateway_app.cpp:767-768`（注释 `:589-590` 说明 enforce/shadow 都会走这条）；`:719` 无效 key 不占 `rate_window_count`；端到端 `server_e2e_contract_test.cc:733`（`rate_limit_qps=1` 下 5 次无 token 全 401、随后带 token 200）；未鉴权不得调推理路由由 `gateway_e2e_test.cc:336` 与 `:505` 覆盖 |
| ST5（`fork` 之后、`exec` 之前不得调用非 async-signal-safe 函数） | **已满足（P1-3 之后仍成立）** | `supervisor.cpp` 的子块按序为 `close` 循环 → `setpgid(0,0)`（P1-3 新增，async-signal-safe）→ `syscall(SYS_prctl,…)` → `getppid` → `sigaction`/`sigprocmask` → `chdir` → `execve` → `_exit`；无 `setenv`/`malloc`/`printf`。P1-2 的 `drop_env_entry` 在 **fork 之前**执行 |
| OP2/OP3（按 id 看状态与最近退出；只重启单个 id） | **已满足** | supervisor 状态含每个 id 的 `state`/`pid`/`restarts`/`last_exit`/`error`（P1-2 真盘输出）；按 id 的 start/stop/restart 有 `supervisor_unittest` 重启用例；P1-3 之后 stop 还会确认整组消失 |
| OP5（收到停止信号后模型先退出；gateway 退出后不留模型进程） | **已满足（本轮由 P1-3 补齐后半）** | 有序停机：`supervisor_unittest.shutdown_is_ordered_and_complete`；整组回收：`stop_signals_the_whole_process_group`（并断言 `stop_ms < 4000` 以证明整组 SIGINT 生效）与 `stop_reaps_grandchild_that_ignores_sigint`（升级路径）；`process_group_gone` 作为断言原语。**残留边界**见问题 1（supervisor 自身被 SIGKILL） |
| OP4（三类 token 缺失或彼此相同时 gateway 必须拒绝启动） | **已满足** | `gateway_e2e_test.cc:969/1012/1020` 三个 e2e 用例 + `trust_tokens_unittest.cc:38,44` 的 `kScrapeTokenCollision` 断言 |

## 评分依据

- 给 7.5 而不是更高：① 问题 1 是一个**真实且无法在本轮范围内消除**的孤儿进程窗口（supervisor 被 SIGKILL），而"停止后不留残留"正是 OP5 的判据，只是当前证据覆盖了正常停机路径；② 问题 2 的日志线程泄漏是第 9 轮 P0-4 的已知残留（detach 后的 `read()` 阻塞未定义），会让停顿子进程的 id 重启变慢；③ 问题 3/4 说明凭据隔离只做到了"继承面"，且是 drop-list 式而非 allowlist 式；④ 四个文件都在 1000 行以上（`supervisor.cpp` 1251），控制面是最难审查的部分。
- 给 7.5 而不是更低：① 本轮两项改造都按最高标准验收——P1-2 有**真盘 + 变异验证**，P1-3 有**正例 + 反例 + 前提自证**（测试先断言 `getpgid(leader)==leader`、`getpgid(leader)!=getpgid(getpid())`、`getpgid(grandchild)==leader` 再发信号）；② 守卫链与限流语义有明确顺序且各有端到端用例，ST2/OP4 的判定不靠读代码；③ GPU 采样是"NVML → nvidia-smi → fail-closed"且 `[N/A]` 不冒充读数，并有专门单测；④ 停止顺序有可执行规则检查（事前/事后均为 0），systemd 单元未被本轮的进程组改造波及（最后提交仍是 `28e365eb`）。
- 参照系：把问题 1（孤儿窗口，至少写进文档并给出 `KillMode=control-group` 的取舍说明）与问题 2（日志线程收口）处理掉之后，本模块有理由到 8.5。
