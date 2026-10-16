# 技术方案

- 问题: P1-3
- 版本: v1（编排方决策版，采纳技术负责人线）
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-3

## 边界

只改子进程的**进程组归属**与停止/重启/关机/失败清理的信号投递面（`src/control/supervisor.cpp` 的 7 处单 pid `::kill`）。不改 systemd 单元：`deploy/mortred-supervisor.service:43` 必须保持 `KillMode=process`，`scripts/stop_order_record.py` 的检查器要求"恰好一行 `KillMode=process`"并显式拒绝 `mixed`（已核实：`stop_order_record.py:128-143`、`:167` 把 `KillMode=mixed` 当反例）。

现状（已逐行核实）：投递点全是单 pid `::kill` —— `stop_server`（`:597` SIGINT、`:603` SIGKILL）、`fail_missing_server_uri_locked`（`:727` SIGKILL）、`perform_ordered_shutdown` 的 models（`:898`/`:907`）与 gateway（`:923`/`:931`）。子进程若 fork 出孙进程（辅助进程、`nvidia-smi`、模型内部子进程），父进程被 SIGKILL 后孙进程存活（rubric OP5）。

## 做法

### 推荐

1. **每个子进程一个进程组**：
   - 父进程在 `fork` 返回后立即 `::setpgid(pid, pid)`，容忍 `EACCES`/`EPERM`（子进程可能已 exec，这是标准做法，两侧都调以消除竞态）；
   - 子进程在 fd 关闭循环（`:456-458`）之后、`execve`（`:481`）之前调用 `::setpgid(0, 0)`——它是 async-signal-safe 的，只吃整数参数、不碰 libc 状态，与既有的 `prctl(PDEATHSIG)`/`getppid()` 复查/`sigprocmask` 并列即可。
2. **`Child` 记录 `pgid`**（= pid，`supervisor.h` 的 Child 结构），7 处投递点全部换成 `::killpg(child->pgid, SIG)`；`pgid <= 0` 时回退单 pid `::kill`（防御性分支，覆盖极早期失败）。
3. **新增可验证原语** `bool process_group_gone(pid_t pgid)`：`::killpg(pgid, 0) == -1 && errno == ESRCH`。stop 的 grace 与 SIGKILL 之后用它确认整组消失——它同时是测试断言原语与运维自检手段。

语义变化（必须写进文档与提交说明）：子进程不再属于 supervisor 的前台进程组，**终端 Ctrl-C 不再直达子进程**，子进程只经 supervisor 的两阶段关机收到 SIGINT。这是更确定的行为（此前 Ctrl-C 依赖终端投递，gateway 与模型可能收到两次或顺序不定）。

### 否决

- **`setsid`**：让子进程脱离会话与控制终端；且 supervisor 本身已由 `scripts/stop_order_record.py:29` 用 `/usr/bin/setsid` 启动，叠加会改 Ctrl-C 语义并让 PDEATHSIG 的调试路径更难懂。
- **父进程递归扫 `/proc` 找后代**：竞态（扫描期间新 fork）+ 每次 stop 的 O(进程数) 开销，且无法保证不漏。
- **把单元改成 `KillMode=mixed`/`control-group`**：违反 `stop_order_record.py` 的规则检查，且 systemd 只应管 supervisor 本身。

## 改动面

- `src/control/supervisor.cpp`（spawn 的父/子两侧 `setpgid`；7 处投递点改 `killpg`；新增 `process_group_gone`）
- `src/control/supervisor.h`（`Child::pgid` 字段与注释更新）
- `test/fake_model_server.cc`（新增 `--spawn-grandchild` 模式：fork 一个孙进程并把它挂在同一进程组）
- `test/supervisor_unittest.cc`（新增"停止后整组消失"用例）

## 明确不改

- `PDEATHSIG(SIGINT)`、`getppid()` 复查、gateway 最后退出的两阶段顺序、`kStopGraceMs=5000` 后 SIGKILL 升级
- 日志管道 reader 的所有权协议（spawn 等 `!readers_live`、`handle_exit` 先 close fd 再 detach joiner）
- systemd 单元与 `stop_order_record.py` 的检查规则本身
- **与 P1-2 串行**：两者都改 `supervisor.cpp`，P1-2 先合入后本项再从新基线切开

## 契约

- 停止/重启一个 id ⇒ 该 id 的整组（含孙进程）在 grace + 2s 内消失；组内任何进程不得跨 stop 存活。
- 停止顺序不变：models 先、gateway 最后；同一 grace 预算内先对整组 SIGINT，超时后对整组 SIGKILL。
- 终端 Ctrl-C 语义变化如上（属预期，写进文档）。

## 性能与资源

每个 spawn 增加 2 次 `setpgid` 系统调用；每次 stop 增加一次 `killpg(pgid, 0)` 轮询。无额外线程/分配，不影响推理路径。

## 验收标准

- A1 `./_bin/supervisor_unittest` 退出 0，且含新用例：子进程 fork 孙进程 → stop → 整组在 grace+2s 内消失（`process_group_gone` 为真）；现有用例（`restarts_do_not_leak_log_pipe_fds`、`shutdown_is_ordered_and_complete` 等）全绿。
- A2 外部验收：起服后取目标 id 的 pid/pgid → `POST /api/v1/servers/<id>/stop` 返回 200 → 10s 内 `kill -0 -<pgid>` 返回非 0 且 `pgrep -g <pgid>` 无输出。
- A3 gateway 组同样消失，且 models-先-gateway-后 的顺序未变（由 A1 的有序停机用例承担）。
- A4 `python3 scripts/stop_order_record.py --check` 退出 0（单元规则与样例轨迹未被改坏）。
- A5 提交只含上述 4 个文件；不含 `cowork/ledger/`、构建产物、`src/common/mortred_version.h`。

## 测试义务

- 上述 5 条，给出真实命令与退出码。
- 白盒：确认 `setpgid` 在子进程侧位于 fd 关闭之后、`execve` 之前；确认 7 处投递点**全部**改为整组投递（`grep -n '::kill(' src/control/supervisor.cpp` 只应剩 `pgid<=0` 的回退分支或 `killpg` 的封装）。
- 反例：孙进程忽略 SIGINT 时，grace 后必须被 SIGKILL 一并回收（覆盖升级路径）。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10f
cmake --build build-full --target supervisor_unittest fake_model_server -j8      # 期望 0
./_bin/supervisor_unittest ; echo $?                                            # 期望 0
python3 scripts/stop_order_record.py --check ; echo $?                          # 期望 0
grep -n '::kill(\|::killpg(' src/control/supervisor.cpp                          # 核对投递点
```

## RD 不得自行决定

- 是否给子进程独立 session（不做）；grace 是否对整组统一（是：SIGINT 整组 → 超时 SIGKILL 整组）；`pgid` 是否经 `Status` 暴露到 API（不暴露）；是否改动 systemd 单元（不改）。
