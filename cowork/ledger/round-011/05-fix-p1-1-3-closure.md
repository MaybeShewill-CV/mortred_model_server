# 修复收口：P1-1 / P1-2 / P1-3（+ P1-4 撤回、一处既有回归修复）

- 基线: `sub-agent-co-work @ f149cd1f`
- 分支: `fix/p1-1-4-metadata`（`4761143d` 修复、`ad18897c` doctor 顺序修正）
- 合入: **`bdf03d5d`**（`--no-ff`），10 文件 +220/−27
- 独立测试: **cv-qa 结论 `pass`**（18 行验收全绿；3 项"不适用"、4 条残余风险，见 §5）
- 编排方重跑: 带断言的 `logs/_orun_fixes.sh` → **`SUMMARY FAILS=0`**（与 QA 报告逐项一致）

## 1. P1-1 `stop_order_record.py --check` 对出厂 unit 报红

**问题不是"检查器陈旧"，而是单元文件本身有语义回归**：round-13 把 `KillMode` 从 `process` 改成 `control-group`，而 `control-group` 会把 `KillSignal`（SIGTERM）**同时**发给 gateway 与所有模型进程，直接破坏本单元存在的目的——"模型先退、gateway 后走"的有序停机（这正是 `stop_order_record.py` 规则要保护的东西）。

**正确取值是 `mixed`**：`KillSignal` 只发给主进程（保留有序停机），`TimeoutStopSec` 到期后 `FinalKillSignal`（SIGKILL）仍覆盖 cgroup 内所有残留进程（避免 round-13 想解决的"硬杀后遗留占显存的子进程"）。两者兼得。

- `deploy/mortred-supervisor.service`：`control-group` → `mixed`，注释说明为何两者都不对。
- `scripts/stop_order_record.py`：期望值改 `mixed`；负例改为 `process`/`control-group`/`none`/带空格；`unit_ok` 现在**忽略注释行**（此前任何含 `KillSignal` 字样的注释都会误伤），并新增两条注释相关断言（注释掉的指令不算通过；提到错误值的注释不影响合法单元）。
- `test/CMakeLists.txt`：`stop_order_record_rules` 不变（仍跑 `--check`）。

**验证**：`--check` 退出 0、首行 `KillMode=mixed 为通过`；`process/control-group/none/带空格/重复/缺失/ExecStop=` 全被拒（QA 跑了 12 例规则探针，12/12）；`ctest -R stop_order_record_rules` 通过。

**不适用**：本机 WSL 无 systemd PID1，`mixed` 的 cgroup SIGKILL **运行时**语义未实测（只验证了单元文本规则与检查器）。

## 2. P1-2 同名两把 metrics 令牌（静默 401 → 显式失败）

`conf/local/mortred_metrics_token` 是给 Prometheus 用的凭据文件，由 `write_prometheus_credentials.sh` 从 `MORTRED_METRICS_TOKEN` 派生；一旦令牌轮换而文件未重写，抓取就是 401，而**产品侧没有任何信号**。本机实测两把令牌确实不同（长度均 48、SHA 不同）。

- `scripts/write_prometheus_credentials.sh`：新增 `--check`（不匹配 → 退出 1 + 可执行的修复命令；**只读**，实测文件 sha256 前后一致）与 `--self-test`（匹配/陈旧/缺失/未设置四情形，输出 `SUMMARY FAILS=0`）。
- `scripts/mortredctl_doctor.sh`：凭据不一致时硬失败。**检查必须放在引擎门之前**——第一版放在之后，而本机 `yolov8s` 缺静态引擎导致引擎门先退出，该段**不可达**；`ad18897c` 修正了顺序。
- `test/CMakeLists.txt`：注册 `prometheus_credentials_check`（跑 `--self-test`，不需要活令牌）。

**验证**：`--self-test` 退出 0 / `SUMMARY FAILS=0`；真实文件 `--check` 退出 1 并打印 stale + 修复命令；doctor 退出 1，`== Prometheus scrape credential ==` 段**先于**任何引擎门文本出现；`ctest -R prometheus_credentials_check` 通过。

**残留**：本机 `conf/local/mortred_metrics_token` 仍是陈旧值（按要求未改），因此 doctor 的 FAIL 是**预期状态**；修复命令已在其输出里。另外 doctor 只在凭据文件存在时检查（文件缺失不硬失败）。

## 3. P1-3 占用门在 GPU 采样缺失时 fail-open

`occupancy_gate.h` 把"指纹不匹配"与"空闲显存低于 stamp"两项检查包在 `if (in.live.available)` 内：探针不可用时它们**静默消失**，带 stamp 的 TensorRT id 会以"已验证"的姿态启动，而实际什么都没验证。

- `src/control/occupancy_gate.h`：新增 `allow_unknown_gpu`；当 `candidate_is_tensorrt && has_gpu_mem_mib && gpu_mem_mib>0 && !live.available && !allow_unknown_gpu` 时**拒绝启动**，错误里给出 `MORTRED_ALLOW_UNKNOWN_GPU=1` 的显式放行方式。该检查位于 measured/worker/fingerprint/free-mem 之前。
- `src/control/supervisor.cpp`：从 env 读取开关（精确匹配 `1`/`true`/`TRUE`/`yes`）。
- `test/occupancy_gate_unittest.cc`：新增 3 例（fail-closed 默认、显式放行、非 TRT id 不受影响）。既有 10 例因契约改变而失败——它们测的是 stamp/预算/指纹，故在夹具里显式声明"已允许未知 GPU"，并把 `fingerprint_skipped_without_live_gpu` 重命名为 `..._when_override_set` 以免名字继续误导。

**验证**：`occupancy_gate_unittest` **26/26**（基线 23）。

**不适用/前提**：未制造真实 GPU 探针故障（破坏 nvidia-smi/拔卡），fail-closed 由单测注入 `live.available=false` 验证；且"无 override 不可能静默跳过"的前提是 `pack_active` 且检查未被禁用——全局 `occupancy_policy=off` + `MORTRED_OCCUPANCY_ALLOW_DISABLE=1` 仍可整体绕过（原有显式逃生门，未改动）。

## 4. P1-4 撤回（不是缺陷）与一处既有回归修复

- **P1-4 撤回**：`supervisor_app.cpp:924-931` 早已把相对 `MORTRED_CONTROL_CONFIG` 按 `resolve_project_root()` 重基为绝对路径。QA 与编排方各自从**非仓库 cwd**、**相对配置**、**未设 `MORTRED_PROJECT_ROOT`** 起服：gateway `state=running, ready=true, restart_count=0`；旧错误 `invalid control config` **0 次**。此前观察来自陈旧二进制。
- **回归修复（早于本分支，r14 引入）**：env allowlist 丢掉了测试用的 `MORTRED_ARGV_FILE`，使 `SupervisorTest.spawn_passes_model_flag_for_unified_exe` 一直红。修法为让 `test/fake_model_server.cc` 优先从**自己的 TOML**（`fake_argv_file=`）取该路径，env 仅作直接启动的兜底；**没有**把测试变量加进生产 allowlist（QA 已核对 `src/control/supervisor.cpp` 中 `MORTRED_ARGV_FILE` 出现次数为 0）。

**验证**：`supervisor_unittest` **26/26**（含该用例）。

## 5. QA 结论与保留

cv-qa 结论 **`pass`**，无缺陷。它同时如实列出：

- **不适用**：① 真实 systemd stop/restart（WSL 无 systemd PID1，unit 未安装）；② 真实 GPU 探针故障（未拔卡/未破坏 nvidia-smi）；③ 真实模型 spawn 全链路（本机 `yolov8s` 缺静态引擎，与本次改动无关）。
- **残余风险**：① 本机 metrics 凭据仍陈旧（预期 FAIL，未按 QA 要求改动）；② 写路径对默认 reader 65534 会尝试 `sudo chown`，无免密 sudo 时停在密码提示（原有设计）；③ doctor 仅在文件存在时检查；④ P1-3 的"不可能静默跳过"以 `pack_active` 且未禁用为前提。

## 6. 编排方自己的两处纠错（记录在案）

1. doctor 的凭据检查最初被放在引擎门之后 → 本机不可达；已前移（`ad18897c`）。
2. 编排方重跑脚本里"新用例出现次数"断言写成 1，实际 gtest 会打印 `[ RUN ]` 与 `[ OK ]` 两行 → 正确值 2；修正后 `SUMMARY FAILS=0`（属脚本断言错误，不是产品问题）。
