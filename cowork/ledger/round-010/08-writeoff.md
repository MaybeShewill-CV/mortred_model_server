# 关闭记录

- 轮次: 10
- 已关闭: P1-7、P1-1、P1-6、P1-4、P1-2、P1-5、P1-3（**7/7**）
- 阻塞: **条目层面无**。一处在验收形式上有保留：P1-3 的 A2/A3 未以 HTTP 活体形式运行（QA 判 `blocked`，编排方据 A1 的进程级证据裁决合入，详见 `issues/p1-3/05-qa.md` 末节与下方附录）
- 未阻塞的开放项: **相对本轮起点减少** —— 起点 7 条 P1（P0 为无），**7 条全部关闭并合入**；本轮新入列 **10 条候选（2 条 P1 级 N-1/N-2、8 条 P2 级 N-3…N-10）**，见"新入列"
- 本轮复核加权分: **7.4**（九份模块记录见 `07-modules/`，计算与上限核验见 `07-score-check.md`）
- 结果: **running**（不是 `done`、不是 `awaiting-env`、不是停止；三条排除依据见 `07-score-check.md` 的"终态判定"）
- 状态: 已收口

## 已关闭的问题陈述

| id | 模块 | 问题陈述 | 合入 | 记录 |
| --- | --- | --- | --- | --- |
| P1-7 | test（+ src-server-apps 一处注释） | 三个契约测试与产品行为不一致（测试未随产品改动更新） | `d4419a94` | `issues/p1-7/{03-design-final,04-rd,05-qa,05-merge}.md` |
| P1-1 | src-models | 检测参数 `clip_boxes` 是无效开关，且与实现相反：解析并存储但无人使用，而 unmap 实际上恒定裁剪到源图边界 | `db231f5c` | `issues/p1-1/…` |
| P1-6 | src-server-apps | `/metrics` 的 method 标签取自未鉴权的原始请求方法，标签基数无界 | `7453b1e2` | `issues/p1-6/…` |
| P1-4 | src-models | 请求路径上的 `CHECK` 断言会杀掉整个模型进程 | `19c649fa` | `issues/p1-4/…` |
| P1-2 | src-control | 管理/抓取凭据泄漏到每个被监管的模型进程 | `19eb0cb2` | `issues/p1-2/…` |
| P1-5 | src-server-apps（+ conf、docs） | 无界等待与配置矛盾：`model_run_timeout<=0` 时请求永不回复 | `7558abdf` | `issues/p1-5/…` |
| P1-3 | src-control | 停止/重启只 kill 单 pid，没有进程组，孙进程存活 | `a980ef03` | `issues/p1-3/…` |

七条都已按队列的"改完后能观察到的变化"验收，且由**编排方在 worktree 重跑**确认退出码与 QA 报告一致（逐条记录在各自的 `05-merge.md`）：

- P1-7：`request_envelope_unittest` 17/17、`server_e2e_contract_test` 45/45、`gateway_e2e_test` 49/49，三者退出码 0。
- P1-1：`detection_params_unittest` 8/8、`detector_common_unittest` 15/15、`object_detection_output_contract_unittest` 13/13；`grep -rn clip_boxes src/ test/ conf/ docs/openapi.json` 只余拒绝分支与新用例，出货 TOML 0 命中。
- P1-6：`prometheus_metrics_test` 11/11（含 4 条新用例，原 `method="POST"` 断言 0 删除行）、`server_e2e_contract_test` 45/45；QA 另用自编探针证明 13 个未知 method 只折叠成一个 `method="other"`，且 `GET|500` 无法伪造标签对。
- P1-4：`tensor_contract_unittest` 14/14、`http_contract_test` 6/6、`model_output_contract_unittest` 10/10、`object_detection_output_contract_unittest` 13/13、`backend_unittest` 3 passed/5 skipped；`grep -rn 'CHECK(checked_shape_nbytes\|CHECK_EQ(dtype' src/` 无输出；QA 另在 `/tmp` 编 `fence_proof` 证明"坏 shape → code=2、正常 → 0、进程存活"；**合入后编排方穷举复核整个 `src/models`：运行时 abort 构造 0 命中**。
- P1-2：`supervisor_unittest` 24/24、`supervisor_malformed_e2e_test` 1/1，且**真盘**重跑 QA 的 live 脚本得到 `SUMMARY FAILS=0`、`leftovers: none`（真 supervisor + 真 gateway + 真 mortredctl；模型子进程四键计数 0、gateway 子进程四键各 1、`mortredctl status` 退出 0）。
- P1-5：带断言脚本重跑 `SUMMARY FAILS=0`——提交面 `files=10 forbidden=0`、`base_server_impl.h diff_lines=0`、`server_runtime_config_unittest` 6/6、`server_e2e_contract_test` **47/47**（含 `model_timeout_returns_504` 与新的 `model_timeout_zero_is_clamped_to_a_bounded_budget`）、`conf/server` 内 `model_run_timeout=0` 计数 0、显式 300000 恰 5 份、中文文档"无限"0 命中。
- P1-3：带断言脚本重跑 `SUMMARY FAILS=0`——提交面 `files=4 forbidden=0`（并确认 `deploy/` 不在提交内）、四文件未被 QA 触碰、`deploy/mortred-supervisor.service` 最后提交仍是 `28e365eb 2026-10-13` 且 `KillMode=process` 恰 1 行、`stop_order_record.py --check` **exit 0**、`supervisor_unittest` **26/26**（新例 716 ms / 5723 ms，证明整组 SIGINT 一次到位、且反例真走 grace→SIGKILL 升级）、投递点仅剩 `:45/:86/:90` 三处、`pgrep -a fake_model_server` 无残留。合入后 `process_group_gone` 在基线上命中 **8** 处（合入前 0，见"完整性检查"）。

## 仍开放的问题

无。（本轮锁定的 7 条 P1 全部关闭；新发现的问题进入下方"新入列"，未混入本轮的开放项。）

## 完整性检查（收口时对基线做的物证核对）

`git merge-base --is-ancestor` 对七个条目分支全部成立；`git log --merges` 列出 7 个 Round 10 合并提交（`d4419a94`/`db231f5c`/`7453b1e2`/`19c649fa`/`19eb0cb2`/`7558abdf`/`a980ef03`）；关键落点文件全部 tracked；产物级抽查：P1-5 的 5 份配置 `model_run_timeout=300000` 命中 5、P1-6 的 `normalize_http_method` 命中 3（定义 + counter/histogram 入口）、P1-1 的 `not supported` 命中 2、P1-3 的 `process_group_gone` 由合入前 **0** 变为合入后 **8**。该检查具备判别力（能检出缺项）。

## 附录

### 用户指定移出的两类问题（本轮再次被独立证实，但仍不入列）

| 类别 | 本轮新证据 |
| --- | --- |
| 上一版 P0-1：构建/门禁在 HEAD 上不成立 | P1-4 的 QA 在 `build` 与 `build-full` 两个配置下再次复现 `packed_batch_nvi_unittest` 链接失败，并定位到根因是它的 `link.txt` 只链 `libcommon.so`、缺 `libmodels.so`（`gpu_jpeg::GpuDecodeSlot` 的 8 个符号在后者中）。编排方进一步确认 `packed_batch_nvi_unittest`（`test/CMakeLists.txt:54`）与 `cv_image_input_unittest`（`:68`）都在 `TEST_LIST` 内，而 `:348` 对每一项 `add_dependencies(check ${src})`，因此 CI 的 `tests` job（`ci.yml:114` 构建 `check`）**按构造无法通过**。此外 `enlightengan_output_contract_unittest.cc` 另有独立的编译失败（`enlightengan.inl:169-170` 缺 `model_runtime.h`）。以上均按用户指令不入列，仅留档 |
| 上一版 P1-4：数值门禁可跳过 | 本轮未新增证据，状态同上一轮 |

### 完成标准行进展（详见 `07-score-check.md`）

- **已满足**：ST1、ST2、ST3、ST5、ST6、OP4（各带文件/命令/测试出处，部分为编排方实跑）。
- **未满足**：ST4 第二半——"停掉驻留集一个模型后另一个 id 仍成功"没有任何可复现证据（`GatewayE2ETest` 夹具只起一个 upstream）；按 `rubric.md:91` 稳定性行不得判阻塞，故按未满足处理并列入"新入列"。
- **阻塞**：TP2（无已记录的真盘产物；且 `resident_pair_record.py --check` 的退出 0 **不是**证据，它只是对内置样例跑 judge）、TP3（需在缺 engine/占用戳状态下跑 `mortredctl doctor --strict`；`weights/` 被所有 worktree 目录联接共享，多条目并行期间不实跑）。
- **待判**：PF1–PF5、TP1、AV1、OP1–OP3、OP5、UX1–UX4、EX1–EX3。其中 PF/TP 系列需要 TensorRT calibrate + `http_infer_rps.py --raw` 压测等 GPU 证据，本轮 7 个 P1 不产出这类证据；OP5 待 P1-3 关闭后按进程组回收重判。

### 加权分（收口时填）

- 模块记录：九份齐备（`07-modules/*.md`）——`ci` 5.0(5)、`src-server-apps` 7.0(20)、`src-models` 7.5(15)、`src-control` 7.5(15)、`ops` 8.0(15)、`src-factory-common` 8.0(10)、`test` 7.0(10)、`conf` 7.5(5)、`docs` 8.0(5)。
- 原始分 = 737.5 / 100 = **7.375 → 7.4**；上限核验：本轮无未关闭且未阻塞的 P0，只有 P1 级候选 → 上限 8.9，**不构成约束**。
- 结论：本轮的分数瓶颈是**模块自身缺陷**（`ci` 的两道红门、`src-server-apps` 的 gauge 缺陷、`test` 的三条基础设施完整性缺陷、`src-models` 的同族顺序不一致、`src-control` 的孤儿窗口），不是队列未清空——下一轮重心应放在这些模块缺陷上。

### 验收形式上的保留（P1-3 的 A2/A3）

P1-3 的 QA 判 `blocked`：A1/A4/A4b/A5 与全部白盒项 pass，但 A2（真 HTTP `POST /api/v1/servers/<id>/stop` → 含孙进程的组在 10 s 内消失）与 A3（gateway 组消失）**未以 HTTP 活体形式运行**——`conf/local/trust.env` 不存在，且活体孙进程只能由 `test/fake_model_server.cc` 造（它读自身配置的 `fake_spawn_grandchild`，supervisor 无法传该 flag），实跑需手搓一整套 harness。编排方裁决合入的理由：① 被改的那一层（进程组归属 + 信号投递）已由 A1 用**真实子进程与真实孙进程**、经与 HTTP 端点**同一条** `stop_server()` 路径证明修复，反例还证明 grace→SIGKILL 升级真的执行；② A2 未跑的部分是**未改动的既有面**。保留意见：若评审要求"必须 HTTP 活体"，本项应回退为阻塞并只重跑 A2/A3（恢复路径见 `issues/p1-3/05-qa.md`），无需重开方案。

## 新入列

以下为**本轮新发现**、满足入列条件（出处 / 现在的行为 / 一次外部验收即可确认的变化）的条目候选，供下一轮锁定范围时取舍。编号仅为候选序号，不代表优先级。

| 候选 | 级别 | 问题 | 出处 | 现在的行为 | 一次外部验收即可确认的变化 |
| --- | --- | --- | --- | --- | --- |
| N-1 | P1 | `mortred_waiting_jobs` gauge 永不回落 | `base_server_impl.h:774-775`（快照）vs `:557-561`（递减） | 单请求回包后 gauge 停在最后一个回包时的在飞数（单请求场景恒为 1）；RD 实测轮询 40 次 × 3s 仍读到 1，运维无法从 `/metrics` 判断排空 | 发一个请求并等其完成，`/metrics` 的 `mortred_waiting_jobs` 回到 0 |
| N-2 | P1 | 单模型故障后"另一个 id 仍可用"无验证、无脚本 | `test/gateway_e2e_test.cc:497` 的单 upstream 夹具；`test/process_stop_unittest.cc` 只测信号语义；`scripts/resident_pair_record.py` 覆盖的是 TP2 | 网关把上游拒连映射为 503 且路由按 id 独立，但没有任何测试/脚本同时持有"一个活 id + 一个死 id" | 起驻留集两个 id，停其中一个：gateway 对该 id 返回 503、对另一个 id 仍返回 200，并固化成脚本或用例 |
| N-3 | P2 | 非法 base64 的 violation 指针是压缩下标，会指错元素 | `parsed_request.h:130-131`（遍历的是 `decoded.value.images`） | `{"images":[5,"ok"]}` 报 `/images/0`，而用户看到的元素下标是 1 | 该请求的 violation 指针为 `/images/1` |
| N-4 | P2 | `src/common/mortred_version.h` 是入库生成物，工作区恒为 CRLF-dirty | 任意 worktree 的 `git status`；`.gitattributes` 的 `* text=auto` | 每个 worktree 都显示 ` M src/common/mortred_version.h`，四个实现者都要手工排除它 | `git clone` 后无论是否构建，`git status --porcelain` 都不含该文件 |
| N-5 | P2 | 凭据擦除是 drop-list 而非 allowlist | `src/control/supervisor.cpp:403-406`、`:60` | 将来新增的凭据型环境变量若未登记，就会重新被整份 `environ` 带进被管子进程 | 契约条目要求"新凭据键必须登记进 scrub 列表"，并有测试遍历该列表 |
| N-6 | P2 | TSan 标签门按构造为红：7 个目标带 `sanitizer` 标签但该步只构建 6 个 | `test/CMakeLists.txt:349-355`；`ci.yml:328`（构建 6 个）与 `:335`（`ctest -L sanitizer`）；实跑 `ctest -N -L sanitizer` 在三个目录均登记 7 个 | 被选中但未构建的 `rate_limit_gcra_unittest` 没有可执行文件，该步失败；数据竞争实际上没有被门控，且注释"runs exactly these"与实现不符 | `ctest -L sanitizer` 只选到已构建的目标（或该步补建第 7 个），且 TSan 步在 CI 中为绿 |
| N-7 | P2 | v5/v6/v7 与 v8 的后处理顺序不一致（`unmap → NMS` vs `NMS → unmap`） | `yolov5_detector.inl:88-91`、`yolov6:85-88`、`yolov7:136-141` vs `yolov8:75-85`；`request_geometry.h:139-142` 的恒定夹取 | NMS 在不同坐标系、不同框集合上求 IoU，同族模型对同一图可能给出不同框集合 | 统一顺序后，同图同参下 v5 与 v8 的框集合关系可解释（或文档写明差异并有用例固定） |
| N-8 | P2 | 两个测试目标不可链接/不可编译（构建面） | `packed_batch_nvi_unittest` 的 `link.txt` 缺 `libmodels.so`；`test/enlightengan_output_contract_unittest.cc` 缺 `model_runtime.h` | `check` 目标因此中断，CI `tests` job 无法通过；`BackendCvModel::run_batch` 的 packed 栅栏只有代码走查 | 两个目标均可构建，且 `cmake --build build-ci --target check` 能跑完 |
| N-9 | P2 | `process_group_gone` 的 pid 复用竞态与快照不一致 | `src/control/supervisor.cpp:655`（快照 `pgid`）vs `:674`（重新取 `child->pgid`） | `reap_group_with_grace` 用的是快照下来的 *pid 值*；极端情况下该 pid 在 grace 窗口内被复用并新建同号组，探测会误判"组仍在"从而多打一次 `killpg(SIGKILL)` 到无辜组（窗口 ≤5 s、需绕回 `pid_max`，只会多杀不会漏杀） | 两处取 `pgid` 的来源统一（或快照改为不可复用的句柄），并有注释说明为何安全 |
| N-10 | P2 | 现成的真 HTTP 宿主未进 `build-full`，使"HTTP 驱动的停止"这类验收无便宜路径 | `test/supervisor_multiinstance_test.cc`（in-process supervisor + 真 listener + `POST /api/v1/servers/<id>/stop`）由 `test/CMakeLists.txt:106` 条件追加，`_bin` 无该产物 | P1-3 的 A2/A3 因此未能以 HTTP 活体形式验收（见上方"验收形式上的保留"）；将来任何人要做同类验收都要另搭 harness | 该目标在 `build-full` 中可构建，并能作为"HTTP → 管理 API → stop"的活体验收宿主（可再加 `fake_spawn_grandchild=1` 覆盖孙进程） |
