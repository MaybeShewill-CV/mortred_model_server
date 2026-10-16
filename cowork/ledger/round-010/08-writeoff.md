# 关闭记录

- 轮次: 10
- 已关闭: P1-7、P1-1、P1-6、P1-4、P1-2（5/7）
- 阻塞: 无（7 个条目无一因环境阻塞；被阻塞的是完成标准里的 engine/吞吐类行，见"完成标准行进展"）
- 未阻塞的开放项: **相对本轮起点减少** —— 起点 7 条 P1（P0 为无），当前 2 条仍未关闭（P1-5 在 QA、P1-3 在实现），其余 5 条已合入
- 状态: **进行中**（本文件在收口过程中补齐；P1-5、P1-3 关闭后本文件为最终版）

## 已关闭的问题陈述

| id | 模块 | 问题陈述 | 合入 | 记录 |
| --- | --- | --- | --- | --- |
| P1-7 | test（+ src-server-apps 一处注释） | 三个契约测试与产品行为不一致（测试未随产品改动更新） | `d4419a94` | `issues/p1-7/{03-design-final,04-rd,05-qa,05-merge}.md` |
| P1-1 | src-models | 检测参数 `clip_boxes` 是无效开关，且与实现相反：解析并存储但无人使用，而 unmap 实际上恒定裁剪到源图边界 | `db231f5c` | `issues/p1-1/{03-design-final,04-rd,05-qa,05-merge}.md` |
| P1-6 | src-server-apps | `/metrics` 的 method 标签取自未鉴权的原始请求方法，标签基数无界 | `7453b1e2` | `issues/p1-6/{03-design-final,04-rd,05-qa,05-merge}.md` |
| P1-4 | src-models | 请求路径上的 `CHECK` 断言会杀掉整个模型进程 | `19c649fa` | `issues/p1-4/{03-design-final,04-rd,05-qa,05-merge}.md` |
| P1-2 | src-control | 管理/抓取凭据泄漏到每个被监管的模型进程 | `19eb0cb2` | `issues/p1-2/{03-design-final,04-rd,05-qa,05-merge}.md` |

五条都已按队列的"改完后能观察到的变化"验收，且由**编排方在 worktree 重跑**确认退出码与 QA 报告一致（逐条记录在各自的 `05-merge.md`）：

- P1-7：`request_envelope_unittest` 17/17、`server_e2e_contract_test` 45/45、`gateway_e2e_test` 49/49，三者退出码 0。
- P1-1：`detection_params_unittest` 8/8、`detector_common_unittest` 15/15、`object_detection_output_contract_unittest` 13/13；`grep -rn clip_boxes src/ test/ conf/ docs/openapi.json` 只余拒绝分支与新用例，出货 TOML 0 命中。
- P1-6：`prometheus_metrics_test` 11/11（含 4 条新用例，原 `method="POST"` 断言 0 删除行）、`server_e2e_contract_test` 45/45；QA 另用自编探针证明 13 个未知 method 只折叠成一个 `method="other"`，且 `GET|500` 无法伪造标签对。
- P1-4：`tensor_contract_unittest` 14/14、`http_contract_test` 6/6、`model_output_contract_unittest` 10/10、`object_detection_output_contract_unittest` 13/13、`backend_unittest` 3 passed/5 skipped；`grep -rn 'CHECK(checked_shape_nbytes\|CHECK_EQ(dtype' src/` 无输出；QA 另在 `/tmp` 编 `fence_proof` 证明"坏 shape → code=2、正常 → 0、进程存活"。
- P1-2：`supervisor_unittest` 24/24、`supervisor_malformed_e2e_test` 1/1，且**真盘**重跑 QA 的 live 脚本得到 `SUMMARY FAILS=0`、`leftovers: none`（真 supervisor + 真 gateway + 真 mortredctl；模型子进程四键计数 0、gateway 子进程四键各 1、`mortredctl status` 退出 0）。

## 仍开放的问题

| id | 模块 | 问题陈述 | 状态 |
| --- | --- | --- | --- |
| P1-5 | src-server-apps（+ conf、docs） | 无界等待与配置矛盾：`model_run_timeout<=0` 时请求永不回复 | 未做（实现已提交 `d86e2216`，QA 进行中） |
| P1-3 | src-control | 停止/重启只 kill 单 pid，没有进程组，孙进程存活 | 未做（实现进行中，分支 `cowork/r10-p1-3-process-group`） |

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

- 模块记录：`ci` 5.0（权重 5）、`ops` 8.0（15）、`src-factory-common` 8.0（10）；其余 6 份待评。
- 上限规则：本轮无未关闭 P0，故只有 P1 时上限 8.9（`rubric.md:182`）。

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
