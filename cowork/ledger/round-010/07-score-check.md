# 评分复核

- 轮次: 10
- 评分人: 编排方（单一执行方模式）
- 基线: `sub-agent-co-work`
- 状态: **进行中** —— 本文件在收口过程中逐行补齐；九份模块记录完成后，本文件的"完成标准行"与"加权分"两节即为最终判定。

## 完成标准行（逐行：已满足 / 未满足 / 阻塞）

判据来自 `cowork/ledger/rubric.md:95`：未满足的行必须进入 P0/P1/P2 并写出出处、现在的行为和一次外部验收即可确认的变化；阻塞仅用于环境跑不了的验收，且只阻塞性能、吞吐、engine 和占用行（`rubric.md:91`）。

### 稳定

| 编号 | 判定 | 证据 |
| --- | --- | --- |
| ST1 | **已满足** | ① 未知字段被拒：`src/common/request_envelope.h:97` 生成 `unknown request field '<key>'; allowed: ...`，e2e `test/server_e2e_contract_test.cc:505` `strict_envelope_rejects_unknown_fields_and_bad_params`、`:1116` `async_submit_unknown_field_metrics_use_http_422`，且 OpenAPI 侧 `test/openapi_consistency_test.cc:173-176` 断言 `additionalProperties == false`；② 空图被拒：`request_envelope_unittest.cc` 的 `images_shape_errors` 含 `{"images":[""]}` → `non-empty` 违规；③ 错误 token → 401（`gateway_e2e_test.cc:336` `model_route_requires_token` 等）；④ "失败被报成空成功"：契约违规映射为 HTTP 500（`src/server/http_status.h`），由 `test/http_contract_test.cc:43-44` 断言 |
| ST2 | **已满足** | ① 鉴权失败**不消耗**限流配额：`src/control/gateway/gateway_app.cpp:767-768` 在 401 路径上 `ip_limiter_->release(ip_gate.subject)` 归还已 admit 的配额（`:589-590` 注释明写"An allowed admit is not released here: process() does that on the pre-forward 401 path, for both enforce and shadow"），`:719` 注释说明无效 key 不占 `rate_window_count` 槽位，`:722-725` 保证"合法但被限流的 key 得到 429 而非 401"；② 有端到端用例：`test/server_e2e_contract_test.cc:733` `unauthenticated_stays_401_under_rate_limit` —— 在 `rate_limit_qps=1` 配置下连发 5 次无 token 请求全部得到 **401**（而非 429），随后一次带 token 请求得到 **200**（"A subsequent authorized request still gets a full QPS budget"）；③ 未鉴权不能调推理路由：同文件 `:505`/`gateway_e2e_test.cc:336` |
| ST3 | **已满足（其一）** | 队列打满 → 429 + `Retry-After`：`test/server_e2e_contract_test.cc:937-965` `queue_limit_returns_429_with_retry_after`（断言 429、`retry-after ∈ [1,60]`、`mortred_queue_rejected_total > 0`）；"拒绝结束后后续合法请求仍能成功、槽位可复用"：P1-5 新增用例在 `max_queue_depth=1` 下用 `wait_for_job_release` + "第二个请求不再 429" 证明（见 `issues/p1-5/04-rd.md`）；"进程按重启策略回到 ready"由 `supervisor_unittest` 的重启用例承担。**注**：本行的 `Retry-After` 取值上限受 `ewma_seed_ms` 影响，那 5 个原 `model_run_timeout=0` 的 id 已随 P1-5 由 1s 变为触顶 60s（见 `discovered.md`），属契约内的取值变化、不是未满足 |
| ST5 | **已满足** | supervisor 在 `fork` 之后、`execve` 之前只调用 async-signal-safe 原语：`src/control/supervisor.cpp:455-482` 依次为 `::close`（fd 循环）、`::syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)`（注释明写"syscall, not libc prctl"）、`::getppid`、`::sigemptyset`/`::sigaction`（SIGINT/SIGQUIT 复位 SIG_DFL）、`::sigprocmask`、`::chdir`、`::execve`、`::_exit`。无 `setenv`/`malloc`/`printf`/`std::string` 等非 async-signal-safe 调用。**P1-2 的 `drop_env_entry` 在 fork 之前执行、P1-3 计划插入的 `::setpgid` 亦为 async-signal-safe**，两项落地后本行仍成立（合入后需复跑本行的核对命令确认） |
| ST6 | **已满足** | 采样链是"NVML 优先、nvidia-smi 兜底、否则 fail-closed"：`src/control/gpu_probe.h:226`（"Prefer NVML, then nvidia-smi. First available sample wins; else fail-closed"），`backend` 字段记录 `nvml`/`nvidia-smi`/`fake`（`:36`）；`src/control/gpu_sampler.h:62,67` 把空串/`[N/A]`/`N/A` 一律当无效读数走 fallback（**不会把 N/A 当 0% 利用率**），`:14` 注明"proven against hung drivers"、`:111` 用带硬 deadline 的非阻塞 probe；面板侧 `src/control/cli/cli_app.cpp:251` 渲染 `n/a`，`supervisor_app.cpp:849` 注明 HUD 采样线程"graceful offline"。**测试**：`test/gpu_probe_unittest.cc` 用 `FakeGpuProbe`/`ChainGpuProbe` 覆盖 `fake_probe_returns_canned_fail_closed_when_unavailable`（`:11`）、`fake_probe_available_sample`（`:21`）、`chain_prefers_first_available`（`:37`），即"不可用时 fail-closed"有专门用例 |
| ST4 | **未满足（第二半缺可复现证据）** | 第一半（停掉一个模型后 gateway 对该 id 不是 503 = 未满足）**已有证据**：`gateway_app.cpp:371` 把 `ECONNREFUSED` 映射为 503，`test/gateway_e2e_test.cc:497` `dead_upstream_maps_to_503` 实证（停掉 fake upstream 后请求得到 503 + 体内 `"status":65`）。第二半（**驻留集里另一个 id 的合法请求仍成功**）**没有可复现证据**：`GatewayE2ETest` 夹具只起**一个** upstream（`fake_pid_` 单实例，`server_exe="fake_model_server.out"`），`test/process_stop_unittest.cc` 只验证 `ProcessStop` 的信号语义（5 个用例均为 WaitGroup/析构级），`scripts/resident_pair_record.py` 覆盖的是 TP2 的"两边同时 ready 且重叠推理"而非"停一个、另一个仍可用"。按 `rubric.md:91`，阻塞只允许用于性能/吞吐/engine/占用行，**ST4 属稳定性行，不得判阻塞**，故按未满足处理。**入列建议（P1）**：出处 = 上述夹具与脚本的覆盖缺口；现在的行为 = 网关把上游拒连映射为 503 且路由按 id 独立（`gateway_app.cpp:371`、`:288` 的 `/v1/models/` 前缀路由），但没有任何测试或脚本同时持有"一个活 id + 一个死 id"；一次外部验收即可确认的变化 = 起驻留集两个 id，停其中一个，断言 gateway 对该 id 返回 503、对另一个 id 仍返回 200，并把该场景固化成脚本或用例 |
| TP2 | **阻塞** | 覆盖标准要求"`yolov8s` 与 `libface` 同时 ready 时对两者并发请求且都有成功响应"。仓库里有专用程序 `scripts/resident_pair_record.py`（live 与 `--check` 共用同一 `judge()`：要求时间重叠、两边 before 均 ready、after pid 一致、两条 HTTP 均为 200 且 `status==0`）。**但本树没有已记录的真盘产物**（`logs/resident_pair_yolov8s_libface.json` 不存在），而 `--check` 按设计**不读也不写**该 json——它只是对内置样例跑一遍 judge。编排方实跑确认：`python3 scripts/resident_pair_record.py --check` **退出 0**，同时打印 `no recorded resident-pair json`，即**该退出码不能作为 TP2 的已满足证据**（这一点很容易被误读，特此写明）。TP2 属吞吐行，判阻塞；**恢复条件**：在有 weights/engine 且 8787/8080 空闲时跑 live 形式生成记录，再以 `--check` 复核 |
| ST4 | 待补 | 需要一份"驻留集里停掉一个模型 → gateway 对该 id 返回 503、其它 id 仍可用"的活体证据；收口时用既有 e2e/活体记录判定 |

### 运维

| 编号 | 判定 | 证据 |
| --- | --- | --- |
| OP4 | **已满足** | 三类凭据缺失或彼此相同时 gateway 拒绝启动，且**有 e2e 用例**：缺 metrics token → `test/gateway_e2e_test.cc:969` `MetricsTokenUnsetRefusesToStart`、`:1012` `NonLoopbackWithoutMetricsTokenRefusesToStart`；metrics token 与推理/管理 token 相同 → `:1020` `MetricsTokenMatchingInferenceRefusesToStart`；单元层由 `test/trust_tokens_unittest.cc:38,44` 两处 `EXPECT_STREQ(st.reason, kScrapeTokenCollision)` 钉住（`reason` 定义在 `src/control/trust_tokens.h:22`、判据在 `:37`），网关侧据此在 `src/control/gateway/gateway_app.cpp:941-946` 打印 "refusing to start: MORTRED_METRICS_TOKEN matches an inference or management token" |
| OP5 | **部分**：gateway 先于模型退出、模型进程不残留这一半已由 P1-1 轮次前的有序停机用例覆盖（`supervisor_unittest` 的 `shutdown_is_ordered_and_complete`），但"模型进程先退出"整条链要等 **P1-3（进程组）合入后**按 `process_group_gone` 重判；本行在 P1-3 关闭后补最终判定 | 待补 |

## 仍待判定的行

下列各行的判定依赖尚在合入中的条目或需要 GPU/压测证据，收口时逐行补齐：

- **PF1–PF5、TP1–TP2**：需要 TensorRT engine 的 calibrate + `http_infer_rps.py --raw` 压测（并发 = 已部署 `worker_nums` × 2、≥30s）、`docs/benchmark-h3/<id>.txt` 的 trtexec 对照、gateway 与模型分段耗时、以及 85%/150% 的复测阈值。本轮 7 个 P1 均不产出这类证据 → 收口时按 `rubric.md:91` 判"阻塞"（engine/性能行）或"未满足"，并写明缺的环境与恢复后要重跑的验收。
- **PF3**：N>1 张图片一次提交要返回 N 条逐项结果且 batch 观测为 1 —— 属服务行为而非纯性能，收口时用既有 e2e 证据判定（`server_e2e_contract_test` 的多图用例）。
- **AV1**：`type=mnn`/`type=onnx` 的已服务 id 能单独拉起并完成一次合法推理 —— 本机缺 MNN/ONNX 权重时相关用例 SKIPPED（P1-4 QA 已记录 `backend_unittest` 5 例 SKIPPED），收口时按"阻塞"或"未满足"据实判定。
- **ST1–ST4、ST6、TP3、UX1–UX4、OP1–OP3、EX1–EX3**：收口时逐行给出证据或未满足理由；其中 `UX3`（CI 检查 OpenAPI 一致性、GPU 作业被跳过不得写成 TensorRT 证明）与 `OP1–OP3`、`TP3` 的相关证据正在由 `07-modules/ci.md`、`07-modules/ops.md` 的评分调用收集。

## 测试面测量口径（写 `test` 模块记录前必读）

编排方在基线 `0918361a` 上实测了注册测试数与"二进制是否存在"：

| 构建目录 | 注册测试数 | 二进制存在 | 二进制缺失 |
| --- | --- | --- | --- |
| `build`（tests-only，主检出） | 91 | 34 | 57 |
| `build/full-cpu`（BUILD_FULL=ON） | 96 | 35 | 61 |
| `build/probe-cpu` | 95 | 34 | 61 |
| `build/probe-ci`（CI tests-only 口径） | 63 | 33 | 30 |

**口径警告（防止误判，也防止我自己再次over-claim）**：上表"二进制缺失"**不等于"无法构建"**。这四个目录都是**部分构建**的产物——本轮各条目的 QA 只构建并运行了各自点名的目标，从未把每个测试目标都构建一遍。因此"缺失"主要表示"在该目录里没构建过"，**不能**据此写成"57/93 个用例从未运行"这类结论。

可以确证**无法构建**的只有三个具名目标：`packed_batch_nvi_unittest`（`link.txt` 缺 `libmodels.so`）、`cv_image_input_unittest`（同类链接失败）、`enlightengan_output_contract_unittest`（缺 `model_runtime.h` 的 include 链）。这三者连带使 CI 的 `check` 聚合目标中断（`test/CMakeLists.txt:348` 对 `TEST_LIST` 每一项都 `add_dependencies(check ${src})`），属用户已移出范围的问题，只作留档。

本轮的**可运行**证据是另一回事，且是硬的：编排方在 7 个条目上实际跑过约 20 个不同测试二进制（`request_envelope_unittest` 17、`server_e2e_contract_test` 45→47、`gateway_e2e_test` 49、`detection_params_unittest` 8、`detector_common_unittest` 15、`object_detection_output_contract_unittest` 13、`prometheus_metrics_test` 7→11、`tensor_contract_unittest` 14、`http_contract_test` 6、`model_output_contract_unittest` 10、`backend_unittest` 3+5skip、`openai_clip_output_contract_unittest` 6、`lightglue_output_contract_unittest` 15、`server_runtime_config_unittest` 5→6、`supervisor_unittest` 24、`supervisor_malformed_e2e_test` 1、`request_admission_unittest` 5、`log_buffer_unittest` 1 等），全部退出码 0，SKIP 项均有明确原因（权重缺失）。

## 加权分

- 模块分与权重见 `07-modules/*.md`（本文件只做复核与上限套用）。
- 计算规则（`rubric.md:69`）：原始分 = Σ(模块分 × 权重) / 100，保留一位小数；再套上限（`rubric.md:182`）：有未关闭且未阻塞的 P0 → ≤7.9；只有 P1 → ≤8.9；阻塞项不触发上限。
- 本轮起点队列：P0 = 无，P1 = 7 条（P1-1…P1-7），P2 = 见 `02-p0-p1-final.md`。
- 收口时按"已关闭 / 未关闭且未阻塞 / 阻塞"三类逐条归位后套用上限。

### 已评模块与暂定加权分（7/9，待 `src-control`、`test` 补齐）

| 模块 | 权重 | 分数 | 乘积 | 记录 |
| --- | --- | --- | --- | --- |
| `src-server-apps` | 20 | 7.0 | 140.0 | `07-modules/src-server-apps.md` |
| `src-models` | 15 | 7.5 | 112.5 | `07-modules/src-models.md` |
| `ops` | 15 | 8.0 | 120.0 | `07-modules/ops.md` |
| `src-factory-common` | 10 | 8.0 | 80.0 | `07-modules/src-factory-common.md` |
| `conf` | 5 | 7.5 | 37.5 | `07-modules/conf.md` |
| `docs` | 5 | 8.0 | 40.0 | `07-modules/docs.md` |
| `ci` | 5 | 5.0 | 25.0 | `07-modules/ci.md` |
| **小计** | **75** | — | **555.0** | 尚有 25 权重未评 |
| `src-control` | 15 | 待评 | — | 被 P1-3 占着，合入后评 |
| `test` | 10 | 待评 | — | 被 P1-3 占着，合入后评 |

- 公式代入：最终原始分 = (555.0 + 15 × `src-control` + 10 × `test`) / 100。
- 供参照的量级判断（**非结论**）：若 `src-control` 落在 7.5、`test` 落在 7.0，则原始分 = (555.0 + 112.5 + 70.0) / 100 = **7.375 → 7.4**。
- 上限核验：若 P1-3 关闭后仍存在未关闭且未阻塞的 P1（例如 ST4 的未满足行按 `rubric.md:95` 入列后形成的条目），上限 8.9 不构成约束；7.4 远低于该上限，说明本轮分数的瓶颈是模块本身的缺陷（`ci` 的两道红门、`src-server-apps` 的 gauge 缺陷等），而不是队列未清空。

## 终态

按 `rubric.md:186`，`done` 需三件同时成立：完成标准每一行"已满足" + 九份模块开放项均为"无" + 阻塞项为空。就当前证据看，PF1–PF5/TP1–TP2 这类需要 GPU 压测的行本轮不可能满足或只能判阻塞，因此**终态不会是 `done`**；收口时按 `rubric.md:188-190` 在 `running` 与 `awaiting-env` 之间据实选择，并写明恢复后要重跑哪些行。
