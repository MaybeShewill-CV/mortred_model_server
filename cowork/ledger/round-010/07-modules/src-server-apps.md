# 模块：src-server-apps

- 路径: `src/server`、`src/apps`
- 权重: 20
- 分数: 7.0
- 基线: sub-agent-co-work @ `b06aca30`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **规模**：`src/server` **22 文件 / 5,058 行**；最大的五个文件是 `base_server_impl.h`(781)、`batch_collector.h`(512)、`async_endpoints.h`(458)、`async_job_table.h`(405)、`prometheus_metrics.h`(393)，其后是 `response_serializers.h`(270)、`server_runtime_config.h`(224)。
- **`src/apps` 是薄壳而非业务**：`control/gateway.cpp` 14 行、`control/mortredctl.cpp` 14 行、`control/supervisor.cpp` 24 行、`model_server_main.cpp` 94 行、`common/model_server_main.h` 110 行、`product_index.cpp` 239 行——即"一个二进制服务 44 个 id"在应用层也是成立的（与 EX2 的 `--list` 活体证据一致）。
- **契约测试面厚且本轮有实跑**：`http_contract_test` **6/6**（含 `unified_contract_codes_map_to_http`、`server_errors_map_to_5xx`、`MODEL_OUTPUT_CONTRACT_FAILED → 500`）、`server_e2e_contract_test` **45/45 → 47/47**（P1-7 与 P1-5 两次重跑）、`server_runtime_config_unittest` **5/5 → 6/6**、`prometheus_metrics_test` **7/7 → 11/11**、`request_admission_unittest` **5/5**、`log_buffer_unittest` **1/1**——全部退出码 0。
- **本轮在此模块完成的改造**：P1-6（`prometheus_metrics.h` 新增 `normalize_http_method`，counter 与 histogram 两处入口统一归一化，未知/含 `'|'`/控制字符/超长一律 `other`）、P1-5（`server_runtime_config.h` 的 `model_run_timeout <= 0` 夹到 300000ms 并告警）、P1-7（`server_config_schema.h` 的 `rate_limit_qps` 语义注释）。**`src/server` 内 `TODO/FIXME` 计数为 0**。
- **异步与同步路径分开且各自有 429 语义**：`async_endpoints.h`( `:273`、`:328` 的队列满 429)、`async_job_table.h`、`batch_collector.h`、`sync_request_graph.h`、`item_exec.h`；同步路径的队列门在 `base_server_impl.h:462-470`（`queue_would_overflow` + `Retry-After`），由 `queue_limit_returns_429_with_retry_after` 与 `async_queue_full_returns_429` 覆盖。
- **指标收口在 collector**：`/metrics` 的写入口只有 `prometheus_metrics.h` 的公开方法（`inc_http_requests`/`observe_http_duration_ms` 等），`src/server` 内 16 处调用点全部经它；两个 map 为 `private`，外部无法绕过——P1-6 的归一化因此是单点收口。

## 问题

1. **`mortred_waiting_jobs` gauge 永不回落（N-1，P1 候选）**：该 gauge 只在 `reply_sync_request` 内被 `set_waiting_jobs(_m_waiting_jobs.load())` **快照**（`base_server_impl.h:774-775`），而递减发生在 `schedule_sync_request` 注册的 series callback（`:557-560`，**快照之后**）。P1-5 的实现者实测轮询 40 次 × 3s 仍读到 1；P1-5 的 QA 独立复核源码后同意该机制。影响：运维无法从 `/metrics` 判断队列是否排空，`mortred_queue_depth` 同理（setter 为直存、无按需重算，全仓无第二处调用）。
2. **非法 base64 的 violation 指针是压缩下标（N-3，P2 候选）**：`parsed_request.h:130-131` 遍历的是 `decoded.value.images`（只含通过类型检查的字符串），因此 `{"images":[5,"ok"]}` 会报 `/images/0` 而用户看到的元素下标是 1——指针会指错元素。
3. **`status` 标签未做白名单（P1-6 的 QA 残余）**：`method` 已归一化，但 `status` 仍是原样拼接。当前**不可达**（所有调用点传字面量或 `std::to_string(int)`，QA 逐个核对过），但若将来有调用点把请求侧原始串当 status 传入，`'|'` 伪造标签对的问题会以同样方式复活。
4. **请求路径分散在 780+512+458 行三个头文件里**：`base_server_impl.h` 同时承担准入、超时、指标与回复，`batch_collector.h` 承担批处理编排，`async_endpoints.h`/`async_job_table.h` 承担异步作业——"看清一次请求的时间构成"需要在 4 个头文件间跳转。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| ST3（队列打满后新请求在超时前收到 429 + `Retry-After`；拒绝结束后后续合法请求仍能成功且进程回到 ready） | **已满足** | `queue_limit_returns_429_with_retry_after`（`server_e2e_contract_test.cc:937-965`：断言 429、`retry-after ∈ [1,60]`、`mortred_queue_rejected_total > 0`）、`async_queue_full_returns_429`（`:1306`）；"拒绝结束后仍能成功"由 P1-5 新用例在 `max_queue_depth=1` 下以"第二次请求 200 而非 429"独立证明；"回到 ready"由 `supervisor_unittest` 的重启用例承担 |
| ST1 的一半（HTTP 侧的错误码映射） | **已满足** | `http_contract_test` **6/6**（`unified_contract_codes_map_to_http`、`server_errors_map_to_5xx`、`MODEL_OUTPUT_CONTRACT_FAILED → 500`），退出码 0 |
| PF2（同一次压测要能读到模型进程 `/metrics` 里分开的排队等待与推理耗时，且排队等待不得被算进推理耗时） | **阻塞** | 需要 PF1 的压测（TRT calibrate + `http_infer_rps.py --raw`）；且**当前该读数的可信度本身存疑**——排队等待用的 `mortred_waiting_jobs`/`mortred_queue_depth` 正是问题 1 那个不刷新的 gauge。**恢复条件**：先修问题 1（或改用 counter 差值读排队量），再在压测中同时读两段耗时 |
| ST2（鉴权失败不得消耗限流配额；未鉴权不得调推理路由） | **不属本模块**（判定见 `07-score-check.md` 的 ST2 行） | 实现主体在 `src/control/gateway`（401 路径上 `ip_limiter_->release(...)`），模型侧的对应面是 `base_server_impl.h:330-342` 的鉴权豁免与 `:343-345` 的 `(void)_m_rate_limit_qps`；端到端证据为 `server_e2e_contract_test.cc:733` 与 gateway 侧用例 |
| PF3（N>1 张图须返回 N 条逐项结果且 batch 观测为 1） | **已满足** | `server_e2e_contract_test` 47/47（含多图与逐项结果用例）；批次内单条失败由 `backend_cv_model.h:208-220` 广播 `item_status`，不再 abort 整进程（P1-4 合入后） |

## 评分依据

- 给 7.0 而不是更高：① 问题 1 是**本模块内的真实可观测性缺陷**（P1 级，已被本轮列为 N-1），它让"队列是否排空"这个最常用的运维读数失真，并且恰好落在完成标准 PF2 的判据上；② 问题 2 会让用户在非法请求上收到**指错元素**的指针，属对外契约的准确性缺陷；③ 781+512 行的两个头文件把准入、超时、指标、回复、批处理编排混在一起，与 `rubric.md` 里"看清一次请求的时间构成"的目标直接冲突；④ `status` 标签的白名单缺失是一个"当前不可达但会随新调用点复活"的敞口。
- 给 7.0 而不是更低：① HTTP 契约测试面厚且本轮全部实跑通过（6/6、47/47、6/6、11/11、5/5、1/1）；② 指标写入口单点收口（两个 map 为 private，16 处调用点全部经 collector），P1-6 的归一化因此能一处生效；③ 应用层是薄壳（14/14/24 行的 control 三主程序 + 94 行的 model_server_main），"加 id 不加二进制"在应用层成立；④ `src/server` 内 `TODO/FIXME` 为 **0**；⑤ 同步/异步两条路径的 429 语义各自有测试固定。
- 参照系：修掉问题 1（或把排队量改成 counter 差值）并把 `base_server_impl.h` 的职责拆开之后，本模块有理由到 8.0 以上；`status` 白名单化是廉价的加分项。
