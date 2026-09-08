# base_server_impl.h 拆分技术方案

> **来源声明**：本方案来自 **GLM-5.3-MAX** 的修改意见（2026-09-09，基于对当前
> `src/server/base_server_impl.h` @1,936 行版本的逐行审读），供项目维护者评估与执行。

## 1. 现状诊断：为什么说它是上帝类

`BaseAiServerImpl<WORKER, MODEL_OUTPUT>` 目前同时承担以下职责（行号为当前版本参考）：

| # | 职责 | 代表代码 | 依赖 |
|---|---|---|---|
| 1 | 配置解析与校验 | `parse_common_server_config` / `parse_server_security_config` / `parse_worker_nums` | toml |
| 2 | HTTP 线路原语 | `peer_ip_of` / `header_value_of` / `is_json/raw_content_type` | workflow |
| 3 | 响应序列化 | `reply_json` / `reply_unified_json` / `unified_rejection` / `reply_status` | workflow + envelope |
| 4 | 准入策略 | 415/413/条数上限/背压 429+Retry-After/限流与鉴权检查 | 纯逻辑 + workflow |
| 5 | 异步任务 HTTP 端点族 | `handle_async_{jobs,submit,status,wait,result}` + waiter 注册表 | workflow + AsyncJobTable |
| 6 | 异步任务执行 | `async_run_job` | worker 池 |
| 7 | 同步执行编排 | `do_work` / `run_items` / `aggregate_item_statuses` | worker 池 |
| 8 | go task 生命周期 | `go_task_functor` / `do_work_cb` / `request_meta` | workflow |
| 9 | 动态批处理 | `batch_entry` / `request_state` / `batch_loop` / `process_batch` / `complete_batch_item` | 自持线程 |
| 10 | 过载仪表 | `update_run_time_ewma` / 卡死检测（次数×跨度） | 纯逻辑 |
| 11 | 生命周期排水 | 析构中 inflight 计数 + worker 队列水位 | 全部 |

量化指标：**约 37 个 `_m_` 数据成员、48 处 `_m_metrics` 调用、50+ 个成员函数**，
单个文件 1,936 行。11 类职责、两个模板参数——按任何常见标准（职责数 > 10、
字段内聚性、修改原因数）都已构成上帝类。

**公允说明**：这是"纪律良好的臃肿"——内部不变式注释（go task 所有权、析构
排水次序、卡死判定跨度）质量很高，拆分时这些论证必须原样随代码迁移，不允许
稀释。本方案的目标不是"打散"，而是**把不需要相互知道的职责隔开**。

## 2. 拆分总览：6 个新模块 + 1 个瘦身编排器

遵循项目已有且验证过的拆分先例（`AsyncJobTable` 从本类剥离时的原则：
*"subordinate component that owns ONLY its bookkeeping, zero dependencies on
Workflow HTTP / worker pool / metrics — which is what makes it directly
unit-testable"*）。

```
拆分前                                    拆分后
─────────────────────────────            ──────────────────────────────────────────
base_server_impl.h (1936 行, 11 职责)     http_wire.h            HTTP 原语 + 应答基础件
                                          server_runtime_config.h TOML → 配置结构体
                                          admission.h            准入决策纯函数
                                          worker_health.h        EWMA + 卡死检测状态机
                                          batch_collector.h      动态批收集器（自持线程）
                                          async_endpoints.h      /jobs 端点族 + waiter 注册表
                                          base_server_impl.h     纯编排（目标 <700 行）
```

## 3. 模块明细

### 3.1 `src/server/http_wire.h` —— HTTP 线路原语与应答基础件

**内容**：`peer_ip_of`、`header_value_of`、`authorization_header_of`、
`is_json_content_type`、`is_raw_body_content_type`、`reply_json`、
`reply_unified_json`、`unified_rejection`、`reply_status`、`reply_async_error`、
`generate_req_id`、`monotonic_ms`。

**为什么拆**：
- 它们是**模板类的 static 成员**——今天每个 `(WORKER, MODEL_OUTPUT)` 实例化
  都会复制一份完全相同的机器码（generic_cv_server 的每个模型 + 两个测试夹具
  都在实例化）。改为非模板自由函数后，链接器只保留一份，编译产物体积与
  编译时间双降，这是本方案唯一的性能红利；
- 依赖面（workflow + envelope）与业务零交叠，是最干净的 seam。

### 3.2 `src/server/server_runtime_config.h` —— 配置结构体与解析

**内容**：`ServerRuntimeConfig` 聚合结构（连接/线程/超时/限流/鉴权/批处理/
async/stuck 共约 20 个字段）+ `parse_server_runtime_config(toml) -> config` +
`parse_worker_nums`。BaseAiServerImpl 持有一个值对象，不再直接触碰 toml。

**为什么拆**：
- 把"从 TOML 读 20 个键 + 默认值 + 告警"从类里拿走后，编排器只见结果不见
  解析过程；schema 校验（`server_config_schema.h`）与解析同居一文件，未来
  改配置面只动一个文件；
- 解析 → 结构体的映射可以独立做表驱动单测（键缺失/类型错/非法值各回什么），
  目前这类测试要起整个 server 才能覆盖。

### 3.3 `src/server/admission.h` —— 准入决策纯函数

**内容**：`AdmissionDecision {verdict, status, retry_after_s, violations}` 与
`admit_request(content_type, content_length, item_count, waiting, ewma, workers,
limits) -> AdmissionDecision`——覆盖现有 415/413/条数上限/背压 429 路径的判定
逻辑（Retry-After 算术复用 `backpressure.h`）。

**为什么拆**：
- 准入规则是产品契约（HTTP 状态码映射），却是当前最难单测的部分——逻辑
  内嵌在 280 行的 `serve_process` 中间。纯函数化后，`http_contract_test`
  的状态码断言可以在毫秒级单测里穷举边界；
- 未来网关限流（评审长期项）落地后，边缘与服务端的准入规则需要对照维护，
  现在先给它们一个共同的可读形态。

### 3.4 `src/server/worker_health.h` —— 过载与卡死仪表

**内容**：`RunTimeEwma`（CAS 更新的 EWMA）+ `StuckWorkerDetector`
（"连续次数 × 时间跨度"双条件状态机，含 LOG/EXIT 动作）。

**为什么拆**：
- 卡死检测的判定条件（并发等待重叠导致纯计数会说谎，必须次数与跨度同时
  达标）是本文件里最精妙也最容易被误改的逻辑，值得一个独立文件 + 专属
  单测（假时钟注入）锁定；
- `do_work` 与 `process_batch` 两处调用同一状态机，拆出后消除现有的
  重复实现（两处几乎相同的 20 行判定代码）。

### 3.5 `src/server/batch_collector.h` —— 动态批收集器

**内容**：`BatchCollector<WORKER, MODEL_OUTPUT>` 持有
`batch_queue`/`batch_thread`/`batch_entry`/`request_state`/`batch_loop`/
`process_batch`/`complete_batch_item`，对外仅暴露
`submit(task_request shared state)`、`start()/stop()` 与配置。

**为什么拆**：
- 它是唯一**自持线程**的子组件，与 HTTP 编排没有任何共享状态以外的耦合；
  独立成类后，其"请求超时后 late completion 落入弃置状态"的内存所有权
  论证（shared_ptr per-entry）整体迁移，不动一个字；
- 停机次序（先停 collector、失败在队条目、再排 worker）在编排器析构里
  变成一行显式调用，比现在散在 20 行注释里的隐式约定清晰。

### 3.6 `src/server/async_endpoints.h` —— 异步任务端点族

**内容**：`AsyncEndpoints<MODEL_OUTPUT>` 持有 `/jobs` 六个 handler、
waiter 注册表（`register/unregister/wake`）、`parse_wait_timeout_ms`、
`reply_async_status`，以及计数器唤醒所需的 `async_inflight`/`wait_inflight`
原子量；结果组装通过注入的 `fill_response` 回调完成。

**为什么拆**：
- 端点族 ~400 行是本文件最大的一块，且只被 `serve_process` 的一个分支
  引用——最典型的"整块搬移"；
- waiter 注册表（命名计数器唤醒机制，R8 的核心改造）值得与
  `async_job_table_unittest` 同级的独立测试，目前它藏在类私有区不可达。

### 3.7 `src/server/base_server_impl.h`（瘦身后）—— 唯一编排器

**保留**：`serve_process` 端点分发序、`do_work`/`run_items`/
`aggregate_item_statuses`/`go_task_functor`/`do_work_cb`/`request_meta`
（**go task 生命周期族明确不拆**——所有权论证横跨这三个函数与析构排水，
拆散会把一段完整的生命周期证明摊到多个文件，是负优化）、析构排水、
worker 池、对六个子模块的组合与委托。

**预期**：约 600–700 行、`_m_` 成员从 37 降到 ~15、修改原因从 11 个收敛到
3 个（编排策略 / 生命周期 / 子模块接线）。

## 4. 拆分后的文件组织形式

```
src/server/
├── abstract_server.h            # 不变：WFHttpServer 组装 + start/stop 门面
├── base_server_impl.h           # 瘦身：编排 + go task 生命周期 + 排水（<700 行）
├── http_wire.h                  # 新：HTTP 原语/应答基础件（非模板自由函数）
├── server_runtime_config.h      # 新：TOML → ServerRuntimeConfig（含 schema 校验接线）
├── admission.h                  # 新：准入决策纯函数（415/413/429/Retry-After）
├── worker_health.h              # 新：RunTimeEwma + StuckWorkerDetector
├── batch_collector.h            # 新：BatchCollector<WORKER, MODEL_OUTPUT>
├── async_endpoints.h            # 新：/jobs 端点族 + waiter 注册表
├── inference_task.h             # 已有：InferenceTask/InferenceResult
├── parsed_request.h             # 已有：信封 → 绑定
├── async_job_table.h            # 已有：异步账本
├── backpressure.h               # 已有：Retry-After 算术（admission.h 复用）
├── rate_limiter.h / prometheus_metrics.h / http_status.h / ...   # 不变
└── generic_cv_server.h          # 不变：CvServerSpec → CvModelServer 装配
```

依赖方向（只允许向下）：

```
generic_cv_server.h
   └─ base_server_impl.h（编排）
        ├─ batch_collector.h ──┐
        ├─ async_endpoints.h ──┼─ inference_task.h / async_job_table.h
        ├─ admission.h ────────┼─ backpressure.h
        ├─ worker_health.h ────┘
        ├─ server_runtime_config.h（toml）
        └─ http_wire.h（workflow）
纯逻辑层（admission / worker_health / backpressure / 账本）保持
workflow-free，tests-only CI 可直接单测。
```

## 5. 迁移计划（四步，每步独立可合入、行为零变化）

| 步骤 | 内容 | 安全网 |
|---|---|---|
| 1 | 抽 `http_wire.h` + `server_runtime_config.h`（纯搬移，机械改写调用点） | `server_e2e_contract_test` 42 用例全绿、不改动 |
| 2 | 抽 `worker_health.h`（消灭 do_work/process_batch 的重复判定）+ `admission.h`（判定逻辑先以纯函数镜像实现，编排器改为调用） | 状态码单测（新增）+ e2e |
| 3 | 抽 `batch_collector.h`（含 `request_state`/`batch_entry` 整体迁移） | e2e 批处理用例 + `do_work_lifetime_unittest` |
| 4 | 抽 `async_endpoints.h`（waiter 注册表随迁并补专属单测） | e2e async 全生命周期用例 + 新增 waiter 单测 |

每步之后主文件行数单调下降、公共 API（`BaseAiServer` 门面、
`CvServerSpec` 装配、HTTP 契约）保持字节级不变——**调用方（factory/
generic_cv_server/全部测试）零改动**是验收的硬条件。

## 6. 验收度量

- `base_server_impl.h` < 700 行，`_m_` 成员 ≤ 15；
- 新模块单测进入 tests-only CI（workflow-free 模块不依赖 vendored workflow）；
- 全套 e2e/契约测试零改动通过；TSAN/ASAN 门禁维持绿；
- 编译产物符号数下降（http_wire 去模板化的直接收益），`full` preset
  编译时间有可测改善；
- 六个月后回看：修改准入规则只动 `admission.h`、调批处理只动
  `batch_collector.h`——上帝类的最终判据是"改一处只开一个文件"。
