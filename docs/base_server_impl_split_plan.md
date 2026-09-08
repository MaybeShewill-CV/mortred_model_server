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

---

# base_server_impl.h 拆分技术方案（Grok 4.6 Extra High）

> **来源声明**：本方案来自 **grok4.6-extra-high** 的修改意见（2026-09-09，基于对当前
> `src/server/base_server_impl.h` @1,939 行版本的逐行审读，并对照仓库其余
> serving / control / factory 分层），供项目维护者评估与执行。
>
> 上一节是 GLM-5.3-MAX 的方案。本节是另一条切割：模块数量接近，但边界按
> **所有权与修改原因** 划分，而不是按「把能看见的函数簇各塞进一个文件」。

## 1. 现状诊断：为什么说它是上帝类

`BaseAiServerImpl<WORKER, MODEL_OUTPUT>` 目前同时承担以下职责（行号为当前版本参考）：

| # | 职责 | 代表代码 | 依赖 |
|---|---|---|---|
| 1 | 配置解析与校验 | `parse_common_server_config` / `parse_server_security_config` / `parse_worker_nums` | toml；解析函数里还会 `start` batch 线程 |
| 2 | HTTP 线路原语 | `peer_ip_of` / `header_value_of` / `is_json/raw_content_type` | workflow |
| 3 | 统一信封应答 | `reply_unified_json` / `unified_rejection` / `reply_status` | workflow + envelope |
| 4 | 准入策略 | 415/413/条数上限/背压 429+Retry-After；鉴权与 QPS 在 `serve_process` 入口 | 纯逻辑 + workflow |
| 5 | 异步任务 HTTP 端点族 | `handle_async_{jobs,submit,status,wait,result}` + waiter 注册表 | workflow + AsyncJobTable |
| 6 | 异步任务执行 | `async_run_job` | worker 池（与同步路径共用） |
| 7 | 同步执行编排 | `do_work` / `run_items` / `aggregate_item_statuses` | worker 池 |
| 8 | go task 生命周期 | `go_task_functor` / `do_work_cb` / `request_meta` | workflow |
| 9 | 动态批处理 | `batch_entry` / `request_state` / `batch_loop` / `process_batch` | 自持线程 + 第二份取 worker |
| 10 | 过载仪表 | `update_run_time_ewma` / 卡死检测（次数×跨度） | 纯逻辑，但嵌在两处 checkout 里 |
| 11 | 生命周期排水 | 析构中 inflight 计数 + worker 队列水位 | 全部 |

量化指标：**约 39 个 `_m_` 数据成员、48 处 `_m_metrics` 调用、约 44 个成员函数**，
单个文件 1,939 行。11 类职责、两个模板参数——按职责数、字段内聚性、修改原因数
都已构成上帝类。

**公允说明**：这是"纪律良好的臃肿"。go task 所有权、析构排水次序、卡死判定
跨度这些注释必须原样随代码迁移。仓库其余部分已经分层（`src/control` 控制面、
`generic_cv_server.h` 把 N 个手写 Server 收成 `CvServerSpec`、`AsyncJobTable`
只做账本）。上帝类就剩这一层 serving runtime。本方案的目标不是打散，而是
**把不需要相互知道的职责隔开**，并且让「取 worker」只存在一份协议。

与「只按函数列表切文件」相对，下面几条是当前文件里真正的错误边界：

- `parse_common_server_config` 既读 TOML 又 `start` batch 线程——解析不该有副作用。
- 卡死检测在 `do_work` 与 `process_batch` 里近乎各复制 20 行。根因不是「仪表
  没独立成文件」，而是 **checkout 协议有两份**。
- `handle_async_submit` 把信封校验、条数上限、422/413 与同步路径各写一遍。
- `reply_async_error` 的 JSON 形状不是统一信封，不能和 `reply_unified_json` 当成
  同一类 HTTP 原语。
- `monotonic_ms` 与 `AsyncJobTable` 里的 `async_now_ms` 已经是两份时钟。
- 五个连接参数是 `public`，只为了 `abstract_server.h` 去抠 `_m_compute_threads`
  等字段；拆配置时第一步仍要填充这些字段，避免门面跟着炸。
- 模板类里的 static HTTP 助手每个 `(WORKER, MODEL_OUTPUT)` 实例化都复制一份。

## 2. 拆分总览：6 个新模块 + 1 个瘦身编排器

遵循项目已有且验证过的拆分先例（`AsyncJobTable` 从本类剥离时的原则：
*"subordinate component that owns ONLY its bookkeeping, zero dependencies on
Workflow HTTP / worker pool / metrics — which is what makes it directly
unit-testable"*）。

相对上一节 GLM 方案，本方案 **模块数量相同（6 个新文件），切割不同**：

1. **不抽独立的 `worker_health.h`**。EWMA 和卡死状态机没有自己的生命周期，
   它们是「从队列取出 / 归还 worker」协议的一部分。抽出仪表却把队列留在编排器，
   治标不治本。改为 `worker_pool.h`：队列 + deadline checkout/checkin + EWMA +
   卡死。`do_work`、`async_run_job`、`process_batch` 走同一条取还路径。
2. **不抽独立的 `admission.h` 巨型 `admit_request()`**。415、413、条数上限、
   队列 429 的输入完全不同，合成一个 `AdmissionDecision` 只会把 `serve_process`
   的分支藏进结构体。准入做成若干正交纯函数，扩写已有 `backpressure.h` 的姊妹
   文件 `request_admission.h`（Retry-After 仍由 `backpressure.h` 计算）。
3. **`http_wire.h` 不收 `reply_async_error`**。那是 `/jobs` 专用的非信封 JSON，
   跟统一信封应答不是一类东西，随 async 端点族走。
4. **`server_runtime_config.h` 不合并 `server_config_schema.h`**，也不启动线程。
   schema 已经是独立、可单测的模块；解析只调用它。batch 线程由编排器在拿到
   配置之后显式 `start()`。
5. **`async_run_job` 留在编排器**。它用的是 worker 池和 `run_items`，不是 HTTP。
   端点族只负责 `/jobs` 的解析、202、轮询、wait、result 与 waiter 注册表。

```
拆分前                                    拆分后
─────────────────────────────            ──────────────────────────────────────────
base_server_impl.h (1939 行, 11 职责)     http_wire.h              HTTP 原语 + 统一信封应答
                                          server_runtime_config.h TOML → 配置值对象（无副作用）
                                          request_admission.h     准入纯函数（正交判定）
                                          worker_pool.h            队列 + checkout + EWMA + 卡死
                                          batch_collector.h        动态批收集器（自持线程）
                                          async_endpoints.h         /jobs HTTP + waiter + inflight 排水
                                          base_server_impl.h       纯编排（目标 <700 行）
```

计数：**6 个新模块 + 1 个瘦身编排器**。其中 `request_admission.h` 是小文件
（正交纯函数，Retry-After 仍调用已有 `backpressure.h`，不把算术再搬一次），
但必须独立存在，否则同步 POST 与 `handle_async_submit` 的重复准入消不掉。

## 3. 模块明细

### 3.1 `src/server/http_wire.h` —— HTTP 线路原语与统一信封应答

**内容**：`peer_ip_of`、`header_value_of`、`authorization_header_of`、
`is_json_content_type`、`is_raw_body_content_type`、`reply_unified_json`、
`status_envelope`、`unified_rejection`、`reply_status`、`reply_unauthorized`、
`reply_rate_limited`、`generate_req_id`。全部是**非模板自由函数**。

**不放进这里**：`reply_async_error`（async 专用 JSON）、`monotonic_ms`（不是 HTTP；
worker 池与 async 账本已有 / 将有自己的单调时钟）。

**为什么拆**：
- 它们是模板类的 static 成员——今天每个 `(WORKER, MODEL_OUTPUT)` 实例化都会
  复制一份完全相同的机器码（`generic_cv_server` 的每个模型 + 测试夹具）。改为
  自由函数后链接器只保留一份，这是本方案唯一确定的编译产物红利；
- 依赖面（workflow + envelope）与 worker 池、批处理、账本零交叠，是最干净的 seam。

### 3.2 `src/server/server_runtime_config.h` —— 配置结构体与解析

**内容**：`ServerRuntimeConfig` 值对象（连接/线程/超时/限流/鉴权/批处理/async/stuck
共约 20 个字段）+ `parse_server_runtime_config(toml) -> expected<config, StatusCode>`
+ 现有 `parse_worker_nums`（从 `base_server_impl.h` 迁入，名字与命名空间不变）。
内部调用已有 `validate_server_section`（`server_config_schema.h`），**不把 schema
合并进本文件**。

**硬约束**：解析是纯函数。`max_batch_size > 1` 时**不得**在这里 `std::thread`。
编排器在 `init` 成功路径上调用 `batch_collector.start()`。

**为什么拆**：
- 编排器只见结果不见「读 20 个键 + 默认值 + 告警」；改配置面只动这一个文件 +
  已有 schema；
- `parse_worker_nums` 已有 `worker_nums_unittest`，迁过去后该单测改 include 即可，
  不必再编译整个模板类；
- `BaseAiServerImpl` 仍把连接五元组写回现有的 public `_m_max_connection_nums` /
  `_m_compute_threads` 等字段，这样 `abstract_server.h::init_http_server` 第一步
  不用改。后续若要收掉这层数据泄漏，再给门面一个 `http_params()`，不作为本拆分
  的准入条件。

### 3.3 `src/server/request_admission.h` —— 准入判定纯函数

**内容**：若干**正交**自由函数，而不是一个吞掉所有分支的 `admit_request()`：

- `content_type_kind(content_type) -> json | raw | unsupported`（对应 415）
- `declared_body_exceeds(content_length, limit_mb)`（对应 413）
- `item_count_exceeds(n, max_request_items)`（对应 413 / `REQUEST_ITEM_LIMIT`）
- `queue_would_overflow(waiting, n_items, max_queue_depth)`（对应 429）
- Retry-After 继续调用已有 `compute_retry_after_seconds`（`backpressure.h`）

返回值是简单的 bool / enum，**不含** `WFHttpTask*`、**不写** metrics、**不组**
应答。`serve_process` 与 `handle_async_submit` 都调用同一组判定，各自负责
回写信封和计数。这是同步 / async 准入重复的正确消解方式。

**为什么拆**：
- 准入规则是产品契约，却嵌在 230 行 `serve_process` 和另一份 async submit 里，
  是当前最难单测的部分。纯函数化后可在 tests-only CI 穷举边界，不必起 HTTP 服务器；
- 巨型 `AdmissionDecision` 会把「判什么」和「如何回复」重新焊死，编排器读完还是
  一串 if。正交函数让 `serve_process` 的控制流仍然可读；
- 本文件保持 workflow-free。

### 3.4 `src/server/worker_pool.h` —— worker 队列与取还协议

**内容**：`WorkerPool<WORKER>` 持有

- `BlockingConcurrentQueue<WORKER>` 与 watermark（`_m_worker_nums`）
- `checkout(WORKER&, timeout) -> bool`：deadline / 无限等待；失败时推进
  「连续次数 × 时间跨度」卡死状态机（LOG / `LOG(FATAL)` EXIT）
- `checkin(WORKER&&)`：归还；析构排水等的就是 `available_approx() == watermark`
- `update_run_time_ewma` / `ewma_ms()`：Retry-After 的样本源
- 单调时钟（消化当前的 `monotonic_ms`；async 账本继续用 `async_now_ms`，不在
  HTTP 层再放第三份）

`do_work`、`async_run_job`、`BatchCollector::process_batch` **只通过这对
checkout/checkin 碰队列**。卡死判定从两处 20 行复制变成状态机的一处实现。

**为什么拆**：
- 这是相对「只抽 `StuckWorkerDetector`」更合理的一刀。仪表没有独立生命周期，
  重复代码的根因是取 worker 的协议被写了两次；
- 卡死判定（并发等待重叠导致纯计数说谎，必须次数与跨度同时达标）值得专属
  单测（假时钟注入），但测试的是 `checkout` 失败路径，不是一个游离的计数器；
- 编排器析构里的 worker 水位等待变成 `pool.drain()`，与 batch / async 的停机
  接口对称。

### 3.5 `src/server/batch_collector.h` —— 动态批收集器

**内容**：`BatchCollector<WORKER, MODEL_OUTPUT>` 持有
`batch_queue` / `batch_thread` / `batch_entry` / `request_state` /
`batch_loop` / `process_batch` / `complete_batch_item`。对外：

- `start()` / `stop()`（`stop` 必须失败在队条目并 `join`，对应今天析构里先停
  batch 的注释）
- `submit(InferenceTask) -> shared_ptr<request_state>`（或等价：调用方构建
  state，收集器只收 entry）

取 worker 走注入的 `WorkerPool&`，**禁止**在收集器里再写一份 wait_dequeue +
卡死。`max_batch_size == 1` 时根本不构造 / 不 `start` 本对象，保持今天「默认
单请求路径碰不到 batch 队列」的语义。

**为什么拆**：
- 它是唯一自持线程的子组件；「请求超时后 late completion 落入弃置状态」的
  `shared_ptr` 所有权论证必须整段迁移，一个字都不要改写；
- 停机次序从析构里的隐式约定变成 `batch.stop()` 然后 `async.drain()` 然后
  `pool.drain()`。

### 3.6 `src/server/async_endpoints.h` —— 异步任务 HTTP 端点族

**内容**：`AsyncEndpoints<MODEL_OUTPUT>` 持有 `/jobs` 路由与
`handle_async_{submit,status,wait,result}`、waiter 注册表
（`register` / `unregister` / `wake`）、`parse_wait_timeout_ms`、
`reply_async_status`、`reply_async_error`，以及析构必须等待的
`async_inflight` / `wait_inflight`。对外：

- `handle(WFHttpTask*)`
- `drain()`：等到两个 inflight 计数为 0（今天析构里那两圈 poll）
- 提交成功后要跑模型时，通过注入的 `schedule_job(job_id)` 回调回到编排器
  （编排器里仍是 `create_go_task` + `async_run_job`）。**本类不 checkout worker。**

结果 JSON 通过注入的 `fill_response` 回调组装，与同步路径共用
`BaseAiServerImpl::fill_response_data`。

**为什么拆**：
- 端点族约 340 行（不含 `async_run_job`）是本文件最大的一块，且只被
  `serve_process` 的一个分支引用——整块搬移；
- waiter 注册表（命名计数器、R8）值得与 `async_job_table` 同级的单测，目前藏在
  类私有区不可达；
- inflight 计数属于这族 HTTP 任务的生命周期，不应继续作为编排器上的散装原子量。

### 3.7 `src/server/base_server_impl.h`（瘦身后）—— 唯一编排器

**保留**：

- `serve_process` 端点分发序（限流 / 鉴权 / `/healthz` `/ready` `/metrics`
  `/openapi.json` `/jobs` / 模型 URI）
- `do_work` / `run_items` / `aggregate_item_statuses` / `async_run_job`
- `go_task_functor` / `do_work_cb` / `request_meta`
  （**go task 生命周期族明确不拆**——所有权论证横跨这三个函数与析构排水，
  拆散会把一段完整的生命周期证明摊到多个文件，是负优化）
- `detail::make_model_input`（体积不够自成文件；跟 HTTP 无关，跟 run 有关）
- 析构排水：`batch.stop()` → `async.drain()` → `pool.drain()`
- `PrometheusMetrics`、`FixedWindowRateLimiter`、`AsyncJobTable` 的组合
- 对六个子模块的委托
- `parse_common_server_config` 保留为薄封装（内部调 `parse_server_runtime_config`
  再回填 `_m_*`），这样 `generic_cv_server.h` 与现有测试夹具的 `init()` 不用改形参

**顺手清理（同文件、行为不变）**：复制/赋值改为 `= delete`（类里已有
`std::mutex` / `std::thread` / `atomic`，今天的 `= default` 实际是已删除的
误导注释）。

**不保留为独立模块**：把 `do_work_cb` 与 `handle_async_result` 的信封组装再抽
一层。两处共享的是「item_status → UnifiedResponse」的十几行循环，不够一个文件；
做成编排器上的一个 private 函数即可。

**预期**：约 600–700 行、`_m_` 成员从 39 降到约 12–15（metrics、limiter、
table、pool、batch、async 端点、param_specs、model_name、uri、waiting/received
计数）、修改原因从 11 个收敛到 3 个（编排策略 / go-task 生命周期 / 子模块接线）。

## 4. 拆分后的文件组织形式

```
src/server/
├── abstract_server.h            # 不变：WFHttpServer 组装 + start/stop 门面
├── base_server_impl.h           # 瘦身：编排 + go task 生命周期 + 排水（<700 行）
├── http_wire.h                  # 新：HTTP 原语 + 统一信封应答（非模板）
├── server_runtime_config.h     # 新：TOML → ServerRuntimeConfig（调用已有 schema）
├── request_admission.h          # 新：正交准入纯函数（workflow-free）
├── worker_pool.h                # 新：WorkerPool<WORKER>（队列 + checkout + 卡死 + EWMA）
├── batch_collector.h             # 新：BatchCollector（自持线程，依赖 WorkerPool）
├── async_endpoints.h             # 新：/jobs HTTP + waiter 注册表 + inflight 排水
├── inference_task.h             # 已有：InferenceTask / InferenceResult
├── parsed_request.h             # 已有：信封 → 绑定
├── async_job_table.h            # 已有：异步账本
├── backpressure.h               # 已有：Retry-After 算术（admission 复用）
├── server_config_schema.h      # 已有：TOML schema（不被 runtime_config 吞并）
├── rate_limiter.h / prometheus_metrics.h / http_status.h / ...
└── generic_cv_server.h          # 不变：CvServerSpec → CvModelServer 装配
```

`src/server/CMakeLists.txt` 的 `SERVER_LIB_SRC` 追加上述新头文件（本目录库已是
header-only `LINKER_LANGUAGE CXX`，与现有 `async_job_table.h` 同类）。

依赖方向（只允许向下）：

```
generic_cv_server.h
   └─ base_server_impl.h（编排）
        ├─ batch_collector.h ── worker_pool.h ── inference_task.h
        ├─ async_endpoints.h ── async_job_table.h / inference_task.h / http_wire.h
        ├─ worker_pool.h
        ├─ request_admission.h ── backpressure.h
        ├─ server_runtime_config.h ── server_config_schema.h（toml）
        └─ http_wire.h（workflow + envelope）
纯逻辑层（request_admission / backpressure / WorkerPool 的卡死状态机 / 账本）
保持 workflow-free，tests-only CI 可直接单测。
WorkerPool 本身只依赖队列与时钟，不依赖 WFHttpTask。
```

## 5. 迁移计划（五步，每步独立可合入、行为零变化）

| 步骤 | 内容 | 安全网 |
|---|---|---|
| 1 | 抽 `http_wire.h` + `server_runtime_config.h`。解析不再 `start` 线程：配置落地后由编排器显式启动 batch（若 `max_batch_size > 1`）。`parse_common_server_config` 留作薄封装。 | `server_e2e_contract_test`、`worker_nums_unittest`、`config_schema_test` |
| 2 | 抽 `worker_pool.h`：`do_work` 与 `process_batch` 改为 `checkout`/`checkin`，消灭重复卡死判定。 | 新增 checkout / 卡死假时钟单测 + e2e；`do_work_lifetime_unittest` |
| 3 | 抽 `request_admission.h`：同步 POST 与 `handle_async_submit` 改调同一组纯函数；metrics 与信封回写仍在原处。 | 新增状态码边界单测 + 现有 `http_contract_test` |
| 4 | 抽 `batch_collector.h`（`request_state` / `batch_entry` 整体迁移；收集器只通过 `WorkerPool` 取 worker）。 | e2e 批处理用例 + `do_work_lifetime_unittest` |
| 5 | 抽 `async_endpoints.h`（waiter 注册表、`reply_async_error`、inflight 随迁；`async_run_job` 留在编排器，经 `schedule_job` 回调）。 | e2e async 全生命周期 + 新增 waiter 单测 |

顺序不能对调的原因：batch 必须先能复用 `WorkerPool`，否则收集器还会再写一份
checkout；async 端点最后搬，因为它与 Workflow 计数器、独立 series 的耦合最重。

每步之后主文件行数单调下降。公共 API（`BaseAiServer` 门面、`CvServerSpec` 装配、
HTTP 契约）保持行为不变。`generic_cv_server.h` / factory **零改动**是硬条件。
测试允许改 include（`parse_worker_nums` 的头文件路径）；不允许改断言含义。

## 6. 验收度量

- `base_server_impl.h` < 700 行，`_m_` 成员 ≤ 15；
- 取 worker 的卡死判定在仓库里只剩一份实现（`WorkerPool::checkout`），
  `do_work` 与 `process_batch` 不再各写一段 streak 算术；
- `parse_*_config` 不再创建线程；
- `request_admission.h` / `WorkerPool` 卡死路径进入 tests-only CI（workflow-free）；
- 全套 e2e / 契约测试零断言改动通过；TSAN / ASAN 门禁维持绿；
- `http_wire` 去模板化后 `full` preset 编译时间或符号数有可测改善（次要）；
- 六个月后回看：改准入只开 `request_admission.h`、改取 worker / 卡死只开
  `worker_pool.h`、调批处理只开 `batch_collector.h`、改 `/jobs` 只开
  `async_endpoints.h`——上帝类的最终判据是「改一处只开一个文件」。


---

# base_server_impl.h 拆分技术方案（dsv4-flash-high 独立评审版）

> **来源声明**：本修订意见来自 **dsv4-flash-high**（2026-09-09，基于对当前
> `src/server/base_server_impl.h` @1,939 行版本的逐行复核，并对照上文
> GLM-5.3-MAX 方案），作为独立评审意见追加于本文档末尾，供项目维护者
> 对照评估与执行。与上文方案重合的拆分 seam，本方案承认其为共识项并简述
> 理由；有分歧之处在对应小节以"与上文方案的差异"单独标注。

## 1. 现状诊断（复核要点）

复核结论与上文方案一致：**已是上帝类**。以下为本轮复核补充/修正的量化
证据（行号为 @1,939 行版本）：

| 维度 | 数值 |
|---|---|
| 文件体量 | 1,939 行 / 85 KB，22% 注释行 |
| 类声明 | 802 行（160–961）；另有 19 个类外模板成员定义（963–1935） |
| 数据成员 | **39 个 `_m_` 存储成员**；含 1 处 `public:` 裸数据区（252–257，5 个成员） |
| 嵌套类型/别名 | 11 个（`StuckWorkerAction`/`batch_entry`/`request_state`/`request_meta`/`go_task_functor`/`AsyncTable`…） |
| 成员函数 | 60+（含类内联实现），6 个访问区段 |
| #include | 49 个，横跨 arpa/inet、workflow、rapidjson、toml、glog |

复核时的四个关键观察（决定下文拆分取舍，前两点是相对上文方案的增量依据）：

- **重复的不只是"判定"，而是整段"worker 租约流程"**：`do_work`（≈1247–1278）
  与 `process_batch`（≈1476–1508）几乎逐行相同——带 deadline 的 checkout →
  失败时连续次数/时间跨度记账 → 成功时复位 + queue-wait 观测 → 运行后写
  指标再还 worker。这提示真正的 seam 不是"卡死检测器"，而是 **worker 租约
  (lease) 本身**。
- **派生类与基类的第二份接口是裸字段协议**：`generic_cv_server.h` 的
  `Impl::init`（122–195）直写 `_m_successfully_initialized` / `_m_param_specs` /
  `_m_model_name` / `_m_working_queue` / `_m_server_uri` / `_m_worker_nums`。
  配置对象化若不连这段握手一起收编，派生类仍是"翻字段"，模块化收益打折。
- **go task 生命周期族不能拆**（与上文方案共识）：`serve_process` 派发 →
  `go_task_functor::operator()` 发布 ctx 地址 → `do_work` 写 ctx → `do_work_cb`
  经 `task->user_data` 读 ctx，再连上析构排水——这是一段跨函数的单所有权证明，
  摊开即负优化。
- **metrics 计数纪律分散且靠注释维系**：30+ 处 `_m_metrics.*` 调用、多个
  "仅出口计一次、调用点勿重复计数"的约定——纯搬移时这些单点约束必须逐条保留。

## 2. 拆分总览：8 个职责单元（1 个瘦身编排器 + 7 个新模块）

先说明与上文方案（6 新模块 + 编排器）的差异，避免维护者误读为重复劳动：

| 议题 | GLM-5.3-MAX 方案 | 本方案（dsv4-flash-high） |
|---|---|---|
| 卡死检测 | 抽 `worker_health.h`，do_work/process_batch 各自保留 checkout 骨架 | **再抽 `execution_pool.h` worker 租约层**，整段 checkout 流程只留一个入口（见 3.5） |
| 派生类握手 | 只做配置值对象，派生类仍直写 protected 字段 | 配置模块内**收编 `ServerBootstrap` 握手**（见 3.2） |
| 准入边界 | 415/413/条数/背压 429 | 同，但明确 **422 归 parsed_request、401/限流不纯函数化**（见 3.3） |
| 路由 | 不抽，保留 serve_process 分发序 | 同，补充**不抽 Router 的理由**（见 3.8） |
| `reply_async_error` | 进 `http_wire.h` | 留 `async_endpoints.h`，http_wire 保持零 async 词汇 |

```
拆分前                                   拆分后（8 个职责单元）
─────────────────────                  ──────────────────────────────────────────────
base_server_impl.h (1,939 行)           http_wire.h             HTTP 原语/信封应答（自由函数）
  11 类职责交织 / 39 成员                server_runtime_config.h 配置结构体 + 解析 + ServerBootstrap
  两处重复 checkout 流程                 admission.h             准入决策纯函数
                                        worker_health.h         纯状态机：EWMA + StuckWorkerDetector
                                        execution_pool.h        模板：worker 租约（唯一 checkout 入口）
                                        batch_collector.h       模板：动态批收集器（自持线程）
                                        async_endpoints.h       模板：/jobs 端点族 + waiter 注册表
                                        base_server_impl.h      纯编排 + go task 生命周期 + 排水（<650 行）
```

既有的协作者文件（`parsed_request.h` / `async_job_table.h` / `backpressure.h` /
`prometheus_metrics.h` / `server_config_schema.h` / `inference_task.h` /
`response_serializers.h`）不在本次拆分范围，继续以 has-a / 自由函数形态被
上述单元引用。

## 3. 模块明细

### 3.1 `src/server/http_wire.h` — HTTP 线路原语与应答基础件（自由函数）

**内容**：`peer_ip_of` / `header_value_of` / `authorization_header_of` /
`is_json_content_type` / `is_raw_body_content_type` / `generate_req_id` /
`monotonic_ms`；信封应答基础件 `status_envelope` / `unified_rejection` /
`reply_unified_json(resp, unified)`（含 X-Request-ID / Cache-Control /
Content-Type 固定写法）。

**为什么拆**：
- 全部是**无状态的模板 static 成员**，每个 `(WORKER, MODEL_OUTPUT)` 实例化
  都复制一份机器码（generic_cv_server 的每个模型 + 三个测试夹具都在实例化）；
  改自由函数后链接器只保留一份，编译体积/时间双降——本方案唯一的性能红利；
- 依赖面（workflow + envelope）与业务零交叠，是最干净的 seam；
  `arpa/inet.h` 等 socket include 随之移出主文件。

**与上文方案的差异**：`reply_async_error` 与 `/jobs` 专用回复不放入本文件，
留在 async_endpoints（3.7），保证本文件不出现任何 async 词汇。

### 3.2 `src/server/server_runtime_config.h` — 配置结构体 + 解析 + 派生握手收编

**内容**：
- `ServerRuntimeConfig` 值对象（连接/线程/超时/限流/鉴权/批处理/async/stuck
  共约 20 个字段），编排器持有一个值对象，不再直接触碰 toml；
- `parse_server_runtime_config(toml) -> config` + `parse_worker_nums`，
  schema 校验接线复用 `server_config_schema.h`；
- **`ServerBootstrap` 握手收编**：把 `Impl::init` 中直写
  `_m_param_specs` / `_m_model_name` / `_m_server_uri` / `_m_worker_nums` /
  `_m_working_queue` 的字段协议，收敛为一个受保护方法族（如
  `bootstrap(model_name, param_specs, server_uri)` 与 `commit_workers(n)`，
  后者内含水位提交与 `_m_successfully_initialized` 置位）。

**为什么拆**：
- "从 TOML 读 20 个键 + 默认值 + 告警 + 校验"与执行编排没有共享状态，独立后
  可表驱动单测（键缺失/类型错/非法值各回什么），目前要起整个 server 才能覆盖；
- **裸字段本身就是派生类与基类的第二份"接口"**（上文方案未覆盖的盲区）：
  模块化若留下字段握手，改配置面会同时波及基类成员与所有派生 Impl。收编后
  `generic_cv_server.h` 的 `Impl::init` 与测试夹具的 init 只剩"造 worker +
  调 bootstrap"，保护面收敛为**方法**而非**数据**。

### 3.3 `src/server/admission.h` — 准入决策纯函数

**内容**：`AdmissionDecision {verdict, status, retry_after_s, violations}` 与
`admit_request(content_type, content_length, item_count, waiting, ewma, workers,
limits) -> AdmissionDecision`，覆盖 415 / 413 / 条数上限 / 背压 429
（Retry-After 复用 `backpressure.h`）。

**边界划定（与上文方案的差异）**：**422 不迁入**——它由 `parsed_request.h`
的信封校验产生，语义上属于解析而非准入；**401 与限流不纯函数化**——鉴权
比对与 `FixedWindowRateLimiter` 是有状态的，留在编排器前置门（3.8），
避免用"注入状态"伪装纯函数。

**为什么拆**：准入是产品契约（HTTP 状态码映射），却是当前最难单测的部分
（逻辑内嵌在 `serve_process` ≈230 行中部）；纯函数化后状态码边界可在毫秒级
单测里穷举，并为将来网关/边缘与服务端的准入规则对照提供共同的可读形态。

### 3.4 `src/server/worker_health.h` — 纯状态机（EWMA + 卡死检测）

**内容**：`RunTimeEwma`（CAS 更新）+ `StuckWorkerDetector`（"连续次数 ×
时间跨度"双条件状态机，含 LOG/EXIT 动作）。纯逻辑、workflow-free、
**可注入假时钟**。

**为什么拆**：卡死判定"次数与跨度必须同时达标（并发等待重叠使纯计数说谎）"
是本文件最精妙也最易被误改的逻辑，独立文件 + 专属单测（假时钟）锁定。
**注意边界**：本文件只放状态机，不放 checkout 流程——那属于 3.5 的租约层。

### 3.5 `src/server/execution_pool.h` — worker 租约层（与上文方案的关键差异）

**内容**：`ExecutionPool<WORKER>` 模板，持有 `BlockingConcurrentQueue` 与
worker 水位/排水相关计数器，对外只暴露**一个租约入口**：

```
WorkerLease checkout(deadline)      // 内部：超时→stuck 记账→失败返回；
                                    //       成功→复位 + queue-wait 观测
void return_lease(WorkerLease&&)    // 接口约定：归还前指标/EWMA 必须已写入
```

`do_work`（单请求路径）与 `process_batch`（批路径）各自只写
`auto lease = pool.checkout(deadline); …; pool.return_lease(...)`——两处
重复的"取 worker + 卡死记账 + 复位 + 观测"整段消失；EWMA / queue-wait /
运行时长观测的调用点收敛到租约生命周期内，析构排水依赖的"指标先于还
worker"不变量（当前靠注释维系、两处手写保持一致）变成租约内部保证。

**为什么拆**（与上文方案的差异）：
- 上文方案把卡死判定抽到 `worker_health.h` 后，do_work 与 process_batch 仍
  各自保留 checkout 骨架——只去重了"判定"，没去重"流程"；本方案把整段
  **租约事务**抽走，编排器里不再出现任何队列原语（连 `wait_dequeue_timed`
  的调用点都不留）；
- 批路径与单路径从此共享同一份租约语义，未来加"worker 亲和/优先级/优雅
  停机"等策略只动这一个文件。

### 3.6 `src/server/batch_collector.h` — 动态批收集器（模板）

**内容**：`BatchCollector<WORKER, MODEL_OUTPUT>` 自持批队列与收集线程，
`batch_entry` / `request_state` / `batch_loop` **整体迁移**（一行不改）；
`process_batch` 改为通过注入的 `ExecutionPool` 取租约执行 `run_batch` 后
分发结果；对外仅暴露 `submit(...)` / `start()` / `stop()` 与配置。

**为什么拆**：它是唯一自持线程的子组件，与 HTTP 编排无共享状态以外的耦合；
"请求超时后 late completion 落入弃置状态"的 shared_ptr 所有权论证整体迁移，
不动一个字；停机次序（先停 collector → 失败在队条目 → 再排 worker）在编排器
析构中变成一行显式调用，比现在散在 20 行注释里的隐式约定清晰。与上文方案
基本共识，差异仅是 process_batch 的 worker 获取改走 3.5 租约。

### 3.7 `src/server/async_endpoints.h` — 异步任务端点族（模板）

**内容**：`AsyncEndpoints<MODEL_OUTPUT>` 收纳 `/jobs` 六个 handler
（submit / status / wait / result + 路由与 404/405）+ waiter 注册表
（register / unregister / wake + `async_inflight` / `async_wait_inflight` /
`async_wait_seq`）+ `parse_wait_timeout_ms` + `reply_async_status` +
`reply_async_error`。`async_run_job` 的执行段改走 3.5 租约 + `run_items`
（`run_items` 留在编排器——它同时服务同步路径）。结果组装经注入的
`fill_response` 回调（即现有 `CvResponseFiller` 形态）。

**为什么拆**：async 相关 ≈370 行（527–919）是本文件最大的一块，且只被
`serve_process` 的一个分支引用——最典型的整块搬移；waiter 注册表（命名
计数器唤醒机制）值得与 `async_job_table_unittest` 同级的独立测试，目前
藏在类私有区不可达。

### 3.8 `src/server/base_server_impl.h`（瘦身后）— 唯一编排器

**保留（明确不拆）**：
- `serve_process` 的分发序与**前置门**：鉴权 / 限流 / 401 / 429（有状态，
  不抽象化）；每支仅 3–8 行的 admin 端点（healthz / ready / metrics /
  openapi.json）就地处理——**不抽 Router**。理由：分发阶梯虽长但每支极短，
  抽成路由表会引入间接层，并把"限流在鉴权前、健康端点免鉴权"这类前置次序
  语义藏进表结构，可读性净损失；
- **go task 生命周期族**：`go_task_functor` / `do_work` / `run_items` /
  `aggregate_item_statuses` / `do_work_cb` / `request_meta`（所有权论证横跨
  这几个函数与析构排水，摊开即负优化——与上文方案共识）；
- 析构排水：async inflight 等待 + worker 水位等待，连同现有注释原样保留；
- 对七个子模块的组合与委托；作为成员的只剩 worker 池实例、配置值对象、
  指标对象与派生装配句柄。

**预期**：约 550–650 行、`_m_` 成员从 39 降到 ≤16、修改原因从 11 个收敛到
3 个（编排策略 / go task 生命周期 / 子模块接线）。

## 4. 拆分后的文件组织形式

```
src/server/
├── abstract_server.h            # 不变：WFHttpServer 组装 + start/stop 门面
├── base_server_impl.h           # 瘦身：编排 + 前置门 + go task 生命周期 + 排水（<650 行）
├── http_wire.h                  # 新：HTTP 线路原语 + 信封应答（非模板自由函数）
├── server_runtime_config.h      # 新：ServerRuntimeConfig + 解析/校验 + ServerBootstrap 握手
├── admission.h                  # 新：准入决策纯函数（415/413/条数/背压 429 + Retry-After）
├── worker_health.h              # 新：RunTimeEwma + StuckWorkerDetector（纯状态机，可注入时钟）
├── execution_pool.h             # 新：ExecutionPool<WORKER> worker 租约（模板，唯一 checkout 入口）
├── batch_collector.h            # 新：BatchCollector<WORKER, MODEL_OUTPUT>（自持线程，模板）
├── async_endpoints.h            # 新：AsyncEndpoints<MODEL_OUTPUT> /jobs 端点族 + waiter（模板）
├── inference_task.h             # 已有：InferenceTask / InferenceResult
├── parsed_request.h             # 已有：信封 → ParsedRequest（422 判定在此，admission 不重复）
├── async_job_table.h            # 已有：异步账本（async_endpoints 的 has-a）
├── backpressure.h               # 已有：Retry-After 算术（admission.h 复用）
├── rate_limiter.h / prometheus_metrics.h / http_status.h / server_config_schema.h / …  # 不变
└── generic_cv_server.h          # 不变：CvServerSpec 装配不变；Impl::init 改用 ServerBootstrap
```

依赖方向（只允许向下；纯逻辑层 workflow-free，tests-only CI 可直接单测）：

```
generic_cv_server.h
   └─ base_server_impl.h（编排：路由 + 前置门 + go-task 生命周期 + 排水）
        ├─ async_endpoints.h ──┐
        ├─ batch_collector.h ──┤（经 execution_pool 租约执行；run_items 留在编排器）
        ├─ execution_pool.h ───┤ 依赖 worker_health.h（纯状态机）
        ├─ admission.h ────────┼─ backpressure.h
        ├─ server_runtime_config.h（toml；收编派生握手）
        └─ http_wire.h（workflow + envelope）
 纯逻辑层：admission / worker_health / backpressure / AsyncJobTable / ParsedRequest
 保持 workflow-free → 毫秒级单测；execution_pool / batch_collector /
 async_endpoints 为模板层，经编排器注入 fill_response 与租约，不反向依赖 HTTP。
```

## 5. 迁移计划（五步，每步独立可合入、行为零变化）

| 步骤 | 内容 | 安全网 |
|---|---|---|
| 1 | 抽 `http_wire.h`（纯搬移 static 成员为自由函数，机械改写调用点） | e2e/契约测试全绿、不改动 |
| 2 | 抽 `server_runtime_config.h` + `ServerBootstrap` 握手收编（Impl::init 同步机械改写） | e2e + 新增配置解析表驱动单测 |
| 3 | 抽 `worker_health.h` + `execution_pool.h`：租约化 do_work / process_batch 的 checkout | `do_work_lifetime_unittest` + e2e 批处理用例 + 卡死检测假时钟单测 |
| 4 | 抽 `batch_collector.h`（request_state / batch_entry 整体迁移，process_batch 改租约） | e2e 批处理用例 + `do_work_lifetime_unittest` |
| 5 | 抽 `async_endpoints.h`（waiter 注册表随迁，async_run_job 改租约） | e2e async 全生命周期用例 + 新增 waiter 单测 |

每步之后主文件行数单调下降；公共 API（`BaseAiServer` 门面、`CvServerSpec`
装配、HTTP 契约）保持字节级不变——**调用方（factory / generic_cv_server /
全部测试）零改动**是验收的硬条件。唯一的例外是第 2 步：派生类 init 从
"翻字段"改为调 `ServerBootstrap`，该改动局限于 `generic_cv_server.h` 与
测试夹具，且属于机械替换。

## 6. 验收度量

- `base_server_impl.h` < 650 行，`_m_` 成员 ≤ 16，文件内不再出现
  `wait_dequeue*` / `BlockingConcurrentQueue` / toml / rapidjson 等底层符号；
- 新模块单测进入 tests-only CI；`worker_health` / `admission` / 配置解析以
  假时钟/纯函数形式毫秒级覆盖边界；
- 全套 e2e/契约测试零改动通过；TSAN/ASAN 门禁维持绿；
- 编译产物符号数下降（http_wire 去模板化的直接收益），`full` preset
  编译时间有可测改善；
- 六个月后回看：改准入只动 `admission.h`、改 worker 策略只动
  `execution_pool.h`、改 async 只动 `async_endpoints.h`、改派生装配只动
  `server_runtime_config.h` 的 bootstrap——上帝类的最终判据是"改一处只开
  一个文件"，且没有哪一处改动需要同时理解另外两个模块的内部状态。
