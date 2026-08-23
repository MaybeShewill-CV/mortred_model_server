# 异步任务账本（架构与并发契约）

本文记录异步任务子系统的 P0 正确性重构：缺陷是什么、
[`src/server/async_job_table.h`](../src/server/async_job_table.h) 如何修复、
维护者必须保持的并发契约，以及本地与 CI 的完整验证方法。

## 1. 背景：触发重构的缺陷

重构前，异步机制内联在 `BaseAiServerImpl`（`src/server/base_server_impl.h`）
中：`AsyncJob` 结构体、map/LRU/deque、队列深度计数器和全部四个 HTTP handler
都挤在服务器模板类里。随之而来四个缺陷：

**D1 - 跨锁数据竞争（UB）。** `job->state`、`job->error`、`job->result` 的
**写入**持有全局 `_m_async_mu`，但**读取**在
`handle_async_status` / `handle_async_wait` / `handle_async_result` 中无锁，
且在条件变量谓词里处于**另一把**锁（`wait_mu`）之下。按 C++ 内存模型这是
未定义行为；实际表现为撕裂读/过期读、随机 409，TSAN 必报。

**D2 - 队列深度无锁读。** `_m_async_queue_depth` 是普通 `int`：写路径持锁，
读路径（提交、指标更新、429 检查）全部无锁。

**D3 - 准入 TOCTOU。** 429 检查（`if (_m_async_queue_depth >= _m_async_max_queue)`）
与自增（`++_m_async_queue_depth`）是两个独立步骤。两个并发提交可以同时
观察到深度 15（上限 16）并同时通过，突破配置上限。

**D4 - 丢失唤醒窗口。** 状态在 `_m_async_mu` 下发布，条件变量却在 `wait_mu`
下 notify。在状态写入与 notify 之间完成谓词判断的 waiter 可能阻塞整个轮询
周期；旧代码用 500ms 重轮询掩盖了这一点。

**根因。** `test/async_job_unittest.cc` 测试的是结构体的*重新实现*而非生产
代码——这正是上述缺陷得以存活的原因。因此重构将机制抽取为单测可直接编译的
组件。

## 2. 组件定位

`AsyncJobTable` 是 `BaseAiServerImpl` 的**下属组件（has-a）**，不是平级概念，
也不是第二台服务器。切分维度是*执行编排 vs 状态记账*，而非同步 vs 异步：

```text
BaseAiServerImpl（协议 + 执行编排，两条路径共用）
├─ 同步路径:  parse -> worker 池 -> 模型 -> 序列化
└─ 异步路径:  parse -> AsyncJobTable.submit()      （准入 + 建档）
                      -> WFGoTask -> worker 池      （执行仍在这里）
                      -> 模型运行
                      -> AsyncJobTable.finish()     （终态记账）
                      -> 序列化（数据来自 snapshot/take_result）
```

worker 池在同步/异步请求间刻意共享：这就是资源仲裁设计。`AsyncJobTable`
只管账本——身份、准入、状态机、保留策略（TTL + LRU）与 wait/notify。它对
Workflow HTTP、worker 池、metrics **零依赖**；`task_request` 与
`go_result<MODEL_OUTPUT>` 被上提到该头文件的命名空间作用域，服务器与账本
共享唯一定义。

## 3. 并发不变式（契约）

今后对 `async_job_table.h` 的任何修改都必须保持以下五条：

1. **Job 状态是 `std::atomic<AsyncJobState>`。** 终态判断（淘汰、wait 谓词、
   低成本轮询）无锁读取。内存序刻意用 `seq_cst`：这是状态轮询路径而非热
   循环——可审计性优先于微优化。
2. **每 job 一把互斥量保护 `result` / `error` / `completed_at`——同一把锁
   同时保护条件变量。** 每次迁移在同一个临界区内写载荷字段、发布状态并
   notify，丢失唤醒在构造上不可能。
3. **`queue_depth` 是 `std::atomic<int>`，准入是 CAS 循环**
   （`compare_exchange_weak`）。队列满检查与自增是单一原子步骤，D2 与 D3
   同时消灭。
4. **终态迁移是队列深度唯一递减点，且恰好发生一次。** 对同一 job 的第二次
   终态调用是返回 `false` 的 no-op；状态机保证递减只发生一次。
5. **表级互斥量只保护 id map 与 LRU deque。** 加锁顺序永远是表 -> job
   （在 `evict_expired_locked` 中），永不反向，因此不可能死锁。

## 4. 状态机

```text
            transition_running()          finish(id, result)
 PENDING ─────────────────────────► RUNNING ────────────────► DONE
    │                                   │
    │  （仅 PENDING 可进入 RUNNING；    │ fail(id, error)
    │   任意非终态可终止）              ├──────────► FAILED
    └───────────────────────────────────┤
                                        │ timeout(id, error)
                                        └──────────► TIMEOUT
```

- 终态（`DONE`、`FAILED`、`TIMEOUT`）是吸收态：后续一切迁移返回 `false`。
- 保留策略：终态 job 在下一次 `submit()` 时惰性淘汰——先按 TTL
  （`completed_at` 之后 `job_ttl_ms`），再按 LRU（超出 `max_completed`）。
  非终态 job 永不淘汰。
- `take_request()` 一次性移出 payload（大的 base64 图像）但复制
  `task_id`，因此 `/jobs/{id}/result` 的请求 id 回显在 runner 消费
  payload 之后仍然有效。

## 5. HTTP 端点与 Table API 映射

| HTTP 端点 | Table 调用 | 说明 |
|---|---|---|
| `POST /jobs` | `submit(req)` | 202 返回 `job_id`；CAS 拒绝时 429 |
| `GET /jobs/{id}` | `snapshot(id)` | 未知 id 404；低成本一致视图 |
| `GET /jobs/{id}/wait?timeout=N` | `snapshot(id)` 后 `wait(id, initial, N)` | 在 go task 中执行；轮询期间被淘汰则回退初始快照 |
| `GET /jobs/{id}/result` | `take_result(id)` | 未知 404 / 非 DONE 409 / 200 标准封装；保留期内可重复读取 |

metrics 与 worker 获取留在服务器一侧；账本是纯状态。

已知 Workflow series 语义（保持不变，兼容性优先）：POST /jobs 的 202
响应要等该 job 的 go task 完成后才刷出，因为 runner 被推入了 HTTP task 的
series。这是旧实现遗留的时延瑕疵，本次刻意不改；因此准入 429 只能通过并发
提交观察到（e2e 429 契约测试正是这样驱动的）。

## 6. 验证方法

本地：

```bash
cmake --preset tests-only && cmake --build --preset tests-only && ctest --preset tests-only
cmake --preset tests-only-tsan && cmake --build --preset tests-only-tsan \
  && TSAN_OPTIONS="detect_deadlocks=0:report_mutex_bugs=0" \
     ctest --preset tests-only-tsan -L sanitizer        # TSAN，async 测试
cmake --preset tests-only-asan && cmake --build --preset tests-only-asan \
  && ctest --preset tests-only-asan                     # ASan+UBSan，全套件
```

CI：`sanitizers` job 对带 `sanitizer` label 的测试（`async_job_unittest` +
`async_job_stress_test`）运行 TSAN 门禁，对完整 tests-only 套件运行
ASan+UBSan 门禁。TSAN 运行只关闭 detect_deadlocks 与 report_mutex_bugs：
当前 GCC 运行时无法正确建模 condition_variable::wait_for，会对合法 CV 保护的
互斥量误报 double-lock / lock-order-inversion；数据竞争检测——P0 的真正门禁
——保持全开（已用故意写竞态的对照程序验证仍然报警）。压力测试以 4 个提交线程、2 个 runner、3 个轮询线程、
2 个 waiter 对同一张表持续压测约 3 秒，结束时断言全部不变式（深度归零、
每个接受的 job 恰好终止一次、id 唯一）。

## 7. 兼容性声明

本次重构是纯内部并发修复：

- HTTP wire 零变更：状态码（202/404/405/409/429）、JSON 结构、响应头
  （`Location`、`Content-Type`）与错误文案全部保持不变。
- 配置零变更：`async_enabled`、`async_timeout`、`async_max_queue`、
  `async_job_ttl`、`async_max_completed` 名称、默认值、语义不变
  （`async_max_queue` <= 0 仍然拒绝一切提交）。
- job 状态仍在内存中、重启即失——行为不变；持久化属于未来工作，且不得
  改变 wire 契约。
