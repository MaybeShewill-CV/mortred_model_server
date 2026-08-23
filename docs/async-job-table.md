# Async Job Table (Architecture & Concurrency Contract)

This document records the P0 correctness rework of the async-job subsystem: what
was broken, how [`src/server/async_job_table.h`](../src/server/async_job_table.h)
fixes it, the concurrency contract maintainers must preserve, and how to verify
everything locally and in CI.

## 1. Background: the defects that motivated the rework

Before the rework the async machinery lived inline in `BaseAiServerImpl`
(`src/server/base_server_impl.h`): the `AsyncJob` struct, its map/LRU/deque, the
queue-depth counter and all four HTTP handlers. Four defects shipped with it:

**D1 - Cross-mutex data race (UB).** `job->state`, `job->error` and
`job->result` were *written* while holding the global `_m_async_mu`, but *read*
without any lock in `handle_async_status` / `handle_async_wait` /
`handle_async_result`, and under a *different* mutex (`wait_mu`) inside the
condition-variable predicate. Under the C++ memory model this is undefined
behavior; in practice it manifests as torn/stale reads, spurious 409s, and a
guaranteed TSAN report.

**D2 - Unsynchronized queue-depth reads.** `_m_async_queue_depth` was a plain
`int`: written under the mutex, read without it in the submit path, the metric
updates and the 429 check.

**D3 - TOCTOU on admission.** The 429 check
(`if (_m_async_queue_depth >= _m_async_max_queue)`) and the increment
(`++_m_async_queue_depth`) were two separate steps. Two concurrent submissions
could both observe depth 15 against a limit of 16 and both pass, exceeding the
configured bound.

**D4 - Lost-wakeup window.** State was published under `_m_async_mu` but the
condition variable was notified under `wait_mu`. A waiter that evaluated the
predicate between the state store and the notify could block for the full
poll interval; the old code masked this with a 500 ms re-poll loop.

**Root cause.** `test/async_job_unittest.cc` tested a *re-implementation* of
the struct, not the production code - which is exactly how the defects above
survived. The rework therefore extracts the machinery into a component that
unit tests compile directly.

## 2. Component positioning

`AsyncJobTable` is a **subordinate component (has-a)** of `BaseAiServerImpl`,
not a peer and not a second server. The split is *execution orchestration vs.
state bookkeeping* - not sync vs. async:

```text
BaseAiServerImpl  (protocol + execution orchestration, for BOTH paths)
├─ sync path:   parse -> worker pool -> model -> serialize
└─ async path:  parse -> AsyncJobTable.submit()      (admission + record)
                       -> WFGoTask -> worker pool    (execution stays here)
                       -> model run
                       -> AsyncJobTable.finish()     (terminal bookkeeping)
                       -> serialize (data from snapshot/take_result)
```

The worker pool stays shared between sync and async requests on purpose:
that is the resource arbitration design. `AsyncJobTable` owns only the
ledger - identity, admission, the state machine, retention (TTL + LRU) and
wait/notify. It has **zero** dependencies on Workflow HTTP, the worker pool
or metrics; `task_request` and `go_result<MODEL_OUTPUT>` were hoisted to
namespace scope in its header so the server and the ledger share exactly one
definition of each.

## 3. Concurrency invariants (the contract)

Any future change to `async_job_table.h` must preserve all five:

1. **Job state is `std::atomic<AsyncJobState>`.** Terminal checks (eviction,
   wait predicates, cheap polls) read it without holding any lock. The
   ordering is deliberately `seq_cst`: this is a status-poll path, not a hot
   loop - auditability beats micro-optimization.
2. **One mutex per job guards `result` / `error` / `completed_at` - and the
   same mutex guards the condition variable.** Every transition writes the
   payload fields, publishes the state and notifies inside one critical
   section, so a lost wakeup is impossible by construction.
3. **`queue_depth` is `std::atomic<int>` and admission is a CAS loop**
   (`compare_exchange_weak`). The queue-full check and the increment are a
   single atomic step; D2 and D3 are both eliminated.
4. **The terminal transition is the only depth decrement, and it is
   exactly-once.** A second terminal call on the same job is a no-op that
   returns `false`; the state machine guarantees the decrement happens once.
5. **The table mutex protects only the id map and the LRU deque.** The lock
   order is always table -> job (in `evict_expired_locked`), never the
   reverse, so no deadlock is possible.

## 4. State machine

```text
            transition_running()          finish(id, result)
 PENDING ─────────────────────────► RUNNING ────────────────► DONE
    │                                   │
    │  (only PENDING may enter RUNNING; │ fail(id, error)
    │   any non-terminal may terminate) ├──────────► FAILED
    └───────────────────────────────────┤
                                        │ timeout(id, error)
                                        └──────────► TIMEOUT
```

- Terminal states (`DONE`, `FAILED`, `TIMEOUT`) are absorbing: every further
  transition returns `false`.
- Retention: terminal jobs are evicted lazily on the next `submit()` - first
  by TTL (`job_ttl_ms` after `completed_at`), then by LRU (beyond
  `max_completed`). Non-terminal jobs are never evicted.
- `take_request()` moves the payload out once (the large base64 image) but
  copies `task_id`, so the request-id echo of `/jobs/{id}/result` keeps
  working after the runner has consumed the payload.

## 5. HTTP endpoint to table API mapping

| HTTP endpoint | Table call(s) | Notes |
|---|---|---|
| `POST /jobs` | `submit(req)` | 202 with `job_id`; 429 when the CAS rejects |
| `GET /jobs/{id}` | `snapshot(id)` | 404 unknown id; cheap consistent view |
| `GET /jobs/{id}/wait?timeout=N` | `snapshot(id)` then `wait(id, initial, N)` | runs in a go task; falls back to the initial snapshot if the job was evicted mid-poll |
| `GET /jobs/{id}/result` | `take_result(id)` | 404 unknown / 409 not DONE / 200 with the standard envelope; repeatable until retention ends |

The server keeps metrics and worker acquisition on its side of the boundary;
the table is pure state.

Known Workflow-series semantics (unchanged, preserved for compatibility): the
202 reply of POST /jobs is flushed only after the job's go task completes,
because the runner is pushed into the HTTP task's series. This is a latency
wart of the previous implementation, deliberately not changed here; the
admission gate is therefore observable only via concurrent submissions (which
is exactly how the e2e 429 contract test drives it).

## 6. Verification

Local:

```bash
cmake --preset tests-only && cmake --build --preset tests-only && ctest --preset tests-only
cmake --preset tests-only-tsan && cmake --build --preset tests-only-tsan \
  && TSAN_OPTIONS="detect_deadlocks=0:report_mutex_bugs=0" \
     ctest --preset tests-only-tsan -L sanitizer        # TSAN, async tests
cmake --preset tests-only-asan && cmake --build --preset tests-only-asan \
  && ctest --preset tests-only-asan                     # ASan+UBSan, full suite
```

CI: the `sanitizers` job runs the TSAN gate on the `sanitizer`-labeled tests
(`async_job_unittest` + `async_job_stress_test`) and the ASan+UBSan gate on
the full tests-only suite. The TSAN run disables only detect_deadlocks and
report_mutex_bugs: this GCC runtime cannot model condition_variable::wait_for
and emits false double-lock / lock-order-inversion reports on legally
CV-guarded mutexes; data-race detection - the actual P0 gate - stays fully
enabled (verified with a deliberately racy control program). The stress test
drives 4 submitters, 2 runners,
3 pollers and 2 waiters against one table for ~3 s and asserts the
invariants at the end (depth returns to zero, every accepted job terminates
exactly once, ids stay unique).

## 7. Compatibility statement

The rework is a pure internal concurrency fix:

- No HTTP wire change: status codes (202/404/405/409/429), JSON bodies,
  headers (`Location`, `Content-Type`) and the error strings are unchanged.
- No configuration change: `async_enabled`, `async_timeout`,
  `async_max_queue`, `async_job_ttl`, `async_max_completed` keep their names,
  defaults and semantics (`async_max_queue` <= 0 still admits nothing).
- Job state remains in-memory and is lost on restart - unchanged behavior;
  persistence is future work and must not change the wire contract.
