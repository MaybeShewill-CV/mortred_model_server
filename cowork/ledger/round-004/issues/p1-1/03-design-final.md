# 技术方案

- 问题: P1-1 成功 init 后停服对 worker/async drain 无界等待
- 版本: v2
- 对应列表: `cowork/ledger/round-004/02-p0-p1-final.md` P1-1；本版吸收 v1 PM 条件：验收只留单测可判定条目；原 AC4/AC5 移出验收；AC3 拆成共用截止时刻与未成功 init；自测写死已有的 `-R` 名

## 边界

只处理成功 init 之后析构里对 async drain 与 worker drain 的无界忙等。不改一 catalog 一进程、gateway `:8080`、推理 JSON、`kStopGraceMs`。不做 P1-2、P1-3。测试在 WSL。

## 做法

### 推荐

`AsyncEndpoints::drain` 与 `WorkerPool::drain` 增加可注入时限。生产常量 **`kDrainTimeoutMs = 3000`**。超时后必须返回，并打一条含子串 `drain timeout` 的日志，带上仍未完成的数量。

析构里两处 drain **共用一个截止时刻**，总预算 3000ms。未成功 init 仍不 drain。

单测注入 **200ms**。async 用例放进已有 target **`async_job_unittest`**。不要新建 `async_endpoints` 这个 ctest 名。

### 否决

只靠外层 SIGKILL。超时直接 abort。新建 `async_endpoints` target。

## 改动面

- `src/server/async_endpoints.h`
- `src/server/worker_pool.h`
- `src/server/base_server_impl.h`
- `test/worker_pool_unittest.cc`
- `test/async_job_unittest.cc`

## 明确不改

推理契约、端口、一 catalog 一进程、`kStopGraceMs`、P1-2、P1-3。

## 验收标准

1. **AC1** 卡住 worker 或 async inflight 后，注入 200ms 的 `drain` 在 2 秒内返回。
2. **AC2** 同一场景日志含 `drain timeout` 和未完成数量。
3. **AC3** 注入总预算 200ms，依次调用 async drain 与 worker drain（或析构等价路径）。墙钟小于 2 秒，且总耗时 < 350ms。
4. **AC4** 构造未成功 init 的对象并析构：没有 `drain timeout` 日志，且该调用在 2 秒内返回。

## 自测命令

在 WSL 执行。期望退出码都是 0。

```text
scripts/run_tests.sh build/tests-only -R 'worker_pool'
scripts/run_tests.sh build/tests-only -R 'async_job_unittest'
```

## RD 不得自行决定

生产时限必须是 3000ms。超时后不得再无限等。日志不得去掉 `drain timeout` 或剩余数量。两处 drain 不得各自满额 3000ms。不得新建 `async_endpoints` ctest 名。不得把 P1-2、P1-3 并进同一提交。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v2 | `cowork/ledger/round-004/issues/p1-1/03-discussion/v2-sign-pm.md` |
| cv-rd | 同意 | 无 | v2 | `cowork/ledger/round-004/issues/p1-1/03-discussion/v2-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v2 | `cowork/ledger/round-004/issues/p1-1/03-discussion/v2-sign-tech-lead.md` |
