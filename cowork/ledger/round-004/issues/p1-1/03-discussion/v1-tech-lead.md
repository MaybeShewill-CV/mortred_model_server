# 技术方案

- 问题: P1-1 成功 init 后停服对 worker/async drain 无界等待
- 版本: v1
- 对应列表: `cowork/ledger/round-004/02-p0-p1-final.md` P1-1

## 边界

只处理成功 init 之后析构里对 async drain 与 worker drain 的无界忙等。不改一 catalog 一进程、gateway `:8080`、推理 JSON、`kStopGraceMs`。不做 P1-2、P1-3。测试在 WSL。

## 做法

### 推荐

`AsyncEndpoints::drain` 与 `WorkerPool::drain` 增加可注入时限。生产常量 **`kDrainTimeoutMs = 3000`**。轮询仍可 5ms。超时后必须返回，并打一条含子串 `drain timeout` 的日志，带上仍未完成的数量（async 的 inflight/wait，或 worker 未归还数）。

析构里两处 drain **共用一个截止时刻**，总预算 3000ms，第二段只用剩余时间。未成功 init 仍不 drain。

单测注入 **200ms**，断言 `drain` 在 **2 秒**内返回，并看到 `drain timeout` 日志。不得用生产 3000ms 把单测挂死。

### 否决

只靠外层 SIGKILL。超时直接 abort。后台线程继续等且不记超时。

## 改动面

- `src/server/async_endpoints.h`
- `src/server/worker_pool.h`
- `src/server/base_server_impl.h`
- `test/worker_pool_unittest.cc`，以及 async 侧同等短时限用例

## 明确不改

推理契约、端口、P1-2、P1-3、`kStopGraceMs`。

## 验收标准

1. **AC1** 卡住 worker 或 async inflight 后，注入 200ms 的 `drain` 在 2 秒内返回。
2. **AC2** 同一场景日志含 `drain timeout` 和未完成数量。
3. **AC3** 成功 init 的析构对两处 drain 共用 3000ms 总预算；未成功 init 不进入 drain。
4. **AC4** 不改 gateway 端口、一 catalog 一进程、推理 JSON。
5. **AC5** diff 不含 P1-2、P1-3 的功能改动。

## 自测命令

在 WSL 执行。期望退出码 0。

`scripts/run_tests.sh build/tests-only -R 'worker_pool'`

若 async 用例在别的 target，实现说明写明 `-R` 名，QA 再跑一条，期望退出码 0。

## RD 不得自行决定

生产时限必须是 3000ms。超时后不得再无限等。日志不得去掉 `drain timeout` 或剩余数量。两处 drain 不得各自满额 3000ms。不得用新 toml 或环境变量做生产开关。不得把 P1-2、P1-3 并进同一提交。
