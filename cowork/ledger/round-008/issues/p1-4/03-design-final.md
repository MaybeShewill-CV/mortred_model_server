# 技术方案

- 问题: P1-4 停服日志分不清 drain timeout 与成员开始销毁
- 版本: v1
- 对应列表: `cowork/ledger/round-008/02-p0-p1-final.md` P1-4

## 边界

不改进程模型，不取消外层 SIGKILL，不改 `kDrainTimeoutMs`。只在析构里 drain 返回之后打一条可区分的日志。失败 init 仍提前返回，不打这条日志。

## 做法

### 推荐

在 `BaseAiServerImpl` 析构中，`_m_async_eps.drain` 与 `_m_workers.drain` 都返回之后，打 `LOG(WARNING) << "server members destroying after drain"`。超时路径仍由既有 `drain timeout` 日志表达。两条文案不同。

在 `test/async_job_unittest.cc` 增加用例：`commit_workers(1)` 但不归还 worker，使 worker drain 超时。析构日志同时含 `drain timeout` 与 `server members destroying after drain`，且 destroying 出现在 timeout 之后。既有 `failed_init_destructor_skips_drain` 断言不含 destroying 文案。

### 否决

超时后跳过析构或取消 SIGKILL。

## 改动面

- `src/server/base_server_impl.h`
- `test/async_job_unittest.cc`

## 明确不改

进程模型、`kDrainTimeoutMs`、worker/async drain 的超时语义。

## 契约

超时日志仍含 `drain timeout`。成员开始销毁的日志是 `server members destroying after drain`。

## 性能与资源

多一条失败或正常停服日志。超时测试大约 3 秒。

## 验收标准

1. **AC1** 成功 init 且 worker 未归还时，析构日志含 `drain timeout` 和 `server members destroying after drain`。
2. **AC2** destroying 子串出现在 `drain timeout` 之后。
3. **AC3** 未成功 init 的析构日志不含 `server members destroying after drain`，也不含 `drain timeout`。
4. **AC4** `scripts/run_tests.sh build/tests-only -R 'async_job_unittest'` 退出码 0，且实际执行该测试。

## 测试义务

WSL。tests-only 即可。不要求 GPU。

## 自测命令

```bash
scripts/run_tests.sh build/tests-only -R 'async_job_unittest'
```

期望退出码 0，且不是 0 tests。

## RD 不得自行决定

不得取消 SIGKILL。不得改超时时长。不得在失败 init 路径打 destroying 日志。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | `cowork/ledger/round-008/issues/p1-4/03-discussion/v1-sign-pm.md` |
| cv-rd | 同意 | 无 | v1 | `cowork/ledger/round-008/issues/p1-4/03-discussion/v1-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v1 | `cowork/ledger/round-008/issues/p1-4/03-discussion/v1-sign-tech-lead.md` |
