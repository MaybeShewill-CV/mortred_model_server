# 技术方案

- 问题: P1-2
- 版本: v1
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md` P1-2

## 边界

- 只解决：非 gateway child 在 `uri` 为空时不再长期停在 `starting`。一次读 `GET /api/v1/status` 或 supervisor 可达时的 `mortredctl doctor` 即可看到该 child 的明确终态与原因。
- `catalog.cpp` 加载时仍拒绝空或非 `/` 开头的 `server_uri`。本条不得放宽该必填。空 uri 的可观测路径以 unittest 构造为主。
- 不得只靠 models deadline。P1-1 的 `models_failed` 可复用，但 child 自己必须有终态和原因。
- 不改一 catalog 一进程、gateway `:8080`、supervisor `:8787`。不做 P1-3、P1-4。测试在 WSL。
- `mortredctl doctor` 没有 `--live`。用法仍是 `mortredctl doctor [--strict]`。

## 做法

### 推荐

非 gateway child 若 `entry.uri` 为空：在现有 readiness 探测路径上立刻记失败终态（`state=failed`），写入已有 child `error`。文案须同时点明缺 `server_uri` 和未做 HTTP readiness。gateway 仍走 `/healthz`。

若该 child 属于本次 autostart 模型波：同步置已有 `autostart.models_failed`，`reason` 能对上同一失败。不新增 HTTP 路径或 status 字段名。

### 否决

只靠 deadline。放宽 catalog 允许空 `server_uri`。新开管理 HTTP。

## 改动面

- `src/control/supervisor.cpp`（及必要时头文件）：空 uri 的非 gateway child 记永久失败并写 `error`；autostart 波内复用 `mark_models_failed_locked`。
- `test/supervisor_unittest.cc`：构造无 uri child，断言终态与原因。

## 明确不改

- `catalog.cpp` 对 `server_uri` 的加载拒绝。
- P1-1 的 deadline 语义、P1-3、P1-4、推理契约、doctor 旗标。

## 验收标准

1. **AC1** unittest 构造非 gateway、uri 为空的 child 进入探测路径后，在远短于默认 models deadline 的时间内 `status(id).state == "failed"`，不得仍为 `starting`。
2. **AC2** 同一 child 的 `error` 同时体现缺 `server_uri` 与未做 HTTP readiness。
3. **AC3** 该 child 若属于本次 `autostart_all` 模型波，则 `models_failed == true`，且 `reason` 能关联到该失败。不得只表现为 deadline 文案而 child 仍长期 `starting`。
4. **AC4** 在 AC3 成立且 supervisor 可达时，`mortredctl doctor` 退出码非 0。不得要求 `--live`。若环境抬不起可 curl 的 supervisor，AC4 可用 doctor 脚本里同一段 status 判定、输入为 AC3 的 JSON 形状，并在报告里写明。
5. **AC5** 空或非法 `server_uri` 的产品配置，`Catalog::init` 仍失败。
6. **AC6** gateway 仍按 `/healthz` 就绪；合法非空 `server_uri` 的 model 仍按 `/ready` 探测，不得被本条改成误 `failed`。

## 自测命令

在 WSL、已有 control 测试的 build 目录执行。合入前必须为期望退出码。`BUILD` 换成本机目录（上一题用的是 `build/tests-only`）。

| AC | 命令 | 期望退出码 |
| --- | --- | --- |
| AC1–AC3、AC6 | `scripts/run_tests.sh BUILD -R 'supervisor_unittest'` | 0 |
| AC5 | `scripts/run_tests.sh BUILD -R 'catalog_unittest'` | 0 |
| AC4 | doctor 门闩对 `models_failed=true` 且 reason 含 `server_uri` 的 status JSON | 非 0 |

## RD 不得自行决定

不得让无 uri 的非 gateway child 长期停在 starting。不得只靠 deadline。不得放宽 catalog。不得新增 HTTP 或新 status 字段名。不得做 P1-3/P1-4。`error` 不得省略「缺 server_uri」或「未做 HTTP readiness」任一信息。
