# 技术方案

- 问题: P1-1
- 版本: v2
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md`（v3）。本版吸收 v1 RD 不同意条件：自测与 AC 一一对应；冻结条目里的「doctor --live」按「`mortredctl doctor` 在 supervisor 可达时的 live 检查」理解，验收命令一律写真实 CLI `mortredctl doctor`，不新增 `--live` 旗标。

## 边界

- 只收口：模型波 `start_server` 失败与 `models deadline` 超时目前几乎只 `fprintf(stderr)`，未写入与网关同级的结构化门闩。改完后人为触发模型波失败，一次 `mortredctl doctor`（supervisor 可达走 live 检查）或 `GET /api/v1/status` 即可看到明确终态与原因。
- 必须复用已有 `status.autostart`（`ProcessSupervisor::AutostartGate`）。禁止新增第二套管理面或新 HTTP 路径。
- 不放宽空 scrape token 的 fail-closed（已关闭第 1 轮 P0-2）。不做 P1-2、P1-3、P1-4。
- 一 catalog 一进程；gateway `:8080`、supervisor `:8787`。测试在本机 WSL。

## 做法

### 推荐

在现有 `AutostartGate` / `status.autostart` 上增加模型波终态布尔字段（推荐名 `models_failed`）。网关已放行后，若任意 `start_server` 失败，或 models deadline 时仍有已启动且未到 `running` 的 child，则该字段为真，并写入含失败 id 与 `deadline` 或 spawn 失败关键字的 `reason`。不得把这种情况写成 `models_skipped=true`。`mortredctl doctor` 在 supervisor 可达且该字段为真时，live 检查退出码非 0，并打印同一 `reason`。

### 否决

只扫 `servers[]` 而不写 `autostart`。新建顶层字段或新 API。

## 改动面

- `src/control/supervisor.h`、`src/control/supervisor.cpp`
- `src/control/supervisor/supervisor_app.cpp`
- `scripts/mortredctl_doctor.sh`
- `test/supervisor_unittest.cc`

## 明确不改

空 scrape fail-closed、无 uri readiness、prepare 入口、文档样例、推理契约、默认 deadline 数值。不新增 `mortredctl doctor --live` 旗标。

## 验收标准

1. **AC1** deadline 失败后 `GET /api/v1/status`：`autostart.models_failed` 为真，`reason` 含 `deadline` 与未就绪 id，`models_skipped` 为 false。
2. **AC2** spawn 失败后：`models_failed` 为真，`reason` 含该 id，`models_skipped` 不为 true。
3. **AC3** 空 scrape：`blocked` 与 `models_skipped` 为 true，`models_failed` 为假，零 model spawn。由 `supervisor_unittest` 覆盖。
4. **AC4** 健康路径：模型都在 deadline 内 `running` 时 `models_failed` 为假，且 `mortredctl doctor` 不因本条非 0。
5. **AC5** 失败路径：`mortredctl doctor` 退出码非 0，输出含与 status 同一 `reason`。

## 自测命令

在 WSL、已有 control 测试的 build 目录执行。合入前必须为期望退出码。

| AC | 命令 | 期望退出码 |
| --- | --- | --- |
| AC3 | `scripts/run_tests.sh BUILD -R 'supervisor_unittest'`，用例断言空 scrape 时 blocked、models_skipped 为 true，models_failed 为假，零 model spawn | 0 |
| AC1 | 触发 models deadline 后 `curl -fsS` `GET /api/v1/status`，断言 `models_failed==true`，`reason` 含 deadline 与未就绪 id，`models_skipped==false` | 0 |
| AC2 | 触发某 id spawn 失败后同一 curl，断言 `models_failed==true`，`reason` 含该 id，`models_skipped` 不为 true | 0 |
| AC4 | 健康波：curl 断言 `models_failed` 为假；`mortredctl doctor` 不因本条非 0 | curl 0；doctor 0 |
| AC5 | AC1 或 AC2 终态下 `mortredctl doctor`，输出含同一 reason | 非 0 |
| 合入 | `scripts/run_tests.sh BUILD -R 'supervisor_unittest'` | 0 |

## RD 不得自行决定

不得新建平行管理面。不得用 `models_skipped=true` 表示模型波失败。不得放宽空 scrape。不得做 P1-2/P1-3/P1-4。不得新增 `doctor --live`。字段名若不用 `models_failed`，实现说明必须写清等价名。
