# 技术方案

- 问题: P2-19
- 版本: v1
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md` 锁定队列 v3 的 P2-19

## 边界

本方案只修一件事：同一次成功推理累计超过 4096 次之后，在 gateway 五段 p50 之和仍大于模型 `server_time_ms` p50 的一半时，gateway `GET /metrics` 仍然写出 `mortred_gateway_stage_dominant`，并且五段 `mortred_gateway_stage_duration_ms` 直方图仍在。

锁定的可观察变化（队列 v3）：用会在响应里写很小 `server_time_ms` 的上游，连续 4097 次成功推理，再带 metrics token 读一次 gateway `/metrics`。正文仍有 `gw_in`、`route`、`auth`、`fwd`、`upstream` 各自的 `mortred_gateway_stage_duration_ms`，并且有一行 `mortred_gateway_stage_dominant{model="gateway",stage="..."} 1`。

出处与现状：`src/control/gateway/gateway_app.h` 的 `k_stage_sample_cap = 4096`；`gateway_app.cpp` 的 `record_stage_sample` 始终累加直方图并把样本留在最多 4096 条的 deque；`render_stage_metrics` 用 `stage_observed_ <= k_stage_sample_cap` 挡住主导段输出。第 4097 次成功之后直方图 `_count`/`_sum` 继续增加，主导段不再出现。头文件注释已写明「直方图累计；deque 只为中位数保留最近 4096 行」，与这道门闩矛盾。

P1-1 已合入时的契约写过「累计条数大于 4096 时不再输出主导段」。本条显式推翻那一句：主导段在超过 4096 之后仍按 deque 里最近至多 4096 条配对样本的中位数计算。不把 PF4 改成未满足。

不升为 P0/P1。不改模型进程、TensorRT、engine、worker 数、共享 context。不新建 worktree 以外的产品行为。

## 做法

### 推荐

做法一：只改 `render_stage_metrics` 里主导段的门闩，并同步监控指南里过时的「超过 4096 不再输出」句子。

1. 把 `if (stage_observed_ <= k_stage_sample_cap)` 改成 `if (!stage_samples_.empty())`。门闩内逻辑保持原样：从 deque 算五段与 `server_time_ms` 的中位数；gateway p50 为五段 p50 之和；严格大于 `0.5 * server_p50` 时写出 HELP、TYPE 与一行 `mortred_gateway_stage_dominant{model="gateway",stage="<winner>"} 1`；并列时按 `gw_in`、`route`、`auth`、`fwd`、`upstream` 取靠后名字。
2. `record_stage_sample` 保持原样：`k_stage_sample_cap` 仍限制 deque 长度；满时 `pop_front`；直方图 `stage_sum_` / `stage_bucket_` / `stage_observed_` 仍是进程启动以来的累计。不放大样本上限，不另开无界缓冲。
3. `docs/monitoring-guide.md` 与 `docs/monitoring-guide.zh-cn.md` 中「Past 4096 … dominant gauge is left out / 累计超过 4096 条配对之后不再输出这个主导段」改为：累计超过 4096 后，主导段 p50 仍按 deque 中最近至多 4096 条原始样本的中位数计算；直方图 `_count`/`_sum` 仍是累计值。

代价：主导段在长跑后反映的是最近 4096 次配对的中位数窗口，不再是「全历史且仅当 total≤4096」。内存上界不变。没有 engine、显存或共享 context 代价。

### 否决

做法二：把 `k_stage_sample_cap` 提高到很大或去掉 deque 上限，使 `stage_observed_ <= cap` 永远成立。主导段会回来，但内存随请求线性涨，违背现有「最多 4096 行」的刻意上界。否决。

做法三：主导段改从累计直方图桶近似 p50，完全不读 deque。桶是 `le` 累计结构，近似中位数与现有 `median_of` 精确中位数不一致，会改动已有 `StageMetricsDominantNamesTheLargestStage` 在 count=1 之外的语义，也超出「修错误门闩」的最小面。否决。

做法四：在 scrape 端或 `scripts/coverage_http_report.py` 用五段 `_sum`/`_count` 自己算 dominant，不改 gateway。锁定验收读的是 gateway `/metrics` 正文里的那一行 gauge。否决。

做法五：共享 TensorRT context、改 worker 池、改模型 `/metrics`、或把五段塞进 `mortred_http_request_duration_ms`。超出本条，且违反结构约束。否决。

## 改动面

- `src/control/gateway/gateway_app.cpp`：仅 `render_stage_metrics` 中主导段外层条件由 `stage_observed_ <= k_stage_sample_cap` 改为 `!stage_samples_.empty()`。直方图输出路径、锁、写入路径不改。
- `src/control/gateway/gateway_app.h`：常量与成员布局保持原样（`k_stage_sample_cap`、deque、`stage_observed_`、桶数组）。若头文件注释仍暗示「超过 cap 不算主导」，改成与推荐做法一致的一句；不改字段。
- `test/gateway_e2e_test.cc`：新增用例（建议名 `StageMetricsDominantSurvivesPastSampleCap`）：fake `--mode envelope --server-time-ms 0.125`；连续 4097 次成功 `POST /v1/models/FAKE/infer`；再 `GET /metrics`。断言五段 `mortred_gateway_stage_duration_ms_count` 均为 4097；正文含 `mortred_gateway_stage_dominant{model="gateway",stage="..."} 1`；仍含五段直方图系列。
- `docs/monitoring-guide.md`、`docs/monitoring-guide.zh-cn.md`：按推荐第 3 点改写超过 4096 后的主导段说明。

## 明确不改

- `record_stage_sample` 的 deque 淘汰与直方图累加；`k_stage_sample_cap` 的数值 4096。
- `median_of`、五段名字顺序、并列规则、严格大于比较式、gauge 值固定为 1。
- `stage_timing::enabled()`、`/tmp/mortred_stage_timing`、stderr `stage_trace`。
- 模型进程 `/metrics`、`mortred_http_request_duration_ms`、OpenAPI、产品/server toml、`worker_nums`、engine、static shape、batch=1。
- 每个 worker 各自持有 TensorRT execution context；不增加共享 context 或跨 worker context 池。
- gateway(:8080) 与 supervisor(:8787) 的进程关系；一个 catalog id 仍是一个 OS 进程。
- `StageMetricsVisibleWithoutFlag`、`StageMetricsDominantNamesTheLargestStage` 的既有预期（除本条新增用例外不改断言）。
- P2-20、P2-21、文档示例类开放项。

## 契约

`GET /metrics` 鉴权、Content-Type、五段直方图桶与标签不变。变化只在：当 `stage_samples_` 非空且五段 p50 之和严格大于 `server_time_ms` p50 的一半时，无论 `stage_observed_` 是否大于 4096，正文都追加主导段三行（HELP、TYPE、gauge）。`stage_observed_ > 4096` 时，用于主导段的 p50 来自 deque 中最近至多 4096 条配对；直方图 `_count`/`_sum` 仍反映全历史。HTTP 推理请求/响应 JSON 字段不变。

## 性能与资源

不生成 engine，不改 static shape、batch=1。不改 `worker_nums`。不在 gateway 初始化 CUDA。deque 仍最多 4096 条 `StageSample`；新增用例多打 4097 次本地 fake 推理，只作测试，不进产品常驻路径。每个 worker 仍各自持有自己的 TensorRT execution context。

## 验收标准

1. 用会在响应里写很小 `server_time_ms` 的上游（测试里 fake `--server-time-ms 0.125`），对同一 gateway 连续成功推理 4097 次。每一次 HTTP 200，响应顶层 `server_time_ms` 为该小数。
2. 第 4097 次成功之后，带 metrics token 读一次 gateway `GET /metrics`，HTTP 200。`gw_in`、`route`、`auth`、`fwd`、`upstream` 各有 `mortred_gateway_stage_duration_ms` 直方图（含 `_count` 为 4097）。
3. 同一次正文有一行 `mortred_gateway_stage_dominant{model="gateway",stage="STAGE"} 1`，`STAGE` 为五段中（按 deque 中位数）p50 最大的名字；并列取靠后。另有对应 HELP 与 TYPE。
4. `k_stage_sample_cap` 仍为 4096；`record_stage_sample` 在超过 4096 后仍 `pop_front`，不把 deque 扩成无界。
5. `docs/monitoring-guide.md` 与中文版不再写「超过 4096 不再输出主导段」，改为最近至多 4096 条窗口仍输出主导段；直方图累计语义不变。
6. 既有 `StageMetricsVisibleWithoutFlag`（大 `server_time_ms`、无 dominant）与 `StageMetricsDominantNamesTheLargestStage`（1 次成功、有 dominant）仍通过。

判定：以 `gateway_e2e_test` 新用例与上述两条旧用例为准。外部验收可用同一 fake 路径；不要求拉起真实 GPU 模型、不跑 calibrate、不跑官方 RPS。

## 测试义务

QA 在本机 WSL Ubuntu 跑：

- `StageMetricsVisibleWithoutFlag`
- `StageMetricsDominantNamesTheLargestStage`
- 本条新增的 `StageMetricsDominantSurvivesPastSampleCap`（或设计实现时的等价名）

期望三者均 pass。再用 `git diff` 确认本条改动只落在改动面所列文件。不把「未跑真实 yolov8s」写成失败。

## 自测命令

在本机 WSL Ubuntu、已编出 `gateway_e2e_test` 的构建树执行（实现时把二进制路径换成 worktree 构建产物）：

```bash
cd /mnt/g/Codex/mortred_model_server
# 构建产物路径以实现时 worktree 为准；过滤器覆盖本条与两条回归
./build/test/gateway_e2e_test \
  --gtest_filter='GatewayE2ETest.StageMetricsVisibleWithoutFlag:GatewayE2ETest.StageMetricsDominantNamesTheLargestStage:GatewayE2ETest.StageMetricsDominantSurvivesPastSampleCap'
```

期望退出码 0。另可用只读断言确认错误门闩已移除（实现后在 worktree 根执行，期望打印 ok）：

```bash
python3 - <<'PY'
from pathlib import Path
cpp = Path("src/control/gateway/gateway_app.cpp").read_text(encoding="utf-8")
assert "stage_observed_ <= k_stage_sample_cap" not in cpp
assert "!stage_samples_.empty()" in cpp or "stage_samples_.empty()" in cpp
print("p2-19-dominant-gate ok")
PY
```

## RD 不得自行决定

- 主导段门闩必须改为「deque 非空即可算」，不得继续用 `stage_observed_ <= k_stage_sample_cap`，也不得改成提高 cap 来绕过。
- 不得去掉或放大 `k_stage_sample_cap` 的 deque 内存上界。
- 不得改为直方图近似 p50 替代 `median_of`。
- 不得改五段名字、并列规则、严格大于比较、gauge 值 1。
- 不得共享 TensorRT execution context，不得改 worker 池、engine、产品 toml、`worker_nums`。
- 不得把验收改成只改文档或只改 scrape 脚本。
- 监控指南两句过时描述必须改掉，与代码一致。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | `cowork/ledger/round-002/issues/p2-19/03-discussion/v1-sign-pm.md` |
| RD | 同意 | 无 | v1 | `cowork/ledger/round-002/issues/p2-19/03-discussion/v1-sign-rd.md` |
| 技术负责人 | 同意 | 无 | v1 | `cowork/ledger/round-002/issues/p2-19/03-discussion/v1-sign-tech-lead.md` |
