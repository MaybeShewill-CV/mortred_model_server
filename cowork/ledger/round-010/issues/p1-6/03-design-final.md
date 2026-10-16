# 技术方案

- 问题: P1-6
- 版本: v1（编排方决策版，采纳技术负责人线）
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-6

## 边界

只处理 `/metrics` 的 `method` 标签基数。`status` 标签、其它 label、`/metrics` 路由与鉴权均不动。模型侧（`base_server_impl.h` 7 处）与 gateway 侧（`gateway_app.cpp` 11 处）共用同一个 collector，必须一并覆盖。

## 做法

### 推荐

收口在 collector 单点（`src/server/prometheus_metrics.h`），17 个调用点零改动：

- 新增 `normalize_http_method(std::string_view)`：trim + 转大写后 ∈ {GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS} → 原样返回；否则一律返回 `"other"`。含 `'|'`、控制字符、或长度 > 16 的输入一律 `"other"`。
- `inc_http_requests`（`:41-43`）与 `observe_http_duration_ms`（`:52-55`）在拼 key 前调用它——两者都以 `method + "|" + status` 为 key，只改 counter 会漏掉直方图。

为什么必须在 collector 内做：`method + "|" + status` 这个 key 在渲染时按 `'|'` 切分（`:159`、`:290`），所以含 `'|'` 的 method 不只是基数问题，还能伪造出任意 `method|status` 标签对。

### 否决

- 对未知 method 返回 405/400 且不记标签：会改状态码契约（`GET /nope` 现为 404、未知 method 访问 `/healthz` 现为 200），破坏既有 e2e 断言。
- 只在 `base_server_impl.h` 逐点归一化：漏掉 gateway——而 gateway 才是对外监听、攻击成本更低的那一层。
- 只做白名单、不过滤 `'|'`：仍可注入伪标签对。

## 改动面

- `src/server/prometheus_metrics.h`（新增归一化函数 + 两处入口调用）
- `test/prometheus_metrics_test.cc`（新增基数用例）

## 明确不改

- 既有状态码契约（401/404/405/413/415/429/200/202）
- `/metrics` 路由与鉴权、"模型侧不做限流"的 P0 决定
- `base_server_impl.h`、`gateway_app.cpp`、`async_endpoints.h` 的调用点
- `docs/openapi.json`（该标签不在其中）

## 契约

`/metrics` 的 `method` 取值集合是固定有限集（7 个已知方法 + `other`）；任何请求方法既不能新增 series，也不能注入 `'|'` 或控制字符。已知方法（尤其是 `POST`）的既有字面断言保持不变。

## 性能与资源

每次计数多一次栈上短串归一化，无分配；对推理热路径零影响。

## 验收标准

- A1 `./_bin/prometheus_metrics_test` 退出 0，且 `method="POST"` 既有字面断言（`:37`、`:123`）未改；新用例断言 N 个不同未知 method 只产生**一个** `method="other"` series。
- A2 归一化对直方图同样生效：未知 method 的 duration 直方图不新增 `method!="other"` 的 series。
- A3 含 `'|'` 的 method（如 `A|B`）不产生新的 `method=` 取值，且渲染出来的标签里不出现 `|`。
- A4 `./_bin/server_e2e_contract_test` 退出 0（401/404/405/413/415 契约未变）。
- A5 提交只含 `prometheus_metrics.h` 与其测试；不含 `cowork/ledger/`、构建产物、`src/common/mortred_version.h`。

## 测试义务

- 上述 5 条，给出真实命令与退出码。
- 白盒：确认 counter 与 histogram 走同一个归一化函数；确认 `status` 未被动过。
- 反例：大小写混写（`post`、`PoSt`）折叠为 `POST` 而非 `other`（否则会把正常流量打成 other）。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10b
cmake --build build --target prometheus_metrics_test server_e2e_contract_test -j8   # 期望 0
./_bin/prometheus_metrics_test ; echo $?          # 期望 0
./_bin/server_e2e_contract_test ; echo $?         # 期望 0
```

## RD 不得自行决定

- 白名单集合（不含 TRACE/CONNECT）；折叠标签名（固定 `other`）；是否顺带白名单化 `status`（不做）。
