# 技术方案

- 问题: P1-2
- 版本: v1（编排方决策版，采纳技术负责人线）
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-2

## 边界

只处理三个凭据环境变量向**模型子进程**的继承面：`src/control/supervisor/supervisor_app.cpp` 的全局 `setenv`，以及 `src/control/supervisor.cpp` 的 `environ` 拷贝与两条分支注入。不改 supervisor 自身鉴权、gateway 鉴权语义、argv/catalog 契约。

现状（已逐行核实）：
- `supervisor_app.cpp:970` 全局 `::setenv("MORTRED_API_TOKEN", auth_token_, 1)`，注释自认"The process environment is deliberately global: children inherit it"；
- `spawn_locked`（`supervisor.cpp:374-410`）把父进程整个 `environ` 拷进 `env_storage`，然后**只按分支注入、从不删除**：gateway 分支注入 `MORTRED_INTERNAL_TOKEN`/`MORTRED_METRICS_TOKEN`/`MORTRED_GATEWAY_AUTH_TOKEN`/`MORTRED_API_TOKEN`（`:384-395`），模型分支只注入 listen host / auth token / worker 数等（`:399-408`）；
- 结果是模型进程 `/proc/<pid>/environ` 里能读到管理 bearer（若操作者的 supervisor 环境里存在，或由 :970 注入）。

## 做法

### 推荐

1. **删掉全局 `setenv`**（`supervisor_app.cpp:969-971`）。已核实它在 gateway 路径上是冗余的：`set_gateway_trust(options.metrics_token, options.gateway_auth_token, auth_token_)`（`supervisor_app.cpp:854`）把同一个 `auth_token_` 交给 supervisor 的 `_gateway_admin_token`（`supervisor.cpp:170`），gateway 分支随后显式注入 `MORTRED_API_TOKEN`（`:393-395`）。删除后 gateway 行为不变，而 supervisor 自身 `/proc/self/environ` 也不再带管理 bearer。
2. **新增 `drop_env_entry(std::vector<std::string>&, const char*)`**（紧邻既有 `override_env_entry`，`supervisor.cpp:47-57`），在 `spawn_locked` 拷完 `environ` 后、**fork 之前**统一删除四个键：`MORTRED_API_TOKEN`、`MORTRED_METRICS_TOKEN`、`MORTRED_GATEWAY_AUTH_TOKEN`、`MORTRED_INTERNAL_TOKEN`；然后按分支注入——gateway 分支会重新注入它需要的三个（`:387-395`）与 internal token（`:384`），模型分支完全不注入这四个。
   - 删 `MORTRED_INTERNAL_TOKEN` 的理由：它是 gateway 与模型进程之间的内部令牌，模型侧用的是被显式注入的 `MORTRED_AUTH_TOKEN`（`:400`），模型进程不需要前者。
3. **测试**：`test/supervisor_unittest.cc` 新增用例——把三个（四个）凭据键放进 supervisor 的环境，拉起模型子进程与 gateway 子进程，读 `/proc/<pid>/environ` 断言：模型侧四键计数为 0；gateway 侧 `MORTRED_METRICS_TOKEN`/`MORTRED_GATEWAY_AUTH_TOKEN` 各 1。沿用既有 `fork/execv` 与 fake 二进制（`MORTRED_FAKE_BIN_DEFAULT`）测试手法，不改 `fake_model_server`。

### 否决

- **只删 `setenv`、不在 `spawn_locked` 里删键**：操作者 shell / systemd `EnvironmentFile` 里的三键仍会被整份拷进模型进程（`test/supervisor_malformed_e2e_test.cc:120-128` 正是这种注入姿势），缺陷只修了一半。
- **exec 前 `clearenv` + 最小环境重建**：会打断既有注入（`LD_LIBRARY_PATH`、`MORTRED_PROJECT_ROOT`、`MORTRED_MODEL`、`MORTRED_DECODE_AUTO` 等），风险远大于收益。
- 只给模型进程做 allowlist：与"最小改动 + 保留 gateway 注入"相比，把未来所有新键都变成需要显式登记，容易漏。

## 改动面

- `src/control/supervisor/supervisor_app.cpp`（删全局 setenv）
- `src/control/supervisor.cpp`（新增 `drop_env_entry`；`spawn_locked` 在分支注入前删四键）
- `test/supervisor_unittest.cc`（新增 `/proc/<pid>/environ` 用例）

## 明确不改

- supervisor / gateway 的 token 校验逻辑与 `trust_tokens` 文案；`set_gateway_trust` 的调用契约
- 模型进程仍拿 `MORTRED_AUTH_TOKEN`（internal token）与 `MORTRED_LISTEN_HOST=127.0.0.1`
- `MORTRED_GATEWAY_CORS_ORIGINS`（只给 gateway 用，模型侧拿到也无害——本轮不收窄）
- gateway 在环回上允许"零 token"的既有行为

## 契约

- 模型子进程 environ：`MORTRED_API_TOKEN` / `MORTRED_METRICS_TOKEN` / `MORTRED_GATEWAY_AUTH_TOKEN` / `MORTRED_INTERNAL_TOKEN` 计数均为 0。
- gateway 子进程：上述键按配置非空时各存在一次，`GET :8080/healthz` 200、带 scrape token 的 `GET :8080/metrics` 200。
- supervisor 的 `:8787` 管理接口与 `mortredctl` 继续用操作者提供的 `MORTRED_API_TOKEN`（现在只通过 supervisor 自身进程持有，不再外泄给子进程）。

## 性能与资源

spawn 路径多四次线性 erase；无额外进程/fd/线程，不触碰推理与 TensorRT 路径。

## 验收标准

- A1 `./_bin/supervisor_unittest` 退出 0，含新用例：模型子进程四键计数 0、gateway 子进程计量键各 1（在同一次运行内完成，不依赖外部起服）。
- A2 起服后外部验收：`M=$(pgrep -f 'mortred-model-server.out --model' | head -1); tr '\0' '\n' </proc/$M/environ | grep -cE '^(MORTRED_API_TOKEN|MORTRED_METRICS_TOKEN|MORTRED_GATEWAY_AUTH_TOKEN|MORTRED_INTERNAL_TOKEN)='` 输出 0。
- A3 同一时刻 gateway 进程 `tr '\0' '\n' </proc/$G/environ | grep -c '^MORTRED_METRICS_TOKEN='` 输出 1，且 `curl -s -o /dev/null -w '%{http_code}' :8080/healthz` 为 200。
- A4 `./_bin/mortredctl.out --addr http://127.0.0.1:8787 --token "$MORTRED_API_TOKEN" status` 退出 0 且列出 id。
- A5 提交只含上述 3 个文件；不含 `cowork/ledger/`、构建产物、`src/common/mortred_version.h`。

## 测试义务

- 上述 5 条，给出真实命令与退出码。
- 白盒：确认删除发生在 `fork` 之前（否则子进程已拷贝）；确认 gateway 分支仍在删键之后注入（顺序：拷 environ → drop → 分支 inject）。
- 回归：`supervisor_unittest` 既有用例（重启、fd 不泄漏、有序停机）全绿；`supervisor_malformed_e2e_test` 全绿。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10e
cmake --build build-full --target supervisor_unittest fake_model_server mortred-gateway.out mortredctl.out -j8   # 期望 0
./_bin/supervisor_unittest ; echo $?    # 期望 0
# 起服后的外部验收见 A2–A4
```

## RD 不得自行决定

- 是否把 `MORTRED_GATEWAY_CORS_ORIGINS` 一并收窄（不做）；采用 drop-list 还是模型侧 allowlist（drop-list）；是否删除 `MORTRED_INTERNAL_TOKEN`（删，理由见推荐第 2 条）；gateway "零 token 环回启动"是否改变（不改）。
