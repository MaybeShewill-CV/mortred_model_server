# 技术方案

- 问题: P1-7
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-7
- 作者: 编排方（单一执行方模式；本轮通用子代理按角色卡执行，本份由编排方自撰并在 03 确认节标注）

## 边界

只修「测试与产品行为不一致」这一件事，外加把 `rate_limit_qps` 的真实语义写进配置 schema 的说明。不改任何产品限流行为，不改 fail-closed 启动守卫。

开工前基线（在 `/mnt/g/Codex/mortred_wt_r10` 实测，构建 `build`(tests-only) 与 `build-full`(BUILD_FULL=ON)）：

| 二进制 | 结果 | 失败用例 |
| --- | --- | --- |
| `request_envelope_unittest` | 3 failed / 17，exit 1 | `valid_single_image_with_defaults`、`valid_multi_image_with_req_id_and_options`、`images_shape_errors` |
| `server_e2e_contract_test` | 2 failed / 45，exit 1 | `rate_limited_returns_429`、`healthz_exempt_from_ip_rate_limit` |
| `gateway_e2e_test` | 3 failed / 49，exit 1 | `NonLoopbackKeysOnlyStartsAndEnforcesKeys`、`NonLoopbackWithOnlyApiTokenStarts`、`NonLoopbackWithDistinctMetricsTokenStarts` |

三处失效的根因都是**产品行为是有意的、测试没跟上**：
1. base64 在 item 生成处解码（`src/server/parsed_request.h:126-140`，含解释性注释；2026-09-21 `d9a49675` 引入），同文件 `:235` 的 raw-body 用例期望的正是解码后字节；
2. 模型侧限流已移除，gateway 是唯一限流权威（`src/server/base_server_impl.h` 只保留 `(void)_m_rate_limit_qps`）；
3. 非 loopback 且关闭限流时 gateway 拒绝启动（fail-closed），而 `GatewayAuthTest` 的三个非 loopback 用例都没开限流。

## 做法

### 推荐

1. `test/request_envelope_unittest.cc`：把三处断言改成与解码后契约一致——`items[0].data == "hello"`、`items[1].data == std::string("\x89\xE5")`（与 `:235` 的 raw-body 用例同构）；`images_shape_errors` 的 `{"images":["ok",5]}` 期望 **2** 条 violation，并断言两条的 JSON pointer 分别是 `/images/0`（非合法 base64，长度 2）与 `/images/1`（不是字符串）。指针断言必须显式写出，不允许只断数量。
2. `test/server_e2e_contract_test.cc`：`rate_limited_returns_429` 改写为「`rate_limit_qps=1` 在模型侧不再生效」的回归守卫——连续两次请求都必须是 200；`healthz_exempt_from_ip_rate_limit` 改写为「`/healthz` 与 `/ready` 在两种配置下都免鉴权且 200」。用例名同步改名以反映契约，避免名字继续骗人。模型侧真正的卸载路径保留在既有 `queue_limit_returns_429_with_retry_after`（不改）。
3. `test/gateway_e2e_test.cc`：三个非 loopback 用例显式打开限流（`MORTRED_RATE_LIMIT=enforce`），断言在"公开监听 + 限流开启 + 有凭据"下能启动并执行鉴权语义；既有的 `NonLoopbackWithoutAnyAuthRefusesToStart` 继续作为 fail-closed 负例保留。
4. `src/server/server_config_schema.h`：在 `rate_limit_qps` 所在的可选键说明处补一行注释，写明它**已不再参与模型侧准入**、gateway 的 GCRA 是唯一权威（`base_server_impl.h` 里的同义注释保留），使配置面不再暗示模型进程会限流。

### 否决

- 改产品去恢复模型侧限流：与 gateway 唯一权威的设计决定相反，且会让两份限流状态互相漂移。代价最大、收益为负。
- 把 429 用例放宽成"200 或 429 都算过"：等于用放宽断言换绿，`cv-qa` 白盒会直接判 `fail`。
- 只跳过/删除三个用例：会让 gate 失去对这三段契约的覆盖，也不满足完成标准里"CI 仍要检查契约一致"的要求。

## 改动面

- `test/request_envelope_unittest.cc`（含 `images_shape_errors` 的双违规断言，顺序无关写法）
- `test/server_e2e_contract_test.cc`（两条改名用例：`model_side_rate_limit_qps_is_ignored_so_admission_stays_200`、`healthz_and_ready_are_exempt_from_auth_in_both_configs`）
- `test/gateway_e2e_test.cc`（三个正例加 `MORTRED_RATE_LIMIT=enforce`；**另按第 3 步条件**给三个负例 `NonLoopbackWithoutAnyAuthRefusesToStart`、`WildcardListenWithoutExposeRefusesToStart`、`NonLoopbackWithoutMetricsTokenRefusesToStart` 加 `MORTRED_RATE_LIMIT_ALLOW_DISABLE=1`，使其抵达各自要验证的守卫）
- `src/server/server_config_schema.h`（注释，且注释行本身须含字面量 `rate_limit_qps` 以满足 A4）

## 明确不改

- 任何限流/鉴权/启动守卫的产品逻辑；`gateway_app.cpp`、`base_server_impl.h` 的行为分支。
- `docs/openapi.json` 与 `scripts/gen_openapi.py`：本轮不动生成物（`rate_limit_qps` 是 server toml 键，不出现在 OpenAPI 的请求/响应契约里；若 RD 发现 OpenAPI 文本里确有与之矛盾的断言，停下来记入「新增问题」而不是顺手改生成物）。
- 不新增测试框架、不改 CMake 目标清单。

## 契约

- 模型 HTTP 面：`rate_limit_qps` 不影响准入；队列打满仍返回 429 + `Retry-After`（既有用例覆盖）。
- gateway：非 loopback + 关闭限流 → 拒绝启动（fail-closed 保持）；开启后正常启动并执行 key/静态 token 鉴权。
- 请求信封：`items[i].data` 是**解码后的字节**，origin 标记来源；非法 base64 产生 `/images/<i>` violation。

## 性能与资源

无产品性能影响（只改测试与注释）。

## 验收标准

- A1 `./_bin/request_envelope_unittest` 退出码 0，`[  PASSED  ] 17 tests`。
- A2 `./_bin/server_e2e_contract_test` 退出码 0，45 例全过（含改名后的两条）。
- A3 `./_bin/gateway_e2e_test` 退出码 0，49 例全过。
- A4 `grep -n "rate_limit_qps" src/server/server_config_schema.h` 输出中含说明其"不再参与模型侧准入、gateway 为唯一限流权威"的注释行。
- A5 问题分支提交只包含上述 4 个文件；`git show --stat <commit>` 不含 `cowork/ledger/`、不含 `build/`、`build-full/`、`_bin/`、`_lib/` 产物。
- A6 三处断言是"改成与产品一致"而非"放宽"：`rate_limited_returns_429` 改名后的用例必须**断言两次都是 200**；`images_shape_errors` 必须断言两条 violation 的 pointer。

## 测试义务

- 每条验收标准一行；A1–A3 给出命令与真实退出码。
- 白盒：核对改名后的用例断言与 `parsed_request.h`/`gateway_app.cpp` 的实际分支一致；确认没有把断言放宽成"任一状态码"。
- 反例：确认 `queue_limit_returns_429_with_retry_after` 仍通过（模型侧卸载路径未被顺手削弱）。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10
cmake --build build      --target request_envelope_unittest server_e2e_contract_test -j8   # 期望 0
cmake --build build-full --target gateway_e2e_test -j8                                    # 期望 0
./_bin/request_envelope_unittest    ; echo $?   # 期望 0
./_bin/server_e2e_contract_test     ; echo $?   # 期望 0
./_bin/gateway_e2e_test             ; echo $?   # 期望 0
grep -n "rate_limit_qps" src/server/server_config_schema.h
```

注意：三个二进制都必须在仓库根（worktree 根）作为 CWD 运行；`build-full` 与 `build` 共用 `_bin/` 输出目录时后构建的会覆盖同名二进制，按上表分两次构建即可。

## RD 不得自行决定

- 不得修改产品限流/鉴权/守卫逻辑来让测试通过。
- 不得放宽或删除断言、不得跳过用例。
- 不得改动 `docs/openapi.json`、`scripts/gen_openapi.py`、CMake 目标清单。
- 不得把 `cowork/ledger/`、构建产物或 `src/common/mortred_version.h` 带进提交。

## 第3步确认（单一执行方模式）

三个角色线由通用子代理按角色卡分节给出（非 Cursor 角色 agent），均只读核验、未构建未跑测试。

| 角色 | 结论 | 条件 | 方案版本 | 处置 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | A4 的 grep 必须命中**含字面量 `rate_limit_qps`** 的注释行；A2/A6 写明两条改名后的用例名 | v1 | 已并入：见 A4 与改动面；S6 已定名 |
| cv-rd | 同意 | `images_shape_errors` 双违规按实际追加顺序（`[0]="/images/1"`、`[1]="/images/0"`）或顺序无关断言；三个非 loopback **负例**需补 `MORTRED_RATE_LIMIT_ALLOW_DISABLE=1`（或 `enforce`），否则它们在限流守卫处就返回 false，不再验证各自注释声称的守卫 | v1 | 已并入：RD 已改为顺序无关断言；负例补 env 已并入改动面与 S7 |
| cv-tech-lead | 同意 | 契约里把 `/images/<i>` 说明为 compacted index；`{"images":[5,"ok"]}` 的错位指针记入「新增问题」，不写进测试；负例补 env 后仍只改测试 | v1 | 已并入：S8 记为新增问题；负例仅加 env，不动产品分支 |

### 第3步确认带来的正文修订（v1 → v1.1）

- **S6（新增验收）**：两条改名用例名固定为 `model_side_rate_limit_qps_is_ignored_so_admission_stays_200` 与 `healthz_and_ready_are_exempt_from_auth_in_both_configs`；A2 的 45 例计数不含改名影响。
- **S7（新增验收）**：三个非 loopback 负例在"限流关闭但显式允许"下仍因**各自注释声称的原因**拒绝启动；验收=`./_bin/gateway_e2e_test` 退出 0，且这三个用例名仍在。
- **S8（新增问题，不写进测试）**：`parsed_request.h:130-131` 遍历的是 `decoded.value.images`（只含通过类型检查的字符串），因此 base64 违规的 `/images/<i>` 是**压缩下标**：`{"images":[5,"ok"]}` 会报 `/images/0` 而真实下标是 1。本设计只用 `["ok",5]`（非法元素在末位，二者恰好一致）；错位指针记入 `discovered.md`，不在测试里固化为契约。
- **引用更正**：raw-body 期望解码后字节的用例是 `test/request_envelope_unittest.cc:235`（不是 `parsed_request.h:235`，该文件只有 216 行）。
- **顺序更正**：违规追加顺序为 `violations[0]="/images/1"`（`request_envelope.h:118-124`，decode 阶段）、`violations[1]="/images/0"`（`parsed_request.h:133-137`，base64 阶段）；设计初版文字列反了，RD 按顺序无关写法实现。
- **已核验成立的前提**：`MORTRED_RATE_LIMIT` 取值 `off|shadow|enforce`（`gateway_app.cpp:852-859`）、`MORTRED_RATE_LIMIT_ALLOW_DISABLE` 为复位开关（`:874-876`）；`server_config_schema.h:85` 确在 `server_optional_int_keys()` 中；`docs/openapi.json` 无 `rate_limit_qps` 字面量，故"不改生成物"成立。
