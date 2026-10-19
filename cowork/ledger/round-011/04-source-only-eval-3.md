# 第三次评估：排除 A（内存测量）与 B（内存优化/共享）后的 P0/P1

- 基线：`sub-agent-co-work @ 9a2689d2`（源码未变，仅账本提交）
- 方法：只读源码 + 本机实测；A/B 两条工作线按用户决定暂缓，**本文不含任何"降内存/共享"类建议**
- 目的：在我此前**未逐行读过**的区域里找新的 P0/P1

## 0. 先撤回两个我自己的怀疑（查证后不成立）

| 怀疑 | 查证结果 |
| --- | --- |
| async 任务表无界保留、每次轮询深拷贝导致内存增长 | **不成立**。`async_job_table.h:86-88` 有 `max_queue=16`、`job_ttl_ms=300000`、`max_completed=100`，`:348-378` 实现了 TTL + LRU 淘汰；`:270` 的逐次拷贝是有意为之（"endpoint stays repeatable"） |
| 令牌文件权限过宽（实测 `trust.env` 为 777） | **不成立**。`scripts/mortredctl_init-trust.sh:54` 在建文件前 `umask 077`，承诺的 mode 600 已实现；777 是 WSL DrvFs 挂载不保留 POSIX 权限的假象 |

## 1. 本轮审计过、结论为"做得对"的区域（不计问题）

- **限流身份判定**（`rate_limit/subject.h:423-462`）：非可信对端**完全忽略** `Forwarded`/`X-Forwarded-For`；可信时**从右往左**取第一个非可信 hop，遇不可解析 hop 回退 peer 桶；非法 peer 用 `family=0` 独立桶。这是防伪造的正确算法。
- **网关转发头**（`gateway_app.cpp:459-465`）：默认拒绝，仅放行 `X-Mortred-Params`/`X-Mortred-Options`/`X-Request-ID` 三个。
- **日志轮转**（`log_buffer.h:24-29,65-66,110`）：超限时 `path → path.1`，总占用有上界（2×`log_rotate_mb`/子进程）。
- **drain 与销毁**（`base_server_impl.h:68-100`、`async_endpoints.h:132-152`）：共享 deadline、超时打日志；注释明确列出"被 executor 弹出但尚未访问队列"的微秒级残余窗口及其缓解。属**已知残余**，非新缺陷。
- **API key 存储**（`api_key_manager.h:33,37,65,97,161`）：**SHA-256 哈希落盘**、按哈希查找、`list` 不回显哈希、示例文件提示 `chmod 600`。
- **管理面板鉴权**（`supervisor_app.cpp:639-649`）：`/api/v1/*` 全部要 bearer（仅 health/version 公开）；静态壳（`/`、`index.html`、`app.js`、`style.css`、`favicon`）免鉴权但**不含数据**，且路由是 5 个精确路径白名单，无遍历面。
- **构建文件列表**：无源码 `file(GLOB)`；`cmake/VendoredDeps.cmake:1` 明确写着"replaces file(GLOB)"。

## 2. 仍存在的 P1（不含 A/B）

| # | 问题 | 证据 | 修法 | 量级 |
| --- | --- | --- | --- | --- |
| P1-1 | `stop_order_record.py --check` 对出厂 unit 返回 **1**：检查器要求恰一行 `KillMode=process` 且把 `control-group` 当必须失败的负例，而 `deploy/mortred-supervisor.service:46` 已是 `control-group` | 我实跑 `check_exit=1`；`stop_order_record.py:141,143,168` vs `:46` | 更新检查器期望（并把两个负例改成 `mixed`/`none`） | 小时级 |
| P1-2 | **两把不同令牌同用途**：`conf/local/trust.env` 与 `conf/local/mortred_metrics_token` 值不同（SHA 不同），supervisor 用其中一路注入模型子进程；运维拿错一把就得 401 且无提示 | 我实测：模型 `/metrics` 用 trust.env 的 metrics token → 401，用 internal token → 200；`supervisor.cpp:493-495` 注入的是 `_gateway_metrics_token` | 统一来源，或两者不一致时 fail-closed 拒绝启动并在错误里点明来源 | 小时级 |
| P1-3 | **`_bin`/`_lib` 默认固定在仓库根，被所有构建目录共享**：建一个 `-DMORTRED_BUILD_PROFILE=cpu` 目录会**静默替换** TRT 版二进制；随后正确 profile 的 `cmake --build` 因"目标比对象新"而**不重链** | `CMakeLists.txt:32-35`；我实测两次：cpu 构建后运行期报 `tensorrt backend is not compiled into this build (cpu profile)`，gpu 构建 `exit 0` 却无 `Linking` 行 | 输出目录改为 `${CMAKE_BINARY_DIR}/_bin`（或把 profile 编进路径） | 半天 |
| P1-4 | **相对路径 `MORTRED_CONTROL_CONFIG` 让 gateway 静默 crash-loop**：supervisor 正常，gateway 子进程 CWD 是 `_bin` 打不开相对路径，5 次重启后 `failed` | `supervisor_app.cpp:930` 仅 `lexically_normal()`；`control_config.cpp:120` 直接报 `cannot open control config` | 入口转绝对路径 + 一条用例 | 小时级 |
| P1-5 | **占用门在 GPU 采样失败时部分 fail-open**：`live.available == false` 时，"指纹不匹配"与"空闲显存低于 stamp"两项检查被整体跳过（只剩静态预算求和） | `occupancy_gate.h:484`、`:523` 都把检查放在 `if (in.live.available)` 内 | 采样不可用时改为**拒绝启动**（或要求显式 `--allow-unknown-gpu`） | 小时级 |
| P1-6 | **API key 用无盐 SHA-256**，而示例教用户 `echo -n "your-secret-key" \| sha256sum` —— 人工口令在文件泄漏后可离线爆破 | `api_key_manager.h:33,37`；`conf/api_keys.toml.example` 的生成提示 | 文档要求 ≥32 字节随机密钥；或改 Argon2/scrypt + 每键 salt | 半天～1 天 |
| P1-7 | **ctest 语义下 skip 仍非失败**：`SKIP_RETURN_CODE 77` 让"缺权重"渲染为 `Skipped`，本地 `check` 全绿可能带跳过；只有 cpu-profile 那一个 CI 作业写了 "Skip is fail" | `test/CMakeLists.txt:356-358`；`ci.yml:368` vs `:972` | tests 作业增加汇总步骤：`skipped > 0` 即红 | 小时级 |
| P1-8 | **`src/common/mortred_version.h` 生成物入库**：`configure_file` 写回源码树，检出即 dirty | `CMakeLists.txt:16-17`；`git ls-files` 命中 | 生成到 `${CMAKE_BINARY_DIR}` 并调整 include 路径；从索引移除 | 半天 |
| P1-9 | **FULL 档 `check` 是否能跑完仍未实证**：两个 jpeg 目标已实证可构建，`enlightengan` 缺的符号也已补齐，但没跑过 FULL 全量 | 我 `build-eval`（tests-only）构建两目标 `exit 0`；enlightengan 仅在 `MORTRED_BUILD_FULL` 下注册（`test/CMakeLists.txt:127-131`） | 跑一次 FULL 构建 + `check` | 半天（多为机时） |

## 3. 本轮没有找到新的 P0（这是结论，不是省略）

排除 A/B 后，我在上述**全部新读区域**里没有发现新的 P0 级缺陷——没有新的进程崩溃路径、没有鉴权旁路、没有无界资源增长、没有构建门禁的红。原先的四个 P0 中：

- 两个已修并复验（`check` 的两个不可链接目标；TSan 标签单一真源且有守卫脚本）；
- 两个属"半修/未实证"（skip 语义 → P1-7；FULL `check` → P1-9）。

**剩下的 P0 级问题不是缺陷，而是"保证强度"**：项目现在能跑绿，但没有**覆盖率下限**与**性能回归门**，所以"变差不会红"。对"同类天花板/企业可直接落地"这个目标，我认为这仍是 P0 级缺口（它决定外部评审是否信任你的绿灯），但它的性质与修 bug 不同，需要单独立项。

## 4. 局限

- 仍未逐行读：`src/models` 全部 16 族（约 22.9k 行）、`batch_collector.h`、`worker_pool.h`、`src/factory` 各 task 头、GCRA 的具体算法、`cli_app.cpp` 的各子命令。这些区域的分数来自抽样 + 运行行为。
- 未实跑 GitHub Actions；`ci` 的结论基于构建列表一致性与守卫脚本。
- 本轮所有实测都在 `9a2689d2` 上完成，未改动源码；`_bin` 已恢复为 gpu 版（`mortred-model-server.out` 21:29 重链）。
