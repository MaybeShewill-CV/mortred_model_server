# 交付队列

- 轮次: 4
- 版本: v1
- 加权分上限: 7.9（未关闭 P0）

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | TLS 私钥出现在工作树 conf/local/edge/tls/（ca.key、server.key 及证书）。 | 仓库与 git 跟踪面：停跟踪、gitignore、仅经现有 init-edge 脚本本地生成；代码注释即可，除非验收强制否则不改正文 markdown。 | conf/local/ 已在 .gitignore，但磁盘上仍有密钥文件；无额外停跟踪断言；生成路径依赖 scripts/mortredctl_init-edge.sh。 | git ls-files 不含任何 conf/local/edge/tls 下的 .key/.pem/.crt；.gitignore 明确覆盖；init-edge 脚本注释写明仅本地生成；一致性检查或自测能证明未跟踪。 | 用户供给队列；scripts/mortredctl_init-edge.sh；.gitignore；conf/local/edge/tls/ |
| P0-2 | Gateway 限流默认关闭：conf/mortred.toml [gateway.rate_limit] enabled=false。 | 企业/边缘默认或公网 listen 时 fail-closed，与 GatewayApp 行为一致。 | 默认 enabled=false；MORTRED_RATE_LIMIT 可覆盖；公网/unsafe 暴露时仍可不启限流。 | 默认配置下 rate_limit 对边缘/企业为开启（或等价 fail-closed：公网 listen 且未显式关闭时拒绝启动）；与 GatewayApp 读配置行为一致；可用配置/启动日志或单测观察。 | 用户供给；conf/mortred.toml；src/control/gateway/gateway_app.cpp；src/common/listen_policy.h |
| P0-3 | occupancy_enforcement_disabled / env 可关闭占用门禁，生产 spawn 可静默超额。 | 硬化生产 spawn：doctor-strict alone 不足。 | occupancy_policy=off 或 MORTRED_OCCUPANCY_ENFORCE=0/false/off 时 occupancy_ready_for_spawn 直接放行。 | 生产路径下关闭门禁不能静默放行超额 spawn（须显式危险开关或 fail-closed）；单测覆盖。 | 用户供给；src/control/occupancy_gate.h；scripts/mortredctl_doctor.sh；scripts/pack_occupancy.py |
| P0-4 | GPU zero-copy 路径把 InferenceContext 建出来时丢掉 input.params。 | backend_cv_model / 相关 WIP；含测试。 | GPU 成功路径手建 InferenceContext 只填 geometry，params 为空，每请求覆盖静默失效。 | GPU zero-copy 成功路径 postprocess 收到与请求一致的 params；单测证明覆盖生效。 | 用户供给；已 stash 的 WIP；src/models/backend/backend_cv_model.h；inference_context.h |

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 限流语义分裂：gateway 用 GCRA，model server 用 FixedWindow。 | 对齐或代码内文档 + gateway/server 对运维足够一致。 | gateway ShardedIpLimiter/GCRA；server FixedWindowRateLimiter。 | 运维能从代码注释/同一语义约定理解两边；或实现侧收敛到可观察的一致行为（文档-in-code 可接受）。 | 用户供给；src/control/rate_limit/gcra.h；src/server/rate_limiter.h |
| P1-2 | Occupancy 经 nvidia-smi fork，相对 NVML 有竞态，可 fail-open。 | 硬化 occupancy_gate WIP，趋向 fail-closed。 | 进程级 waitpid(-1) 收割可偷走 probe 子进程，门禁静默降级。 | probe 在被偷收割时仍能解析完整输出或 fail-closed；单测覆盖 reaper 竞态。 | 用户供给；stash WIP occupancy_gate；test/occupancy_gate_unittest.cc |
| P1-3 | 巨型 openapi_doc.h 嵌入，膨胀与漂移风险。 | 减膨胀/漂移且不破坏一致性测试（可构建时生成或瘦身）。 | 已提交约 9k 行 / ~285KB 的 openapi_doc.h，与 gen_openapi --check 绑定。 | 嵌入体积或生成方式改善后，scripts/gen_openapi.py --check 与既有一致性测试仍通过。 | 用户供给；src/server/openapi_doc.h；scripts/gen_openapi.py |
| P1-4 | 多租户只有 API-key QPS，无每租户队列/worker 预算。 | 最小每租户队列/worker 预算，或明确首切片若过大。 | ApiKeyManager 仅 rate_limit_qps（FixedWindow）；无每租户队列深度/worker 预算。 | 首切片：至少一个可观察的每租户预算（队列或 worker）生效，或代码内明确 scoped 首切片边界与验收。 | 用户供给；src/control/api_key_manager.h |
| P1-5 | CMake project VERSION 仍为 0.1.0，与产品深度不符。 | 抬到诚实 semver。 | project(mortred_model_server VERSION 0.1.0)。 | VERSION 与 mortred_version.h / --version 反映诚实 semver（与深度匹配的主.次.修订）。 | 用户供给；CMakeLists.txt；src/common/mortred_version.h.in |

## P2
无

## 附录
| id | 陈述 |
| --- | --- |
| 用户删除-原P1-1 | 产品 max_batch_size=1。用户指令删除，不入列。 |
| 用户删除-原P1-5 | GPU CI 非可移植。用户指令删除，不入列。 |
| WIP | 基线工作区曾有 param/occupancy/async 等相关未提交改动；已 stash 为 r4-wip-param-occupancy-async-before-designs，按问题分支折叠，不丢失。 |

## 确认
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| cv-pm | 同意 | v1 | cowork/ledger/round-004/02-discussion/v1-sign-pm.md |
| cv-tech-lead | 同意 | v1 | cowork/ledger/round-004/02-discussion/v1-sign-tech-lead.md |
| cv-cto | 同意 | v1 | cowork/ledger/round-004/02-discussion/v1-sign-cto.md |
| 用户 | 同意（供给并锁定本队列；删除原独立评估 P1-1/P1-5） | v1 | 用户指令 2026-10-16 |