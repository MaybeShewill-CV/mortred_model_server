# 交付队列

- 轮次: 6
- 版本: v1
- 加权分上限: 8.9（未关闭 P1；无 P0）
- 性质: 长期根本修复（非 band-aid）；顺序锁定

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | Rate-limit authority：公网限流权威分裂，模型侧仍走 FixedWindow 产品路径。 | Gateway 为唯一公网 limiter；全量 GCRA；去掉模型产品路径 FixedWindow；统一 429 头；RateLimiter 抽象；legacy\|unified 后删除 FixedWindow 产品路径。 | Gateway GCRA；model FixedWindowRateLimiter；API-key FixedWindow；429 头虽已有 layer 标签但仍双算法。 | 默认 unified：公网限流仅 gateway；模型进程不再按 peer FixedWindow 拒流；API-key QPS 为 GCRA；统一 RateLimit-* / Retry-After / X-Mortred-RateLimit-Layer；存在 RateLimiter 抽象；产品路径无 FixedWindow。 | 用户锁定 Round 6；gcra.h；rate_limiter.h；gateway_app.cpp；base_server_impl.h；api_key_manager.h |
| P1-2 | Multi-tenant quotas：租户配额对象不完整，未在 gateway 一次强制执行。 | API key → tenant quota（GCRA qps、max_inflight、max_queue、optional worker_budget）全部只在 gateway 执行；metrics + noisy-neighbor 基线测试。 | 仅 rate_limit_qps（FixedWindow）+ max_inflight；无 max_queue / worker_budget；无 noisy-neighbor 测试。 | api_keys 可声明完整 quota；gateway 单点 enforce；/metrics 可见 per-tenant 拒流/排队；noisy-neighbor 单测证明一租户打满不饿死另一租户预算。 | 用户锁定；api_key_manager.h；gateway_app.cpp |
| P1-3 | GPU probe：双份 nvidia-smi fork，无 NVML 主路径，无可插拔探针。 | 可插拔 GpuProbe；NVML 主、nvidia-smi 备；禁止 waitpid(-1) 收割探针；fail-closed；fake-probe 测试；occupancy/sampler 共享指标源。 | occupancy_gate 与 gpu_sampler 各自 fork nvidia-smi；无 NVML；无 GpuProbe 接口。 | GpuProbe 接口 + NvmlProbe + SmiProbe；默认 NVML→smi fallback；fake probe 单测；sampler 与 occupancy 共用同一探针/指标；失败 fail-closed。 | 用户锁定；occupancy_gate.h；gpu_sampler.h |
| P1-4 | Release/CI reproducibility（非 Docker）：fork 无 GPU runner 时看不到可复现证据。 | 单一版本源；可声明 GPU/gate 证据（artifact digest + reproduce commands）。 | VERSION=1.0.0 已对齐；GPU job 依赖 vars；fork 无 digest/reproduce 清单。 | VERSION 单一源断言进 CI；docs/ci 或 evidence 清单含 artifact digest 字段与 reproduce 命令；无 GPU runner 时仍可按清单复现/核验声明。 | 用户锁定；CMakeLists；mortred_version.h；ci.yml；非 Docker |

## P2
无

## 附录
| id | 陈述 |
| --- | --- |
| 用户禁止 | packing / max_batch_size；Docker USER / 本机 Docker 测试。不入列、不改产品。 |

## 确认
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意（供给并锁定本队列；跳过完整 Step-1 重评；顺序锁定） | v1 | 用户指令 2026-10-16 Round 6 |
