# 交付队列

- 轮次: 5
- 版本: v1
- 加权分上限: 8.9（未关闭 P1；无 P0）

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | FixedWindow vs GCRA 双语义仍分裂；R4 仅代码注释。 | 对齐或硬化运维可见一致性（超越注释），若仍双轨。 | gateway ShardedIpLimiter/GCRA；server FixedWindowRateLimiter；api-key 亦 FixedWindow；429 头字段不全一致。 | 运维可从 429 响应头（RateLimit-* / Retry-After / X-Mortred-RateLimit-Layer）区分并调参各层；头字段约定一致；保留双算法分层。 | 用户供给 post-R4；src/control/rate_limit/gcra.h；src/server/rate_limiter.h；gateway_app.cpp |
| P1-2 | gpu_sampler nvidia-smi 仍用阻塞 waitpid(pid)，未像 occupancy_gate 硬化。 | 硬化同 occupancy_gate（reaper/ECHILD / fail-closed）。 | probe 末尾 waitpid(pid,&status,0)，进程级 waitpid(-1) 可偷收割；无 WNOWAIT/ECHILD 路径。 | sampler probe 在被偷收割时不误杀回收 pid，超时/不全输出 fail-closed（空样本/离线）；行为对齐 occupancy_gate 探针。 | 用户供给；src/control/gpu_sampler.h；occupancy_gate.h |
| P1-3 | max_inflight 助手已存在但未接入 gateway 请求路径。 | 接入 try_acquire_inflight/release；更新 api_keys 示例；测试。 | ApiKeyManager 有 max_inflight 与 try_acquire/release；gateway_app 未调用；example 无字段。 | 配置 max_inflight>0 时超并发返回可观察 429；请求结束释放；example 与单测覆盖。 | 用户供给；api_key_manager.h；gateway_app.cpp；conf/api_keys.toml.example |
| P1-4 | VERSION 诚实性：CMake 已 1.0.0 但生成头与 release 帮助仍引 v0.1.0。 | 重生成/提交 mortred_version.h=1.0.0；修正 release 帮助。 | mortred_version.h 仍 MORTRED_VERSION "0.1.0"；make_release_tarball.sh 用法示例 0.1.0。 | --version / mortred_version.h 为 1.0.0；release 脚本帮助示例为 1.0.0。 | 用户供给；CMakeLists.txt；mortred_version.h；make_release_tarball.sh |
| P1-5 | Occupancy 单测漂移：policy_off / env_zero 仍假设 soft-off 单独放行。 | 与双门禁一致：soft-off alone 不得 disable。 | OccupancyGateTest.policy_off_allows_trt_without_stamp 与 env_zero_disables_enforcement 仍 EXPECT_TRUE 在仅 soft-off 时。 | 两测与 soft_off_alone_does_not_disable 一致：无 ALLOW_DISABLE 时仍 enforce；有 ALLOW_DISABLE 时才放行。 | 用户供给；test/occupancy_gate_unittest.cc；occupancy_gate.h P0-3 |

## P2
无

## 附录
| id | 陈述 |
| --- | --- |
| 跳过 | OpenAPI embed（R4 P1-3 已关闭）。 |
| 用户删除 | max_batch_size=1；GPU CI 非可移植。不入列。 |

## 确认
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意（供给并锁定本队列；跳过已关闭 OpenAPI embed；删除 max_batch_size=1 与 GPU CI） | v1 | 用户指令 2026-10-16 Round 5 |
