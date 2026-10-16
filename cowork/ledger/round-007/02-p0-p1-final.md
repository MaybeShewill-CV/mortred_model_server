# 交付队列

- 轮次: 7
- 版本: v1
- 性质: 长期根本修复（非 band-aid）；顺序锁定；无 packing/Docker

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | CI 把 check_version_single_source 当成 check_consistency 的位置参数 | Repository consistency checks 一步 | `python3 scripts/check_consistency.py scripts/check_version_single_source.py` → argparse unrecognized；版本断言未作为独立 fail-closed 步运行 | 独立 step 跑 `python3 scripts/check_version_single_source.py`；失败即红；consistency 不再吞该路径 | 用户锁定 Round 7；ci.yml |
| P0-2 | ApiKeyManager::load 不解析 max_queue / worker_budget | api_keys.toml 声明租户配额 | 字段存在且 gateway 有 enforce 路径，但 load 只读 rate_limit_qps/max_inflight；声明值永远为 0（死字段） | load 解析 max_queue + worker_budget；example 与运行时一致 | api_key_manager.h；conf/api_keys.toml.example |

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | NvmlGpuProbe 空实现；shared_gpu_probe 未接入 sampler/occupancy/supervisor | GPU 指标 / occupancy | NVML stub 总是 fail；sampler/supervisor 仍各自 fork nvidia-smi | 真 NVML（dlopen）或诚实失败；shared_gpu_probe 驱动 sampler/occupancy/supervisor；去掉可去的重复 smi fork | gpu_probe.h；gpu_sampler.h；supervisor.cpp |
| P1-2 | max_queue 非驻留；worker_budget 仅 soft | gateway 上游转发 | acquire queue 后立刻 release；worker_budget 只打 soft metric | queue 持有至 upstream callback；worker_budget 硬拒绝 429 | gateway_app.cpp；api_key_manager.h |
| P1-3 | verify_gpu_gate_evidence 未入 CI/release；版本检查须 fail-closed | fork / release | 脚本与 example 存在但 CI 未调用；版本检查绑错 consistency | CI/release 跑 verify（forks `--allow-missing-artifacts`）；版本检查独立 fail-closed | verify_gpu_gate_evidence.py；ci.yml；docs/ci |
| P1-4 | 产品路径残留 FixedWindow 成员与陈旧 layer 注释 | base_server_impl / rate_limit 头 | `_m_rate_limiter` 仍成员；(void) 占位；注释仍提 model-fixed-window | 删除产品成员；注释与 headers 仅保留测试/历史说明且不冒充产品 layer | base_server_impl.h；gcra.h；headers.h |

## P2
（无）

## 附录
| id | 陈述 |
| --- | --- |
| 用户禁止 | packing / max_batch_size；Docker USER / 本机 Docker 测试。不入列、不改产品。 |

## 确认
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意（供给并锁定本队列；顺序锁定） | v1 | 用户指令 2026-10-16 Round 7 |
