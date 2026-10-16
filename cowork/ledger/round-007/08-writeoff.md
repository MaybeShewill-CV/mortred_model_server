# 关闭记录

- 轮次: 7
- 已关闭: P0-1、P0-2、P1-1、P1-2、P1-3、P1-4
- 阻塞: 无
- 未阻塞的开放项: 相对本轮起点减少

## 已关闭的问题陈述
| id | 模块 | 问题陈述 |
| --- | --- | --- |
| P0-1 | ci | check_version_single_source as own fail-closed CI step |
| P0-2 | api_keys | ApiKeyManager::load parses max_queue + worker_budget |
| P1-1 | control | Real NVML GpuProbe + shared_gpu_probe wiring |
| P1-2 | gateway | max_queue residency + hard worker_budget |
| P1-3 | ci/docs | verify_gpu_gate_evidence in CI/release (fork allow-missing) |
| P1-4 | server | Remove dead FixedWindow product members/comments |

## 仍开放的问题
（无）

## 发现但未入本轮
| 陈述 |
| --- |
| check_consistency 仍要求 openapi_doc.h embed 与 docs/openapi.json 同步，但 header 已是 runtime loader（预存噪音；非本轮队列） |
