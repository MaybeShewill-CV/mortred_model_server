# 交付队列

- 轮次: 8
- 版本: v2
- 性质: 长期根本修复（非 band-aid）；顺序锁定；无 packing/Docker
- 加权分上限: P0 上限 7.9
- 审计基线: `aa761e70`（Round 7 已全部合入；Round 8 候选审计队列由用户改锁为本 v2 follow-up）

## P0

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | check_consistency 仍按 embedding 同步 openapi_doc.h，与薄 loader 冲突，门禁恒红 | clone 后跑 consistency / gen_openapi --check；CI quality-gates | `openapi_doc.h` 已是 runtime loader（`load_openapi_json`），`scripts/gen_openapi.py --check` 已按 loader 形状断言；但 `scripts/check_consistency.py` 的 `check_openapi_doc_header_sync` 仍要求 header 内嵌完整 `docs/openapi.json`，结构上不可能相等 → 退出 1 | `python3 scripts/check_consistency.py` 退出 0；`python3 scripts/gen_openapi.py --check` 退出 0；consistency 断言 loader 形状（含 `load_openapi_json`）并调用/对齐 gen_openapi --check；可选 anti-embed 守卫拒绝大块嵌入。一次外部验收：重跑这两条命令 | `src/server/openapi_doc.h`；`scripts/check_consistency.py`；`scripts/gen_openapi.py` |

## P1

| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | worker_budget 硬门在 max_inflight<=0 时失效 | api_keys 只设 worker_budget、不设 max_inflight | gateway 用 `inflight` 计数比对 worker_budget，但 `try_acquire_inflight` / `release_inflight` 在 `max_inflight<=0` 时不增减计数 → budget 永远看不到占用 | `max_inflight=0` 且 `worker_budget=N>0` 时，第 N+1 个并发请求得 429 + `X-Mortred-RateLimit-Layer: api-key-worker-budget`。实现：独立计数或在 budget>0 时始终 track inflight | `src/control/api_key_manager.h`；`gateway_app.cpp` |
| P1-2 | 缺少 max_queue / worker_budget 的 TOML load 回归 | 改解析或 schema 时 | Round 7 已接通 load 解析两字段，但单测未覆盖 → 回退可静默通过 | 单测写含 `max_queue`/`worker_budget` 的 toml，load 后字段等于声明值；缺省为 0 | `api_key_manager.h`；`test/api_key_manager_unittest.cc` |
| P1-3 | 缺少 gateway e2e：queue 驻留与 worker_budget 429/layer | 租户配额上线验收 | 产品路径已有 residency + hard budget，但 e2e 未断言并发占位与 layer 头 | gateway e2e：max_queue 驻留期间超额 429 + layer=`api-key-max-queue`；worker_budget（可与 max_inflight=0 组合）超额 429 + layer=`api-key-worker-budget` | `test/gateway_e2e_test.cc` |
| P1-4 | NvmlGpuProbe 每次 sample 都 dlopen/dlclose | GPU 采样周期 | `sample()` 内每轮 dlopen/init/shutdown/dlclose | 进程生命周期持有 dlopen handle；sample 只做查询；析构 shutdown+dlclose | `src/control/gpu_probe.h` |
| P1-5 | CI evidence 路径可改进；headers.h layer 注释笔误 | forks 无 GPU runner；429 layer 文档 | verify 已入 CI；schema 保留；`headers.h` Layers 行重复 `api-key-max-queue` | 保持 schema_version=1；无 GPU 仍可验证 schema/version/reproduce/digest；修复 headers 注释去重 | `verify_gpu_gate_evidence.py`；`headers.h` |

## P2

（无）

## 附录

| 类别 | 陈述 |
| --- | --- |
| 用户禁止 | packing / max_batch_size；Docker USER / 本机 Docker 测试。不入列、不改产品。 |
| 未采用 | Round 8 外部审计候选 v1（6 P0 / 7 P1 / 9 P2）。用户在第 2 步直接锁定为本 v2 follow-up 队列。 |
| 已由第 7 轮关闭 | CI version 独立步；ApiKeyManager load 接通；NVML probe 接线；queue residency + hard worker_budget（本轮补 max_inflight=0 缺口与 e2e）；verify_gpu_gate_evidence 入 CI；FixedWindow 产品成员删除 |

## 确认

| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v2 | 用户指令 2026-10-16 Round 8 明确锁定本队列（Steps 2–5） |
| cv-pm | 同意 | v2 | 用户直接锁定；编排方按 skill「用户选定一版」记入 |
| cv-tech-lead | 同意 | v2 | 用户直接锁定；编排方按 skill「用户选定一版」记入 |
| cv-cto | 同意 | v2 | 用户直接锁定；编排方按 skill「用户选定一版」记入 |

## 用户决定

- 日期: 2026-10-16
- 选用版本: v2
- 决定原文: Queue (long-term; no packing/Docker): P0-1 openapi loader consistency; P1-1..P1-5 worker_budget/TOML/e2e/NVML lifetime/CI evidence+headers nit. Steps 2–5, --no-ff.
- 未选用: 外部审计候选 v1
