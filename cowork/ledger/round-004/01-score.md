# 评分

- 轮次: 4
- 方式: 用户供给队列（跳过完整第 1 步九模块重评）
- 基线: sub-agent-co-work @ d5d8a6ac（第 3 轮 done，加权分 9.1）
- 说明: 用户批准独立评估后的企业/天花板问题集，并明确删除原独立评估 P1-1（产品 max_batch_size=1）与 P1-5（GPU CI 不可移植）。本文件仅作入列引用，不重算九模块分数。

## 用户供给入列摘要

### P0
1. TLS 私钥在树内 conf/local/edge/tls/（ca.key、server.key 及证书）— 停跟踪、gitignore、仅经 init-edge 本地生成。
2. Gateway 限流默认关闭 conf/mortred.toml [gateway.rate_limit] enabled=false — 企业/边缘合理默认或公网 listen 时 fail-closed，与 GatewayApp 行为一致。
3. Occupancy 门禁可被 occupancy_enforcement_disabled / env 关闭 — 生产 spawn 不得静默超额（仅 doctor-strict 不足）。
4. GPU zero-copy 路径丢弃 input.params 进 InferenceContext — 修 backend_cv_model 及相关 WIP，含测试。

### P1（保留）
- FixedWindow vs GCRA 限流语义分裂 — 对齐或代码内文档 + gateway/server 对运维足够一致。
- Occupancy 经 nvidia-smi fork vs NVML — 硬化竞态（occupancy_gate WIP）趋向 fail-closed。
- 巨型 openapi_doc.h 嵌入 — 减膨胀/漂移风险且不破坏一致性测试。
- 多租户仅 API-key QPS — 最小每租户队列/worker 预算或明确首切片。
- CMake VERSION 0.1.0 — 抬到与深度匹配的诚实 semver。

### 用户删除（不入列）
- 原独立评估 P1-1：产品 max_batch_size=1
- 原独立评估 P1-5：GPU CI 非可移植

## 本轮起点
- 起点加权分沿用第 3 轮复核 9.1；有未关闭 P0 时加权上限 7.9。