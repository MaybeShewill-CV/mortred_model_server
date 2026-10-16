# 交付队列

- 轮次: 13
- 版本: v1
- 性质: 长期根本修复（非 band-aid）；顺序锁定；无 packing/Docker（systemd KillMode 例外）
- 加权分上限: P0 上限 7.9
- 审计基线: `fb982277`
- 来源: 用户指令 Round 13 cv-cowork — triage-defect-cards P0-2..P1-8

## P0

| id | 问题 | 改完后能观察到的变化 |
| --- | --- | --- |
| P0-2 | TRT StagingSlotGuard abandon 未置位；D2HSlotGuard 无 abandon | 错误路径 sync-then-release / abandon，无 pinned slot UAF |
| P0-3 | http_infer_rps.py 默认 warmup 崩溃 | 默认 CLI 可跑；self-test 在 parse 后；smoke 覆盖默认路径 |
| P0-4 | handler_threads=50 × 多 id 无容量模型 | 默认下调并与 workers 绑定；容量表；RSS 维度若可行 |

## P1

| id | 问题 | 改完后能观察到的变化 |
| --- | --- | --- |
| P1-5 | superpoint `>=` 非严格弱序 | 比较器用 `>` 或 `<` |
| P1-1 | max_queue_depth 默认无界 | 正默认；缺键 toml 补齐；满时 429+Retry-After |
| P1-4 | 相对 MORTRED_CONTROL_CONFIG | 相对路径相对 project_root 解析后再 inject/chdir |
| P1-6 | imdecode 后才限尺寸 | JPEG SOF/PNG IHDR 等头探测后、全量 decode 前拒绝 |
| P1-7 | SIGKILL supervisor 孤儿占 GPU | KillMode=control-group（或等价）+ 孤儿自退 |
| P1-2 | YOLO unmap↔NMS 顺序分叉 | v5/v6/v7 与 v8 统一；更广 matting/seg 记 residual |
| P1-3 | 模型 /metrics 不接受 METRICS_TOKEN | /metrics 接受 MORTRED_METRICS_TOKEN |
| P1-8 | golden skip 本地像 pass | 本地缺权重 exit 77 等；presets/labels SSOT 若需 |

## 确认

| 角色 | 结论 | 记录 |
| --- | --- | --- |
| 用户 | 同意并锁定顺序 | 用户指令 2026-10-16 Round 13 |
