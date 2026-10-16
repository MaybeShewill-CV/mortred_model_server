# 交付队列

- 轮次: 14
- 版本: v1
- 性质: 长期根本修复（merged residuals）；顺序锁定；无 packing/Docker
- 审计基线: `50850251`
- 来源: 用户指令 Round 14 cv-cowork — 6 residuals

## P0

| id | 问题 | 改完后能观察到的变化 |
| --- | --- | --- |
| P0-1 | TRT `buffer.ensure` 失败未置 abandon | H2D/output ensure 失败路径 `abandon=true`；有结构/注入覆盖 |
| P0-2 | 容量只有文档，无 RSS/RAM 门 | calibrate 写 per-id rss/thread/fd；occupancy_gate 有 host RAM 维度 |

## P1

| id | 问题 | 改完后能观察到的变化 |
| --- | --- | --- |
| P1-1 | TSan 三处列表可漂；golden 混合 skip 当 pass | SSOT 脚本闸；empty/mixed → exit 77 + SKIP_RETURN_CODE |
| P1-2 | 非 JPEG/PNG 仍 decode 后才限尺寸 | BMP/WebP/GIF/TIFF 头探测后拒绝 |
| P1-3 | 同族行为分叉 | libface 源空间 NMS+clip；score 可降；matting/seg 族尾对齐；GPU 分叉有文档 |
| P1-4 | supervisor 子环境 drop-list | allowlist 构建；模型可持 METRICS_TOKEN |

## 确认

| 角色 | 结论 | 记录 |
| --- | --- | --- |
| 用户 | 同意并锁定顺序 | 用户指令 2026-10-16 Round 14 |
