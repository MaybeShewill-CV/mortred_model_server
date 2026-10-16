# 第2步评审

- 轮次: 4
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 评审 | 理由 |
| --- | --- | --- |
| P0-1 | 保留 | 已 gitignore conf/local/；补强停跟踪断言与 init-edge 注释即可。 |
| P0-2 | 保留 | 与 GatewayApp / listen_policy 对齐；默认 enable 或公网 fail-closed。 |
| P0-3 | 保留 | 关闭门禁须显式危险开关，spawn 不得静默放行。 |
| P0-4 | 保留 | 折叠 stash 中 image_postprocess_context WIP。 |
| P1-1 | 保留 | 代码内文档 + 必要对齐；不强制一次重写两边算法。 |
| P1-2 | 保留 | 折叠 nvidia-smi reaper WIP。 |
| P1-3 | 保留 | 优先构建时生成或瘦身，保住 gen_openapi --check。 |
| P1-4 | 保留 | 首切片：每 key 队列深度上限即可。 |
| P1-5 | 保留 | 诚实 semver（建议 1.0.0 级深度）。 |

## 不同意的条目
无