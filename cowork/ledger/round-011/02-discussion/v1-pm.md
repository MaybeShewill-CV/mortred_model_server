# 第2步表决

- 轮次: 11
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| P0-2 | 保留 | 换机的人不设 `MORTRED_PACK` 时，能凭 `doctor --strict` 的退出码、点名的 engine 路径和 `next:` 行核对全部 39 个 id。选中 pack 后 engine 缺失已经失败，这一段复用；默认 pack 下仍退出 0 是扩展。 |
| P0-3 | 保留 | 调用 `yolov7` 与 `yolov7x` 的人，能凭同一请求的 HTTP 500、`status` 和 `status_str` 看出契约失败。其余路径已有的 `status` 6 是复用。 |
| P1-7 | 保留 | 照抄请求的人能对每个 id 的 path 校验一条请求示例和一条 200 响应；`yolov8n`、`yolov8l`、`yolov8x`、`landmarks`、`superpoint` 的字段都在可观察结果里。契约页已有的 7 个示例是复用，OpenAPI 示例是扩展。 |
| P1-8 | 拆开 | 无用例 id 的 CPU 失败条件、四个只有指纹的分割/抠图要改成掩码重合失败、产品 toml 的 TRT 用例仍按该 id 的 CPU 规则失败，要分三次看。已有 CPU 门和 `yolov8s` 的 TRT 门是复用。 |
| P1-9 | 保留 | 交基准的人能逐 id 在同一行看到硬件、可直接执行的命令、指标和日期。 |
| P1-10 | 保留 | 按 README 做 GPU 源码构建的人，能对照两份代码块的 GPU 可执行行是否相同，并仍看到两条 `cmake --preset`。preset 已一致的部分是复用。 |
| D-7 | 拆开 | 拉起后是否仍监听 `127.0.0.1` 且带上 token，操作的人能验收。exec 前那些调用是否消失，操作的人从外部一次看不出来。 |
| D-8 | 保留 | 用默认镜像交付的人能看 `id -u` 的结果，或在边界文档里读到必须 root 的原因。 |
| D-9 | 保留 | 鉴权失败后紧接着发合法推理的人，能凭 401 以及随后是转发还是 429 验收配额。`enabled = false` 与无 inference 权限的两次 401 是同一类配额观察。 |
| D-10 | 保留 | utilization 为 N/A 的一次探测，操作的人能看 `available`、offline 字样，以及 CLI 是否把 -1 印成利用率。整次失败已标 offline 是复用，N/A 是扩展。 |
| D-11 | 保留 | 按部署文档找模型的人，能核对两处 cpu 名称是 `resnet`，以及 Bench-only 列表里没有 `msocrnet`。README 目录里的这两个 id 是复用。 |

## 不同意的条目
P1-8 拆成三条：一条只验收点名的无用例 id，检测失败于同类且 IoU < 0.5，分割和抠图失败于掩码重合低于测试写明的阈值，embedding 和深度失败于测试写明的距离或误差；一条只验收 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512`，失败于掩码重合不足；一条只验收点名的 TRT id，加载产品 toml，失败条件与该 id 的 CPU 规则相同。

D-7 拆成两条：一条只验收模型仍监听 `127.0.0.1`，并带上 `MORTRED_AUTH_TOKEN` 或网关的 `MORTRED_METRICS_TOKEN`；另一条只写操作的人能看见的失败。exec 前调用是否消失另列，并改成一次能看见的结果。
