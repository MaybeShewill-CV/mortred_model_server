# 分数核算

- 轮次: 11
- 原始综合分: 7.8
- 加权综合分: 7.8
- 上限规则: P1 上限 8.9
- 相对本轮起点: 8.5 到 7.8。销账不改分数。
- 出口: 继续
- 劣势集合: 相对本轮起点减少
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）
- 与角色自算: 一致

模块分保持 `01-modules/` 的记录。加权仍是 776/100 = 7.76，保留一位小数为 7.8。P0-3 已关闭。未关闭的 P0 只剩挂起的 P0-2，不触发 P0 上限。未关闭且未挂起的 P1 仍在，上限 8.9，不改变 7.8。

## 双目标
| 编号 | 状态: 已满足 / 未满足 / 挂起 | 证据或队列 id |
| --- | --- | --- |
| H1 | 已满足 | P1-7 合入 `8f6143544431ba6f7b9d64584cf7d1f6ff92fb28`。39 个 catalog path 各自有请求 example 和 200 example，并按该 path schema 校验。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 id。 |
| H2 | 未满足 | P1-8、P1-11、P1-12。`yolov8s` 已关闭的同类 IoU 门只覆盖 `yolov8s`。 |
| H3 | 未满足 | P1-9 |
| H4 | 未满足 | P0-2（挂起，未合入）。不设 `MORTRED_PACK` 时，39 个 id 含 `yolov8s` 与 `metric3d_512`。 |
| C1 | 未满足 | P1-10 |
| C4 | 已满足 | `01-modules/ci.md`：`gen_openapi.py --check` 与 `check_contract_sync.py` 仍在。`gpu=skipped` 时 `inference-paths` 退出码 2 并打印 `NOT_A_TRT_PROOF`，不打印 `TRT_PROOF=yes`。 |
| S1 | 已满足 | P0-3 合入 `9b142453e686254a7ff9bab250a39cc826d7fc80`。`yolov7` 与 `yolov7x` 在输出非空且头不匹配时返回契约失败；空向量仍是 `model output empty`。`YoloV7OutputContract` 重跑退出码 0。已关闭的 SAM 原句不在此条。 |
| S2 | 未满足 | D-12 |
| S3 | 未满足 | D-8 |
| S4 | 未满足 | D-9。模型进程在 `allow()` 之前返回 401。未满足的是网关在鉴权前扣配额。 |
| S6 | 未满足 | D-10。`nvidia-smi` 整次失败且输出为空时已标 offline，不在此条。 |

D-11 不是双目标行，留在劣势队列。
