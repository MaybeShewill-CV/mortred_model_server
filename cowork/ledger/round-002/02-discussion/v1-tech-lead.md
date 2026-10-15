# 第2步评审

- 轮次: 2
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 评审 | 理由 |
| --- | --- | --- |
| P2-12 | 保留 | 按英文教程启动会得到 `unknown --model`。只打开 `docs/tutorials_of_model_servers.md`，核对五条 `--model` 为 `mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`，一次就能判定。第一小时仍走 `mortredctl next`，业务调用发得出，维持 P2。`yolov5l`、`bisenetv2`、`attentive_gan` 只改这一份教程里的字符串，不进入覆盖集，也不另开完成标准。与 P2-13 分文件关闭。 |
| P2-13 | 保留 | 中文教程是另一份文件上的同一类错误。只打开 `docs/tutorials_of_model_servers.zh-cn.md` 核对同样五条 `--model`，一次就能判定。与 P2-12 并成一条时，一种语言没改掉，整条关不掉。维持 P2。 |
| P2-14 | 保留 | 英文教程要求加大仓库自带配置的 `worker_nums`，与 `docs/deployment.md` 里 `conf/server` 保持 1、校准写在机器 pack 相反。只打开教程看这一句，一次就能判定。按部署文档仍能校准和换机，维持 P2。与 P2-12 分开：`--model` 改对后，这一句仍可能留着。 |
| P2-15 | 保留 | 中文教程同一句。只打开 `docs/tutorials_of_model_servers.zh-cn.md` 这一句即可判定。与 P2-14 分文件关闭，与 P2-13 也分开。维持 P2。 |
| P2-16 | 保留 | 中文添加入口的 `model_section`、`--model`、`model=` 和 family 与 catalog 的 `densenet` 不一致。英文指南已在 P1-2 关闭，本条不在该关闭范围内。只打开 `docs/how_to_add_new_server.zh-cn.md` 核对这四处，一次就能判定。英文路径已经能给已有家族加模型，维持 P2。`densenet` 不因此进入覆盖集。 |
| P2-17 | 保留 | 英文契约公共响应示例的 `model.name` 是 `MOBILENETV2`，`docs/openapi.json` 里同一 id 的示例是 `mobilenetv2`。只打开 `docs/api-contract.md` 看这一处，一次就能判定。OpenAPI 示例已经在，调用不受影响，维持 P2。 |
| P2-18 | 保留 | 中文契约同一处示例。只打开 `docs/api-contract.zh-cn.md` 即可判定。与 P2-17 分文件关闭。维持 P2。 |
| P2-19 | 保留 | 连续成功超过 4096 次后，五段 `mortred_gateway_stage_duration_ms` 还在，`mortred_gateway_stage_dominant` 不再出现。连续 4097 次成功推理后，带 metrics token 读一次 gateway `/metrics`，正文仍有五段直方图，并且有 `stage` 为五段里 p50 最大名字的 dominant 行，一次就能判定。直方图和 `server_time_ms` 仍在，入口和恢复不受影响，维持 P2。已关闭的 P1-1 是文件开关，不在本条。 |
| P2-20 | 保留 | `gpu-pr-gate` 成功时 `inference-paths` 打印 `TRT_PROOF=yes` 并以 0 退出，同一次四个 golden 的 `backend.type` 都是 `onnx`。看这一次日志和同一次 gtest XML 就能判定：只跑这四个 ONNX 用例，且 yolov8 引擎步骤被 skip 或只写出引擎文件时，日志没有 `TRT_PROOF=yes`；日志出现 `TRT_PROOF=yes` 时，XML 里至少有一个产品 `type=tensorrt` 用例且 `skipped=0`。GPU 作业被跳过时已经是 `NOT_A_TRT_PROOF`、退出码 2。过滤器不扩成全目录。调用不受影响，维持 P2。 |

## 不同意的条目
无
