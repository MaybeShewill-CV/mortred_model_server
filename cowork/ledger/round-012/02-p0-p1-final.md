# 交付队列

- 轮次: 12
- 版本: 沿用第 11 轮销账后的剩余条目
- 综合分上限: P1 上限 8.9
- 起点综合分: 7.8
- 表决: 口径未变，不重新打分，不重新表决。已关闭原句不重开。挂起项跳过。从 P1-12 第 3 步继续。

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-2 | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` | 换到另一台机器，不设置 `MORTRED_PACK`，默认 pack 为 `conf/packs/demo.toml`，范围内某一个 id 的产品 engine 不在本机 | 与 `cowork/ledger/round-011/02-p0-p1-final.md` 的 P0-2 行相同 | 不设 `MORTRED_PACK`，39 个 id 里每一个的产品 engine 缺失时，`mortredctl doctor --strict` 退出非 0，失败输出点名该 engine 路径，并有一行 `next:`，执行该行能在本机生成该 id 的产品 engine | `conf` 劣势 H4；`ops` 劣势 H4 |

P0-2 仍挂起。本轮跳过。

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-8 | H2：点名的 id 没有 CPU 数值用例 | 交付前确认该 id 的 CPU 数值 | 与第 11 轮冻结句相同。点名的 id 在 `model_golden_test --gtest_list_tests` 里没有 CPU 数值门 | 每个 id 有一条 CPU 用例。检测失败于同类且 IoU < 0.5，分割和抠图失败于掩码重合低于测试写明的阈值，embedding 和深度失败于测试写明的距离或误差 | `test` 劣势 H2 |
| P1-12 | H2：除 `yolov8s` 以外、产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较 | 交付前确认该 id 的本机 TensorRT 数值 | 除 `yolov8s` 的 `Yolov8TrtDetection` 外，其余产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` | 每个 id 有一条加载该产品 toml、且不把 backend 改成 cpu 的用例，失败条件与该 id 的 CPU 规则相同。`yolov8s` 的已关闭 IoU 门不写入这条的完成 | `test` 劣势 H2 |

P1-8 仍挂起。本轮跳过。下一条是 P1-12。

## 劣势
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| D-8 | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因 | 用默认镜像交付 | `docker build -t mortred_model_server:gpu .` 之后，`docker run --rm --entrypoint id mortred_model_server:gpu -u` 打印 `0`。`docs/unsupported-boundaries.md` 没有写必须 root 的原因 | 同一条 `id -u` 不再打印 `0`，或该边界文档写明必须 root 的原因 | `ops` 劣势 S3 |
| D-9 | S4：推理请求在鉴权失败时仍消耗限流配额 | 鉴权失败后紧接着用合法 token 调推理 | `enabled = false` 时把 `rate_per_sec` 和 `burst` 改成 1，同一来源先对推理路由发无 Bearer 请求得到 401，同一秒内再带正确 token 得到 429。无 inference 权限的 key，第一次推理是 401，第二次是 429 | 上述 401 不再占配额，紧接着的合法推理进入转发；`enabled = false` 时这道门不计数也不回 429；无 inference 权限的 key 连续两次都是 401 | `src-control` 劣势 S4 |
| D-10 | S6：`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率 | utilization 为 N/A 的一次成功探测 | `GET /api/v1/gpu` 为 `available: true` 且样本 `util` 为 -1，stderr 是 online。`mortredctl ps` 把 -1 印成利用率百分比 | 同一次探测的 `available` 为 false，或日志/状态写出 offline，CLI 不再把 -1 印成利用率百分比 | `src-control` 劣势 S6 |
| D-11 | `deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only | 按部署文档找 `resnet` 或确认 `msocrnet` 是否在 HTTP catalog | `docs/deployment.md` 108 行和 253 行写 `resnet50`。108 行把 `MsOcrNet` 写成 Bench-only、不是 SME HTTP 路径。catalog id 是 `resnet` 与 `msocrnet` | 这两处 cpu 名称是 `resnet`，Bench-only 列表不再包含 `msocrnet` | `docs` 劣势 |
| D-12 | S2：一次拉起后，检查失败并点名 fork 与 exec 之间的调用 | 一次拉起网关或模型 | 一次拉起网关或模型后，有一条检查失败，失败输出点名 fork 与 exec 之间的 `setenv`、`opendir`、`signal` 或 `std::to_string` | 同一条检查通过，且模型仍监听 `127.0.0.1`，并带上 `MORTRED_AUTH_TOKEN` 或网关的 `MORTRED_METRICS_TOKEN` | `src-control` 劣势 S2 |

## 附录
已关闭原句不重开：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、D-1 至 D-6。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`yolov8s` 的 TRT 同类 IoU 门不写入 P1-12 的完成。`centerface` 的 engine 输入名 `input.1` 不重开。

第 11 轮 P1-12 的方案 v1 要求已有 CPU 黄金的 19 条退出码 0。实测 9 条产品 engine 在原阈值下退出码 1，偏离回修一次后仍是 1，该轮因此停止。这 9 条的测量写在 `cowork/ledger/round-011/issues/p1-12/04-rd-repair.md`。本轮方案必须能在这些测量上一次黑盒判定，不得把阈值放宽写成通过。
