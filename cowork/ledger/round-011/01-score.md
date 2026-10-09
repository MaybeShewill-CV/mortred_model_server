# 项目评分

- 轮次: 11
- 自算原始分: 7.8
- 自算加权分: 7.8
- 上限规则: P0 上限 7.9
- 相对上轮综合分: 低于 8.5

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 9.3 | `cowork/ledger/round-011/01-modules/src-models.md` |
| src-server-apps | 15 | 8.2 | `cowork/ledger/round-011/01-modules/src-server-apps.md` |
| src-control | 10 | 7.4 | `cowork/ledger/round-011/01-modules/src-control.md` |
| src-factory-common | 5 | 9.3 | `cowork/ledger/round-011/01-modules/src-factory-common.md` |
| conf | 10 | 8.1 | `cowork/ledger/round-011/01-modules/conf.md` |
| ops | 15 | 7.0 | `cowork/ledger/round-011/01-modules/ops.md` |
| test | 15 | 6.0 | `cowork/ledger/round-011/01-modules/test.md` |
| docs | 10 | 7.2 | `cowork/ledger/round-011/01-modules/docs.md` |
| ci | 5 | 9.0 | `cowork/ledger/round-011/01-modules/ci.md` |

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-2 | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` | 换到另一台机器，不设置 `MORTRED_PACK`，默认 pack 为 `conf/packs/demo.toml`，范围内某一个 id 的产品 engine 不在本机 | `yolov8s` 在 `conf/packs/yolov8.toml` 与 `conf/packs/gpu_yolo_metric3d.toml`，`metric3d_512` 在 `conf/packs/metric3d.toml` 与 `conf/packs/gpu_yolo_metric3d.toml`。选中对应 pack 且该 engine 缺失时，doctor 退出非 0。不设 `MORTRED_PACK`、默认 pack 为 `conf/packs/demo.toml`（只有 CPU `mobilenetv2`）时，`yolov8s` 的产品 engine `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16` 与 `metric3d_512` 的产品 engine `weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16` 缺失，`mortredctl doctor --strict` 仍退出 0，没有 `next:`。同一默认 pack 下，其余 id 的产品 engine 缺失也同样退出 0，TensorRT 段打印 `[ok] pack TRT engines present or pack has no TensorRT backends`，没有 `next:`。这 39 个 id 是：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` | 不设 `MORTRED_PACK`，上述 39 个 id 里每一个的产品 engine 缺失时，`mortredctl doctor --strict` 退出非 0，失败输出点名该 engine 路径，并有一行 `next:`，执行该行能在本机生成该 id 的产品 engine。点名含全部 39 个 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`，含 `yolov8s` 与 `metric3d_512` | `conf` 劣势 H4；`ops` 劣势 H4 | 新证据 |
| P0-3 | `yolov7` 与 `yolov7x` 在输出张量非空、头不匹配时仍报成空输出 | 这两个 id 的输出既没有 `output`/`518`/`532`，也凑不齐 3 个 `[1,3,H,W,C]` 头 | 推理 POST 的响应是 HTTP 500、`status` 5、`status_str` 为 `model output empty`，`results[].data` 为空 | 同一请求变为 HTTP 500、`status` 6、`status_str` 为 `model output contract failed` | `src-server-apps` 劣势 S1 | 新证据 |

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-7 | H1：范围内 id 没有与 `docs/openapi.json` 一致的请求和响应示例 | 按 catalog id 照抄请求体和 200 响应 | `GET /openapi.json` 与 `src/server/openapi_doc.h` 的范围内 path 只有 schema，没有 `example`。`docs/api-contract.md` 只写了 `mobilenetv2`、`yolov8s`、`pphuman_server`、`dbnet`、`modnet`、`realesrgan`、`metric3d_512`。缺示例的 32 个 id：`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`hrnet`、`msocrnet`、`attentive_gan`、`enlightengan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_1088` | 范围内每一个 id 的 path 都有一条请求示例和一条 200 响应示例，且能按该 path 的 schema 校验通过。`yolov8n`、`yolov8l`、`yolov8x` 各自都有，不用 `yolov8s` 代替。`libface` 与 `centerface` 含 `landmarks`。`superpoint` 含 `score`、`location`、`descriptor` | `src-server-apps` 劣势 H1；`docs` 劣势 H1 | 新证据 |
| P1-8 | H2：多数 id 没有与 CPU golden 同级的数值比较，产品 `type=tensorrt` 的 id 也没有同规则 TRT 比较 | 交付前确认该 id 的 CPU 与本机 TensorRT 数值 | `model_golden_test --gtest_list_tests` 列不出这些 id 的数值门：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 只有 fingerprint，没有掩码重合阈值。除 `yolov8s` 的 `Yolov8TrtDetection` 外，其余产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` | 每个点名的无用例 id 有一条 CPU 用例：检测失败于同类且 IoU < 0.5，分割和抠图失败于掩码重合低于测试写明的阈值，embedding 和深度失败于测试写明的距离或误差阈值。每个点名的 TRT id 多一条加载该产品 toml、且不把 backend 改成 cpu 的用例，失败条件与该任务的 CPU 规则相同 | `test` 劣势 H2 | 新证据 |
| P1-9 | H3：范围内 39 个 id 都没有一行同时写出硬件、复现命令、指标和日期 | 换机或交基准时要复现该 id 的数字 | `docs/benchmark-set-a.md` 的硬件、日期、指标和命令不在同一行。缺这一行的 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` | 每个 id 有一行同时含硬件、可直接执行的复现命令、指标和日期 | `docs` 劣势 H3 | 新证据 |
| P1-10 | C1：中文和英文 README 的 GPU 源码构建命令不一致 | 按 README 在另一台机器做 GPU 源码构建 | `README.zh-cn.md` 有 `sudo ./scripts/install_deps.sh --nvidia`。`README.md` 的 GPU 依赖行是 `./scripts/install_deps.sh --all` 与 `./scripts/install_deps.sh --check`。两边的 `cmake --preset full` 与 `cmake --preset full-cpu` 已与 `CMakePresets.json` 一致 | 两份 README 源码构建代码块的 GPU 可执行行集合相同，并且仍包含这两条 `cmake --preset` 命令 | `docs` 劣势 C1 | 新证据 |

## 劣势
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| D-7 | S2：supervisor 在 `fork` 之后、`exec` 之前调用 `setenv` 和其他非 async-signal-safe 函数 | 多线程监督进程拉起网关或模型 | `spawn_locked` 在 `fork` 之后、`execl` 之前调用 `setenv`，同段还有 `opendir`、`signal` 和 `std::to_string` | 同一段子进程在 exec 前不再出现这些调用，环境改由 `exec` 的环境块带入；模型仍监听 `127.0.0.1`，并带上 `MORTRED_AUTH_TOKEN` 或网关的 `MORTRED_METRICS_TOKEN` | `src-control` 劣势 S2 | 新证据 |
| D-8 | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因 | 用默认镜像交付 | `docker build -t mortred_model_server:gpu .` 之后，`docker run --rm --entrypoint id mortred_model_server:gpu -u` 打印 `0`。`docs/unsupported-boundaries.md` 没有写必须 root 的原因 | 同一条 `id -u` 不再打印 `0`，或该边界文档写明必须 root 的原因 | `ops` 劣势 S3 | 新证据 |
| D-9 | S4：推理请求在鉴权失败时仍消耗限流配额 | 鉴权失败后紧接着用合法 token 调推理 | `enabled = false` 时把 `rate_per_sec` 和 `burst` 改成 1，同一来源先对推理路由发无 Bearer 请求得到 401，同一秒内再带正确 token 得到 429。无 inference 权限的 key，第一次推理是 401，第二次是 429 | 上述 401 不再占配额，紧接着的合法推理进入转发；`enabled = false` 时这道门不计数也不回 429；无 inference 权限的 key 连续两次都是 401 | `src-control` 劣势 S4 | 新证据 |
| D-10 | S6：`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率 | utilization 为 N/A 的一次成功探测 | `GET /api/v1/gpu` 为 `available: true` 且样本 `util` 为 -1，stderr 是 online。`mortredctl ps` 把 -1 印成利用率百分比 | 同一次探测的 `available` 为 false，或日志/状态写出 offline，CLI 不再把 -1 印成利用率百分比 | `src-control` 劣势 S6 | 新证据 |
| D-11 | `deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only | 按部署文档找 `resnet` 或确认 `msocrnet` 是否在 HTTP catalog | `docs/deployment.md` 108 行和 253 行写 `resnet50`。108 行把 `MsOcrNet` 写成 Bench-only、不是 SME HTTP 路径。catalog id 是 `resnet` 与 `msocrnet` | 这两处 cpu 名称是 `resnet`，Bench-only 列表不再包含 `msocrnet` | `docs` 劣势 | 新证据 |

## 附录
旧问题改写：上新架构仍要写 C++。范围内 id 已有类。任意新架构零 C++ 不另开。

旧问题改写：多 session / diffusion 更难配。CLIP、LightGlue、SAM、diffusion 不在范围表。

旧问题改写：SAM 非空契约故障报成空输出。SAM 不在范围表，已关闭范围不重开。P0-3 只覆盖 `yolov7` 与 `yolov7x`。

旧问题改写：`yolov8s` 的 TRT 同类 IoU 门已关闭，关闭范围只含 `yolov8s`。其余 id 的数值门在 P1-8。

旧问题改写：缺权重时选中用例全部跳过，`model_golden_test` 退出 77 并打印 `NUMERICAL_NOT_RUN`。

旧问题改写：默认 PR 下 GPU 作业被跳过时，`inference-paths` 退出码 2 并打印 `NOT_A_TRT_PROOF`。C4 不回队列。

旧问题改写：中文 README 路径 B 已使用 `install_deps.sh` 与 `cmake --preset full` / `full-cpu`。P1-10 只覆盖两边 GPU 依赖行不一致。

不计毛病：HTTP OCR 只有 `dbnet` 检测、没有识别。SAM、diffusion、CLIP、LightGlue。一个 catalog id 一个 OS 进程。超时后外层 SIGKILL。训练、标注、多云。engine 在本机本卡生成。不推翻现有 HTTP 契约或产品后端。

## 未读模块
无
