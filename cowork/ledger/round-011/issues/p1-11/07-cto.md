# 销账

- 轮次: 11
- 已关闭: P0-3、P1-7、P1-9、P1-10、P1-11
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）。P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 9.3 | 无 | 无 |
| src-server-apps | 15 | 8.2 | S1：`yolov7` 与 `yolov7x` 在输出张量非空、但既没有 `output`/`518`/`532`，也凑不齐 3 个 `[1,3,H,W,C]` 头时，仍报成空输出。；H1：范围内每个 HTTP catalog id 都没有一条与 `docs/openapi.json` 一致的请求和响应示例。 | 无 |
| src-control | 10 | 7.4 | 无 | supervisor 在 `fork` 之后、`exec` 之前调用 `setenv` 和其他非 async-signal-safe 函数。；推理请求在鉴权失败时仍消耗限流配额。；`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率，日志和状态不标明 offline。 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 8.1 | 无 | H4 实际消费的是 pack，不是这份产品 engine 清单。 |
| ops | 15 | 7.0 | 无 | H4：默认 `mortredctl doctor --strict`（未设 `MORTRED_PACK`，pack 为 `conf/packs/demo.toml`，只有 CPU `mobilenetv2`）在 `yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface` 的产品 engine 缺失时，TensorRT 段仍打印 `[ok] pack TRT engines present or pack has no TensorRT backends`，不打印 `next: mortredctl prepare`。；S3：默认发布镜像是 Dockerfile 最后阶段 `mortred-gpu`（`FROM runtime`）。全程没有 `USER`。`docker-compose.yml` 的 `mortred` 与 `mortred-cpu` 没有 `user:`。`docker build -t mortred_model_server:gpu .` 之后，`docker run --rm --entrypoint id mortred_model_server:gpu -u` 打印 `0`。`docs/unsupported-boundaries.md` 没有写必须 root 的原因。 |
| test | 15 | 6.0 | H2：`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 只有 fingerprint | 这些 id 没有满足 H2 的 CPU 数值比较：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` 在 `test/model_golden_test.cc` 里没有用例。；产品 toml 为 `type=tensorrt`、但没有同规则 TRT 比较的 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。 |
| docs | 10 | 7.2 | H1：以下 32 个 id 没有一条与 `docs/openapi.json` 一致的请求和响应示例。；H3：范围内 39 个 id 都没有一行同时写出硬件、复现命令、指标和日期。；C1：中文和英文 README 的 GPU 源码构建命令不一致。 | `docs/deployment.md` 108 行和 253 行把 cpu 目录写成 `resnet50`，catalog id 与 `conf/server/classification/resnet/resnet_server_cpu_config.toml` 的 `model` 是 `resnet`。同文件 108 行把 `MsOcrNet` 写成 Bench-only、不是 SME HTTP 路径，`scene_segmentation` 的 `catalog()` 与 `docs/openapi.json` 仍注册 `msocrnet`。 |
| ci | 5 | 9.0 | 无 | 无 |

## 附录
P1-11 合入 `8552c0e7fdc224d16926911166d60eee6a797cab`。问题分支提交 `2dceffc1b3e45a3c99d4f0e379ab1a6146433c19`。回测结论 pass。偏离无。没有新的回修提交。合入后 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 这四条比较全分辨率掩码重合，阈值 0.90。

同义句：`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 只有 fingerprint，没有掩码重合阈值。关闭范围只含这四个 id。H2 整行仍未满足，因为 P1-8 仍挂起、P1-12 仍未做。

P0-2、P1-8 仍挂起。P1-12、D-8、D-9、D-10、D-11、D-12 的原句保留。分数照上一份销账，不改。test 仍为 6.0。

## 新准入
无
