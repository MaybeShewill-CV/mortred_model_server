# 销账

- 轮次: 11
- 已关闭: P0-3、P1-7
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 9.3 | 无 | 无 |
| src-server-apps | 15 | 8.2 | H1：范围内每个 HTTP catalog id 都没有一条与 `docs/openapi.json` 一致的请求和响应示例。 | 无 |
| src-control | 10 | 7.4 | 无 | supervisor 在 `fork` 之后、`exec` 之前调用 `setenv` 和其他非 async-signal-safe 函数。；推理请求在鉴权失败时仍消耗限流配额。；`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率，日志和状态不标明 offline。 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 8.1 | 无 | H4 实际消费的是 pack，不是这份产品 engine 清单。 |
| ops | 15 | 7.0 | 无 | H4：默认 `mortredctl doctor --strict`（未设 `MORTRED_PACK`，pack 为 `conf/packs/demo.toml`，只有 CPU `mobilenetv2`）在 `yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface` 的产品 engine 缺失时，TensorRT 段仍打印 `[ok] pack TRT engines present or pack has no TensorRT backends`，不打印 `next: mortredctl prepare`。；S3：默认发布镜像是 Dockerfile 最后阶段 `mortred-gpu`（`FROM runtime`）。全程没有 `USER`。`docker-compose.yml` 的 `mortred` 与 `mortred-cpu` 没有 `user:`。`docker build -t mortred_model_server:gpu .` 之后，`docker run --rm --entrypoint id mortred_model_server:gpu -u` 打印 `0`。`docs/unsupported-boundaries.md` 没有写必须 root 的原因。 |
| test | 15 | 6.0 | 无 | 这些 id 没有满足 H2 的 CPU 数值比较：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` 在 `test/model_golden_test.cc` 里没有用例。`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 虽有 CPU 用例，但 `compare_scene_segmentation` / `compare_matting` 只走 `expect_fingerprint`（64×64 均值差 ≤ 1.0），没有掩码重合阈值。；产品 toml 为 `type=tensorrt`、但没有同规则 TRT 比较的 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。 |
| docs | 10 | 7.2 | H1：以下 32 个 id 没有一条与 `docs/openapi.json` 一致的请求和响应示例。 | H3：范围内 39 个 id 都没有一行同时写出硬件、复现命令、指标和日期。；C1：中文和英文 README 的 GPU 源码构建命令不一致。；`docs/deployment.md` 108 行和 253 行把 cpu 目录写成 `resnet50`，catalog id 与 `conf/server/classification/resnet/resnet_server_cpu_config.toml` 的 `model` 是 `resnet`。同文件 108 行把 `MsOcrNet` 写成 Bench-only、不是 SME HTTP 路径，`scene_segmentation` 的 `catalog()` 与 `docs/openapi.json` 仍注册 `msocrnet`。 |
| ci | 5 | 9.0 | 无 | 无 |

## 附录
P1-7 合入 `8f6143544431ba6f7b9d64584cf7d1f6ff92fb28`。问题分支提交 `2cd30f633230a1037c0a63520f47a4302026f5f1`。QA 结论 pass。偏离无。合入后 39 个 catalog path 各自有请求 example 和 200 example，并能按该 path schema 校验。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 id。`libface` 与 `centerface` 含 `landmarks`。`superpoint` 含 `score`、`location`、`descriptor`。

同义句：范围内 id 没有与 `docs/openapi.json` 一致的请求和响应示例。`GET /openapi.json` 上范围内 path 的 `requestBody` 与 200 响应只有 schema，没有可照抄的请求体和响应体。`docs/openapi.json` 与 `src/server/openapi_doc.h` 全文都没有 `example`。`docs/api-contract.md` 的 7 个人读片段不是 OpenAPI 示例。这几句只留本附录。

P0-3 已在本轮早前关闭，不重开。同义句：`yolov7` 与 `yolov7x` 在输出张量非空、头不匹配时仍报成空输出。`yolov7`/`yolov7x` 的非空头不匹配会被看成空输出。已关闭的 SAM 原句不在此条，不重开。

QA 残余风险不进入队列。P0-3：验收 7 用有限全零头证明框数为 0 时顶层 `status` 为 0，且 `data` 为 JSON 数组；三个名字对齐之后的 dtype、通道、buffer 和非有限值失败，方案写明保持契约失败。P1-7：示例是短占位，方案已写明不要求真实推理。`EnvelopeData` 的字段口径由 `--check` 另判。表外 path 没有示例，方案写明不因此让自测失败。Windows 检出把 `test/golden/mobilenetv2_classification.json` 写成 CRLF，git 对象仍是 LF，且与基线 blob 相同。这几句没有新的一次黑盒可判定的变化。

P0-2 仍挂起。conf 与 ops 的 H4 原句保留，不改写成新劣势。P1-8、P1-9、P1-10、P1-11、P1-12、D-8、D-9、D-10、D-11、D-12 的原句保留。未改模块的分数照 `cowork/ledger/round-011/01-modules/`，不改。src-server-apps 仍为 8.2，docs 仍为 7.2。

## 新准入
无
