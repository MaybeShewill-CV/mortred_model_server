# 技术方案

- 问题: P1-7
- 版本: v1
- 对应列表: v3

## 边界

上一版审查意见无。本方案只覆盖冻结句 H1：按 catalog id 照抄请求体和 200 响应时，范围内每一个 id 的 OpenAPI path 都有一条请求示例和一条 200 响应示例，并且这两条都能按该 path 现有 schema 校验通过。

范围是 rubric 2026-10-09 的 39 个 HTTP catalog id，含已在 `docs/api-contract.md` 有人读片段的 7 个。这 7 个仍要有自己的 path 示例。path 以现有 `server_uri` 为准：

| catalog id | path |
| --- | --- |
| `mobilenetv2` | `/mortred_ai_server_v1/classification/mobilenetv2` |
| `resnet` | `/mortred_ai_server_v1/classification/resnet` |
| `densenet` | `/mortred_ai_server_v1/classification/densenet` |
| `yolov5l` | `/mortred_ai_server_v1/obj_detection/yolov5l` |
| `yolov6s` | `/mortred_ai_server_v1/obj_detection/yolov6s` |
| `nanodet_1x5` | `/mortred_ai_server_v1/obj_detection/nanodet_1x5` |
| `nanodet_416` | `/mortred_ai_server_v1/obj_detection/nanodet_416` |
| `yolov7` | `/mortred_ai_server_v1/obj_detection/yolov7` |
| `yolov7x` | `/mortred_ai_server_v1/obj_detection/yolov7x` |
| `yolov8n` | `/mortred_ai_server_v1/obj_detection/yolov8n` |
| `yolov8s` | `/mortred_ai_server_v1/obj_detection/yolov8s` |
| `yolov8l` | `/mortred_ai_server_v1/obj_detection/yolov8l` |
| `yolov8x` | `/mortred_ai_server_v1/obj_detection/yolov8x` |
| `libface` | `/mortred_ai_server_v1/obj_detection/libface` |
| `centerface` | `/mortred_ai_server_v1/obj_detection/centerface` |
| `bisenetv2` | `/mortred_ai_server_v1/scene_segmentation/bisenetv2` |
| `pphuman_mobile` | `/mortred_ai_server_v1/scene_segmentation/pphuman_mobile` |
| `pphuman_lite` | `/mortred_ai_server_v1/scene_segmentation/pphuman_lite` |
| `pphuman_server` | `/mortred_ai_server_v1/scene_segmentation/pphuman_server` |
| `hrnet` | `/mortred_ai_server_v1/scene_segmentation/hrnet` |
| `msocrnet` | `/mortred_ai_server_v1/scene_segmentation/msocrnet` |
| `dbnet` | `/mortred_ai_server_v1/ocr/dbnet` |
| `attentive_gan` | `/mortred_ai_server_v1/enhancement/attentive_gan` |
| `enlightengan` | `/mortred_ai_server_v1/enhancement/enlightengan` |
| `realesrgan` | `/mortred_ai_server_v1/enhancement/realesrgan` |
| `superpoint` | `/mortred_ai_server_v1/feature_point/superpoint` |
| `dinov2_vits14` | `/mortred_ai_server_v1/feature_embedding/dinov2_vits14` |
| `dinov2_vitb14` | `/mortred_ai_server_v1/feature_embedding/dinov2_vitb14` |
| `dinov2_vitl14` | `/mortred_ai_server_v1/feature_embedding/dinov2_vitl14` |
| `modnet` | `/mortred_ai_server_v1/matting/modnet` |
| `ppmatting_512` | `/mortred_ai_server_v1/matting/ppmatting_512` |
| `ppmatting_1024` | `/mortred_ai_server_v1/matting/ppmatting_1024` |
| `ppmatting_resnet34` | `/mortred_ai_server_v1/matting/ppmatting_resnet34` |
| `ppmatting_v2` | `/mortred_ai_server_v1/matting/ppmatting_v2` |
| `depth_vits14` | `/mortred_ai_server_v1/mono_depth_estimation/depth_vits14` |
| `depth_vitb14` | `/mortred_ai_server_v1/mono_depth_estimation/depth_vitb14` |
| `depth_vitl14` | `/mortred_ai_server_v1/mono_depth_estimation/depth_vitl14` |
| `metric3d_512` | `/mortred_ai_server_v1/mono_depth_estimation/metric3d_512` |
| `metric3d_1088` | `/mortred_ai_server_v1/mono_depth_estimation/metric3d_1088` |

`libface` 与 `centerface` 的 path 在 `obj_detection` 下。示例只加在 `application/json`。`GET /openapi.json` 下发的是 `src/server/openapi_doc.h` 里的内嵌文档，与 `docs/openapi.json` 字节一致才算服务到同一份示例。

不覆盖 P0-2、P0-3、P1-8 至 P1-12、D-8 至 D-12。不重开已关闭原句。diffusion、SAM 以及其他不在上表的 path 不要求示例。

## 做法

### 推荐

在现有生成链上为上表 39 个 path 各写一条 JSON 请求 `example` 和一条 200 响应 `example`。`scripts/gen_openapi.py` 按 `server_uri` 把示例放在该 path 的 `requestBody.content.application/json.example` 与 `responses.200.content.application/json.example`，再生成并提交 `docs/openapi.json` 与 `src/server/openapi_doc.h`。`python scripts/gen_openapi.py --check` 同时检查生成物是否过期，以及这 39 条示例是否存在、是否按该 path 现有 schema 校验通过、是否满足下文契约里的字段口径。校验只用 Python 标准库，覆盖本文档已经使用的关键字：`$ref`、`allOf`、`type`、`required`、`properties`、`additionalProperties`、`items`、`minItems`、`maxItems`、`minimum`、`maximum`、`enum`、`nullable`。`format`、`description`、`readOnly`、`default` 不参与成败。

代价：两份生成物变大，增加的是短占位 JSON，不是权重或图像。共享同一个 `Request_*` schema 的 id 必须在各自 path 上各写一条，生成和校验都要按 path 而不是按 section 去重。`msocrnet` 与三个 DINOv2 的 data schema 仍是没有字段的 `EnvelopeData`，校验器对它们几乎不约束字段，所以字段口径要在 `--check` 里另判。

### 否决

只补 `docs/api-contract.md` 的人读片段。代价是 `GET /openapi.json` 与 `src/server/openapi_doc.h` 的 path 仍然只有 schema。已有的 7 个片段不能算这 7 个 path 的示例，冻结句的可观察变化不出现。

把示例写在共享的 `components.schemas` 上，或只给一个任务族一条示例，用 `yolov8s` 代表 `yolov8n`、`yolov8l`、`yolov8x`。代价是 `yolov8n`、`yolov8s`、`yolov8l`、`yolov8x` 共用 `Request_YOLOV8_DETECTION_SERVER`，组件级示例会同时出现在四个 path 上，n、l、x 没有自己的 path 示例。同一情况还有 `nanodet_1x5` 与 `nanodet_416`，`yolov7` 与 `yolov7x`，三个人像分割，四个 PP-Matting，三个 Depth Anything，两个 Metric3D，三个 DINOv2。手改两份生成物同样否决：下一次 `--check` 会因与生成结果不一致而失败。

## 改动面

- `scripts/gen_openapi.py`：按 39 个 `server_uri` 写入 path 级 JSON `example`，并在 `--check` 中做存在性、schema 校验和字段口径。
- `docs/openapi.json`：只由该脚本写出。
- `src/server/openapi_doc.h`：只由该脚本写出，内嵌与 JSON 文件相同的文档。

`.github/workflows/ci.yml` 已有的 `python3 scripts/gen_openapi.py --check` 与 `python3 scripts/check_consistency.py` 继续作为门，不新增作业。

## 明确不改

- 推理实现、`src/server/response_serializers.h`、状态码、鉴权、限流、gateway 与 supervisor 的进程模型。
- 产品 toml、`conf/`、engine、`mortredctl`、权重。
- `docs/contract_dump.json`、`src/factory/`、`components.schemas` 及其 data `$ref`。`msocrnet` 保持 `EnvelopeData` 且 summary 仍为未知任务；三个 DINOv2 保持 `EnvelopeData`。
- `info.x-contract-hash` 的计算方式。dump 不动时，该值保持 `b320c08dde6d040146cc649381361fae41acb57ea3ba9e531c842e6b0aece400`。
- `docs/api-contract.md` 与 `docs/api-contract.zh-cn.md`。已有 7 个人读片段留在原处，不把它们当成 path 示例，也不把本项做成补全 39 段人读文档。
- `image/png`、`image/jpeg`、`application/octet-stream` 不加 `example`。
- 健康检查、`/metrics`、`/openapi.json` 这三个 path，以及 diffusion、SAM 等表外 path。
- 新的第三方包、新的 CI 作业、`test/openapi_consistency_test.cc`。

## 契约

示例是 OpenAPI 3.0 media type 上的单个 `example` 对象，值是内联 JSON，不是 `$ref`，也不是只写 `examples`。

每个请求示例只有两个键：`req_id` 等于该 path 最后一段 catalog id；`images` 是长度为 1 的字符串数组，字符串非空且不超过 64 个字符。没有 `params`、`options`、`img_data` 或其他键。这个对象符合现有请求 schema：`images` 必填，`req_id` 可选，`additionalProperties` 为 false。

每个 200 示例是成功信封：`status` 为整数 0，`status_str` 为 `OK`，`task_id` 等于 catalog id，`partial` 为 false，`server_time_ms` 为数字，`model.name` 等于 catalog id，`model.version` 为字符串，`results` 长度恰好为 1，`results[0].status` 为 0，`results[0].data` 不是 null。`model.name` 用 catalog id，与现有响应里 `model.name` 取 catalog id 的写法一致，用来区分共用 schema 的 path。

`results[0].data` 按该 path 现有 data `$ref` 与现有序列化字段：

| id | 现有 data `$ref` | 示例 data |
| --- | --- | --- |
| `mobilenetv2`、`resnet`、`densenet` | `ClassificationResult` | 对象，含整数 `class_id`、字符串 `category`、长度 1 到 8 的数字数组 `scores` |
| `yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x` | `DetectionResult` | 长度 1 的数组；元素含整数 `class_id`、数字 `score`、字符串 `category`、4 个数字的 `bbox`、对象 `detail_infos` |
| `libface`、`centerface` | `FaceResult` | 长度 1 的数组；元素在检测字段之外含 `landmarks`，`landmarks` 长度 1 到 5，每个点恰好 2 个数字 |
| `bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet` | `SegmentationResult` | 对象，含非空且不超过 64 字符的字符串 `image` 与 `colorized_mask` |
| `msocrnet` | `EnvelopeData` | `$ref` 保持 `EnvelopeData`；data 仍是含上述两个字符串的对象，以便按分割结果照抄 |
| `dbnet` | `TextRegionResult` | 长度 1 的数组；元素含数字 `score`、4 个数字的 `bbox`、至少 1 个且至多 4 个点的 `polygon`、对象 `detail_infos`；没有 `text` |
| `modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2` | `MattingResult` | 对象只要求非空且不超过 64 字符的 `image`；没有 `colorized_mask` |
| `attentive_gan`、`enlightengan`、`realesrgan` | `EnhancementResult` | 对象含非空且不超过 64 字符的 `image` |
| `depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` | `DepthResult` | 对象含非空且不超过 64 字符的 `image` |
| `superpoint` | `FeaturePointResult` | 长度 1 的数组；元素含数字 `score`、恰好 2 个数字的 `location`、长度 1 到 8 的数字数组 `descriptor` |
| `dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14` | `EnvelopeData` | `$ref` 保持 `EnvelopeData`；data 含整数 `dim` 与数字数组 `embedding`，二者长度相等，且长度在 1 到 8 |

检测、分类、分割、OCR、抠图、增强、深度的内层字段可以沿用 `docs/api-contract.md` 里已有片段的取值。沿用内层取值时，信封的 `req_id`、`task_id`、`model.name` 仍必须是该 path 自己的 catalog id。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 id。

占位字符串不要求是可解码图像，也不要求来自一次真实推理。schema 只要求这些位置是字符串或数字。

## 性能与资源

不增加进程、不加载 engine、不占显存、不改变推理时延。变大的只有每个模型进程内嵌的 OpenAPI 字符串，以及 `GET /openapi.json` 的响应体。39 条示例使用短占位：图像字符串不超过 64 字符，检测、人脸、OCR、特征点各 1 条，分类 `scores` 与特征点 `descriptor`、DINOv2 `embedding` 的长度不超过 8。CPU profile 与 GPU profile 下发同一份内嵌文档。

## 验收标准

1. 上表 39 个 path 各自的 `post.requestBody.content.application/json` 有内联对象 `example`，`post.responses.200.content.application/json` 也有内联对象 `example`。只在 `components.schemas` 上出现的 example 不算。
2. 这 39 条请求示例都能按该 path 的请求 schema 校验通过，并且只有 `req_id` 与 `images` 两个键，`req_id` 等于 path 最后一段，`images` 长度为 1，元素为长度 1 到 64 的字符串。
3. 这 39 条 200 示例都能按该 path 的 200 `application/json` schema 校验通过，并且 `status` 为 0、`status_str` 为 `OK`、`task_id` 与 `model.name` 等于该 catalog id、`model.version` 为字符串、`server_time_ms` 为数字、`partial` 为 false、`results` 长度为 1、`results[0].status` 为 0、`results[0].data` 不是 null。
4. `/mortred_ai_server_v1/obj_detection/yolov8n`、`yolov8l`、`yolov8x` 各自满足第 1 到第 3 条，三条 200 示例的 `model.name` 分别是 `yolov8n`、`yolov8l`、`yolov8x`。
5. `libface` 与 `centerface` 的 `results[0].data` 是长度 1 的数组，元素含长度 1 到 5 的 `landmarks`，每个点恰好 2 个数字，并含 `class_id`、`score`、`category`、`bbox`、`detail_infos`。
6. `superpoint` 的 `results[0].data` 是长度 1 的数组，元素含数字 `score`、恰好 2 个数字的 `location`、长度 1 到 8 的数字数组 `descriptor`。
7. 三个分类 id 的 data 含整数 `class_id`、字符串 `category`、长度 1 到 8 的 `scores`。十个检测 id（不含两个人脸 id）的 data 是长度 1 的检测元素数组，元素含 `class_id`、`score`、`category`、4 个数的 `bbox`、对象 `detail_infos`。
8. `bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet` 的 data 含长度 1 到 64 的 `image` 与 `colorized_mask`。`msocrnet` 的 data `$ref` 仍是 `EnvelopeData`，其示例 data 同样含这两个字符串，并且 200 示例仍能通过该 path 的 schema 校验。
9. `dbnet` 的 data 是长度 1 的数组，元素含 `score`、4 个数的 `bbox`、1 到 4 个点的 `polygon`、对象 `detail_infos`，且没有 `text`。五个 matting id 的 data 含长度 1 到 64 的 `image`，且没有 `colorized_mask`。三个增强 id 与五个深度 id 的 data 含长度 1 到 64 的 `image`。
10. 三个 DINOv2 path 的 data `$ref` 仍是 `EnvelopeData`。示例 data 含整数 `dim` 与等长的 `embedding`，长度在 1 到 8，并且 200 示例仍能通过该 path 的 schema 校验。三个 `model.name` 分别是 `dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`。
11. `docs/openapi.json` 的 `info.x-contract-hash` 仍是 `b320c08dde6d040146cc649381361fae41acb57ea3ba9e531c842e6b0aece400`。表外 path 不因本项缺少示例而导致下面的自测命令失败。

## 测试义务

QA 对照上表 39 个 path，只读 path 上的两个 `application/json.example`，逐条核对验收标准 1 到 11。共用 section 的 id 要分别打开自己的 path，确认 `model.name` 是该 id。

重点看：`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 不是 `yolov8s`；`libface` 与 `centerface` 的 `landmarks` 非空；`superpoint` 同时有 `score`、`location`、`descriptor`；`msocrnet` 与三个 DINOv2 的 `$ref` 仍是 `EnvelopeData`，示例里仍有约定字段。

白盒确认 `--check` 在缺一条范围内示例、示例不符合 schema、或 `model.name` 与 catalog id 不一致时退出码为 1。交付树上的自测命令退出码为 0。不要求启动模型进程，不要求权重、engine 或数值对比。

## 自测命令

在仓库根目录执行。本机入口是 `python`；CI quality-gates 用 `python3` 调用同一脚本。两条命令的期望退出码都是 0。

`python scripts/gen_openapi.py --check`

`python scripts/check_consistency.py`

## RD 不得自行决定

- 不得从 39 个 id 里拿掉已有人读片段的 7 个，也不得用 `yolov8s` 的 path 或共享 schema 顶替 `yolov8n`、`yolov8l`、`yolov8x`。
- 不得把示例只写在 `components.schemas`，不得用 `$ref` 指向另一条 path 的示例，不得只写复数 `examples`。
- 不得改请求、响应、状态码、data `$ref`、`EnvelopeData`、`x-contract-hash`、产品 toml、engine、推理或进程模型来迁就示例。
- 不得手改 `docs/openapi.json` 或 `src/server/openapi_doc.h`，不得新增第三方校验包，不得把占位换成真实图像、真实权重输出或超过契约长度上限的向量。
- 不得把本项扩成数值门槛、基准行、doctor、README 或部署文档修改。
