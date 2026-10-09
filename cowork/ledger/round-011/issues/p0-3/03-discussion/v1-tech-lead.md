# 技术方案

- 问题: P0-3
- 版本: v1
- 对应列表: v3

## 边界

只改 `yolov7` 与 `yolov7x` 这一条后处理。这两个 id 共用 `YoloV7Detector`。输出向量长度大于 0，`output`、`518`、`532` 没有同时找到，并且形状为 `[1,3,H,W,C]`（5 维，batch 为 1，第二维为 3，H、W、C 都大于 0）的张量个数不是 3 时，同一推理 POST 从 HTTP 500、`status` 5、`status_str` 为 `model output empty`、`results[].data` 为 JSON null，变为 HTTP 500、`status` 6、`status_str` 为 `model output contract failed`、`results[].data` 仍为 JSON null。个数少于 3 与多于 3 是现有同一个失败分支，一并改返回码。

输出向量长度为 0 时保持 `status` 5、`status_str` 为 `model output empty`。检测框数为 0 的成功解码保持 `status` 0。

## 做法

### 推荐

在 `YoloV7Detector::postprocess` 里按输出向量是否为空分流这一个失败分支。向量长度大于 0 时返回已有的 `MODEL_OUTPUT_CONTRACT_FAILED`。向量长度为 0 时仍返回 `MODEL_EMPTY_OUTPUT`。`yolov8` 的做法是：名字在、但 dtype、秩、形状、buffer 或有限值不合约定时，`validated_f32_named_output` 返回契约失败；向量为空时返回空输出。这里沿用这个空与非空的分界，头绑定规则仍是 yolov7 自己的三头规则。

HTTP 已有通路不改。`run_items` 把后处理返回值写入该项状态，`aggregate_item_statuses` 把它写成顶层状态，`inference_result_to_unified` 写成 `status` 与 `status_str`，失败项不填 `data`，`http_status_of` 对 5 和 6 都给出 HTTP 500。因此只改后处理返回值，同一 POST 的线字段就会变成冻结句里的结果。`results[].status` 与顶层 `status` 同为 6。

无权重契约测放进现有的 `object_detection_output_contract_unittest`，用子类调用生产 `postprocess`，再经上述信封与 `http_status_of` 断言线字段。`test/CMakeLists.txt` 增加 ctest 别名 `YoloV7OutputContract`，过滤 `YoloV7OutputContract.*`。该二进制已在 `MORTRED_BUILD_FULL=ON` 的测试列表里。

代价：改一处返回，加无权重用例和一个 ctest 别名。不增加推理、session、engine 或显存。`yolov7` 与 `yolov7x` 同时生效。

### 否决

1. 在 `inference_result_to_unified` 或 HTTP 映射里把 `status` 5 改成 6。这条信封和映射是全部模型共用的。真·空输出会一起变成契约失败；若在信封里再识别 yolov7 的头形状，模型契约就抬进了 HTTP 层。
2. 把 yolov7 改成 yolov8 那种单头 `validated_f32_named_output("output0")`。`yolov7x` 的合法输出名字是 `output`/`598`/`612`，现有路径在三个名字不齐时按空间尺寸从大到小绑定恰好 3 个头。改成单头 `output0` 后，这条合法输出会变成 HTTP 500。

## 改动面

- `src/models/object_detection/yolov7_detector.inl` 的 `postprocess`：仅上述失败分支按向量是否为空分流。非空改为契约失败，空向量保持空输出。该分支的日志保持说明头数量不匹配。
- `test/object_detection_output_contract_unittest.cc`：无权重用例覆盖下面验收 1 至 6，断言的是 HTTP 状态、正文 `status`、`status_str`、`results[].status` 和 `results[].data`。
- `test/CMakeLists.txt`：登记 ctest 别名 `YoloV7OutputContract`。

## 明确不改

- `status_code.h` 的枚举、wire 值与文案。`http_status.h`、`item_exec.h`、`f32_output.h`、`http_wire.h`。
- 三个名字同时找到之后，dtype、通道数 `class_nums + 5`、buffer 长度或非有限值失败，保持契约失败。
- 恰好 3 个合格 `[1,3,H,W,C]` 头时按空间尺寸绑定并解码。`yolov7x` 的 `output`/`598`/`612` 走这条。
- letterbox 几何无效保持 `model input empty`（`status` 3，HTTP 400）。
- 解码成功且框数为 0 时保持 HTTP 200、`status` 0、`status_str` 为 `OK`。
- 产品 toml 的 `type=tensorrt`、`model_file_path`、engine 生成、gateway 与 supervisor、进程树。
- `yolov8`、`yolov5`、其余检测器、SAM。P0-2、P1-7 至 P1-12、D-8 至 D-12 的代码与文档。

## 契约

配置不变。tensor 的三头约定不变。HTTP 状态码在本条失败时仍是 500。

变的是该失败 POST 的业务字段：顶层 `status` 为 6，`status_str` 为 `model output contract failed`，`results` 与请求图片数相同，每一项 `status` 为 6，`data` 为 JSON null。`model.name` 分别为 `yolov7` 与 `yolov7x`。`partial` 仍为 false。

空输出向量的 POST 保持顶层 `status` 5、`status_str` 为 `model output empty`、每一项 `status` 为 5、`data` 为 JSON null。

## 性能与资源

返回发生在解码之前。无新常驻进程、session、engine 或显存峰值。cpu profile 不链 TensorRT，这条后处理与契约测仍在 `full-cpu` 上编译运行。不需要 `mortredctl prepare`。

## 验收标准

下列请求都是一张图的推理 POST。`yolov7` 的路径是 `/mortred_ai_server_v1/obj_detection/yolov7`，`yolov7x` 的路径是 `/mortred_ai_server_v1/obj_detection/yolov7x`。判定只看 HTTP 状态和响应 JSON。

1. `yolov7`：输出向量非空，`output`、`518`、`532` 没有同时存在，且没有一个 `[1,3,H,W,C]` 头。响应为 HTTP 500，顶层 `status` 为 6，`status_str` 为 `model output contract failed`，`model.name` 为 `yolov7`，`results` 有 1 项，该项 `status` 为 6，`data` 为 JSON null。
2. `yolov7x`：与第 1 条相同的张量条件。响应字段与第 1 条相同，`model.name` 为 `yolov7x`。
3. 两个 id：输出向量非空，三个名字没有同时存在，合格头只有 1 个。响应与第 1 条的 HTTP 与状态字段相同，`model.name` 为对应 id。
4. 两个 id：合格头只有 2 个，其余条件同第 3 条。响应同第 3 条。
5. 两个 id：合格头多于 3 个，名字没有同时存在，向量非空。响应同第 3 条。
6. 两个 id：输出向量长度为 0。响应为 HTTP 500，顶层 `status` 为 5，`status_str` 为 `model output empty`，对应项 `status` 为 5，`data` 为 JSON null。顶层 `status` 为 6 则本条失败。
7. 两个 id：没有 `518` 与 `532`，但有恰好 3 个 f32、有限值、C 等于 `class_nums + 5` 的 `[1,3,H,W,C]` 头，且请求几何的 source 与 network 边长都大于 0。响应为 HTTP 200，顶层 `status` 为 0，`status_str` 为 `OK`，`data` 不是 JSON null。
8. 下面的自测命令退出码为 0。输出里 `YoloV7OutputContract` 的通过条数不少于 4，并且含有第 1 至 7 条的两类 id。输出出现 `No tests were found`，或 gtest 只有 `0 tests`，则本条失败。

## 测试义务

QA 用第 1 至 7 条核对线字段。白盒确认三件事：用例调用的是生产 `YoloV7Detector::postprocess`；断言经过现有的 `inference_result_to_unified`、信封编码和 `http_status_of`；`yolov7` 与 `yolov7x` 的工厂入口都创建这个检测器。空向量用例必须断言仍是 `status` 5。恰好 3 头的用例必须断言仍是 `status` 0。不要求权重、GPU、TensorRT engine 或 golden。

## 自测命令

```bash
scripts/run_tests.sh build/full-cpu -R '^YoloV7OutputContract$' -V --output-on-failure
```

期望退出码 0。通过条数不少于 4，并覆盖验收 1 至 7。`No tests were found` 或 gtest `0 tests` 算未过。目录未配置时该脚本退出码为 1，不算通过。`build/full` 可替换 `build/full-cpu`，退出码要求相同。`tests-only` 编不出这条。

## RD 不得自行决定

- 空输出向量保持 `status` 5。框数为 0 的成功解码保持 `status` 0。
- 保持 `status` 的枚举、wire 值、文案和 HTTP 映射。失败项的 `data` 保持 JSON null。
- 保持三头绑定。恰好 3 个合格头、以及 `yolov7x` 的 `output`/`598`/`612`，保持可解码。
- 三个名字对齐之后已有的 dtype、通道、buffer、非有限值失败，保持契约失败。
- 测试走生产后处理再进现有信封。用另一套直接返回 6 的函数代替，不算通过。
- 不改产品 toml、engine、进程模型、`yolov8`、SAM，以及 P0-2、P1-7 至 P1-12、D-8 至 D-12。
