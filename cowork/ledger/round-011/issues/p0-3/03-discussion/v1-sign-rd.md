# 第3步签字

- 问题: P0-3
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `YoloV7Detector::postprocess` 在 `output`/`518`/`532` 未齐、且 `[1,3,H,W,C]` 头不是 3 个时，现有同一分支返回 `MODEL_EMPTY_OUTPUT`。按输出向量是否为空改这一处返回即可：向量非空返回已有的 `MODEL_OUTPUT_CONTRACT_FAILED`（wire 6，文案 `model output contract failed`），向量为空仍返回 `MODEL_EMPTY_OUTPUT`。`yolov7` 与 `yolov7x` 的工厂入口都是 `create_yolov7_detector`。`run` 原样返回后处理状态；单图失败时 `aggregate_item_statuses` 把该项状态写成顶层状态，且 `partial` 为 false；`inference_result_to_unified` 写入 `status` 与 `status_str`，失败项不填 `data`，信封把空 Document 编成 JSON null；`http_status_of` 对 5 和 6 都给出 HTTP 500，对 0 给出 200。恰好 3 个合格头仍绑定并解码；source 与 network 边长都大于 0 时 `make_letterbox_geometry` 成功，框数为 0 也由 `fill_object_detection` 写成 JSON 数组，`data` 不是 null。契约测所在二进制已在 `MORTRED_BUILD_FULL` 的测试列表里，`full-cpu` 打开该开关；ctest 别名按现有 `add_test` 登记 `YoloV7OutputContract` 即可。验收 1 至 7 只看 HTTP 状态和响应 JSON，第 8 条只看自测命令的退出码和输出。
