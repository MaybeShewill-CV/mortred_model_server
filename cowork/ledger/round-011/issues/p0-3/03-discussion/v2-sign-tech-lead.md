# 第3步签字

- 问题: P0-3
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的审查条件已写入验收开头和第 3、4、5、7、8 条，以及自测命令。`YoloV7Detector::postprocess` 在 `output`/`518`/`532` 未齐、且形状为 `[1,3,H,W,C]` 的张量个数不是 3 时，现有同一分支返回 `MODEL_EMPTY_OUTPUT`。向量长度大于 0 时改为已有的 `MODEL_OUTPUT_CONTRACT_FAILED`（wire 6，文案 `model output contract failed`），向量长度为 0 时仍返回 `MODEL_EMPTY_OUTPUT`（wire 5，文案 `model output empty`）。`yolov7` 与 `yolov7x` 的工厂入口都是 `create_yolov7_detector`。单图失败时 `aggregate_item_statuses` 把该项状态写成顶层状态，且 `partial` 为 false；`inference_result_to_unified` 写入 `status` 与 `status_str`，失败项不填 `data`，信封把空 Document 编成 JSON null；`http_status_of` 对 5 和 6 都给出 HTTP 500，对 0 给出 200。恰好 3 个合格头仍绑定并解码；source 与 network 边长都大于 0 时，框数为 0 也由 `fill_object_detection` 写成 JSON 数组，`data` 不是 null。头个数按 5 维 `[1,3,H,W,C]` 点数，不含 dtype、有限值和通道数。验收 3、4、5 的个数分别是 1、2、大于 3，恰好 3 个不算这三条通过。验收 1 至 7 只认写明对应张量的那份响应 JSON。验收 8 要求该命令输出里这 7 条各有一行，并带上对应 id、HTTP 状态、顶层 `status` 和 `status_str`。自测命令是 `scripts/run_tests.sh build/full-cpu -R '^YoloV7OutputContract$' -V --output-on-failure`，期望退出码 0；目录未配置时该脚本退出码为 1。`object_detection_output_contract_unittest` 已在 `MORTRED_BUILD_FULL=ON` 的测试列表里，`full-cpu` 打开该开关，`tests-only` 编不出这条。范围停在这两个 id 的这一处返回。
