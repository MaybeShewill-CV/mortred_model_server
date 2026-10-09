# 第3步签字

- 问题: P0-3
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `YoloV7Detector::postprocess` 在 `output`/`518`/`532` 未齐且 `[1,3,H,W,C]` 头个数不是 3 时已是同一失败分支，现返回 `MODEL_EMPTY_OUTPUT`。按 `outputs` 是否为空改成 `MODEL_OUTPUT_CONTRACT_FAILED` 或保持 `MODEL_EMPTY_OUTPUT` 即可。`run_impl` 原样返回该状态；`run_items`、`aggregate_item_statuses`、`inference_result_to_unified`、`envelope::encode`、`http_status_of` 已把状态 6 写成 HTTP 500、`model output contract failed`、失败项 `data` 为 JSON null、`partial` 为 false。`yolov7` 与 `yolov7x` 的 `model_section` 就是这两个 id，工厂都走 `create_yolov7_detector`。恰好 3 头的空间绑定、名字对齐后的 dtype/通道/buffer/有限值检查、非法 letterbox 的 `MODEL_EMPTY_INPUT_IMAGE` 都不在这条分支上。无权重测试可沿用子类调用生产 `postprocess`，再走现有信封头文件；`full-cpu` 已开 `MORTRED_BUILD_FULL`，ctest 别名和 `scripts/run_tests.sh` 的 `-R` 转发都已有。
