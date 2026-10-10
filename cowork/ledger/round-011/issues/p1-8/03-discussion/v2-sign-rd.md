# 第3步签字

- 问题: P1-8
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 19 条都能走现有 `run_case`：`load_model_cfg` 去掉 `../` 并强制 cpu，缺权重在 `init` 前走 `MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available")`。`golden_process_exit` 已覆盖全跳过退出 77 且打印 `NUMERICAL_NOT_RUN`、`MORTRED_CI_REQUIRE_WEIGHTS=1` 时退出 1，以及全部数值通过时打印 `NUMERICAL_PASSED`。`MORTRED_UPDATE_GOLDEN=1` 仍是 `GTEST_SKIP`，进程可以退出 0，标准输出没有 `NUMERICAL_PASSED`。检测沿用 `expect_boxes`，IoU 门 0.5、分数差 0.01，不足时输出含 `low IoU`；只改 `class_id` 时 `EXPECT_EQ` 的失败文本含 `class_id`。Embedding 沿用 `expect_embeddings`，余弦门 0.999，失败文本含 `embedding cosine similarity low`。分割是全分辨率 `CV_32SC1` 的 `segmentation_result`，抠图是乘 255 后的 `CV_32SC1` alpha，深度是全分辨率 `CV_32FC1` 的 `depth_map`。后三类比较加在 registry，验收 7、9 的 UTF-8 JSON 可由 `MORTRED_UPDATE_GOLDEN=1` 写入 `test/golden/<id>.json`。现有 fingerprint 宏继续服务 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting`。段名与构造函数一致。YOLOv7x 已按空间尺寸回退三个 head；YOLOv8 n/l/x 的产品 params 都是 80 类、640，后处理认 `output0`。两个 DINOv2 CI toml 已是 onnx/cpu 并指向方案中的 static onnx。Metric3D 产品 params 含 `focal_length` 与 `intrinsic`。`model_golden_test` 已链接 `models`，自带 `main`，`SKIP_RETURN_CODE` 为 77。现有 CMake、产品 toml、`src/models` 和进程模型已经够用。
