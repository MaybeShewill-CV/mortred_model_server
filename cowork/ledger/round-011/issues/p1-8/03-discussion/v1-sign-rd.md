# 第3步签字

- 问题: P1-8
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 现有 `run_case` 七步、`load_model_cfg` 去 `../` 并强制 cpu、`MORTRED_SKIP_OR_FAIL_WEIGHTS` 和 `golden_process_exit` 已经覆盖缺权重跳过、全跳过退出 77 且打印 `NUMERICAL_NOT_RUN`、`MORTRED_CI_REQUIRE_WEIGHTS=1` 时失败退出 1，以及更新黄金只跳过、不打 `NUMERICAL_PASSED`。19 个 creator 与段名和模型构造函数一致。检测的 IoU 0.5、分数差 0.01、`low IoU`，以及 embedding 余弦 0.999、`embedding cosine similarity low`，都是现成比较。分割是 `CV_32SC1` 的 `segmentation_result`，抠图是 0–255 的 `CV_32SC1` `matting_result`，深度是 `CV_32FC1` 的 `depth_map`；这三类比较可以加在 registry 里，不改现有 fingerprint 宏。两个 DINOv2 的 CI toml 已指向方案中的 static onnx。其余 17 份按现有 CI toml 写成 `type=onnx`、`device=cpu` 即可。YOLOv7x 已按空间尺寸回退三个 head，YOLOv8 n/l/x 与现有 `output0` 契约相同，PP-Human 的 `x` / `softmax_0.tmp_0` 与 `[1,2,H,W]` 后处理一致，Metric3D 使用的 `focal_length` 与 `intrinsic` 在产品 params 里。cpu profile 已能编过同类 ONNX 用例，不必改 CMake、产品 toml 或 `src/models`。
