# 第3步签字

- 问题: P1-11
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 这四条已经在基线 `model_golden_test` 里经 `run_case` 走完缺权重判断、`load_model_cfg`、`force_cpu_backend`、现有 creator、`init`、读图和一次 `run`。`compare_scene_segmentation` 与 `compare_matting` 目前只调用 `expect_fingerprint`，而且只被 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting` 使用。改比较函数不会碰到增强、SAM 或其他 fingerprint。

现有输出类型和方案一致，不用改模型。`bisenetv2` 与 `pphuman` 的 `segmentation_result` 是源图尺寸的 `CV_32SC1` 类别图。`modnet` 的 `matting_result` 是乘 255 后的 `CV_8UC1`，`ppmatting_512` 是乘 255 后的 `CV_32SC1`。头文件里已有 rapidjson 读写和 `write_golden_text`，全分辨率整数 JSON 可以在同一次比较里生成和读取。类别平均 IoU 与前景 `>= 128` 的 IoU 都是测试进程内的整数像素统计，不新增进程，不加载 engine。

退出码不用第二套实现，也不用改 `golden_process_exit`。四条都因缺权重跳过时，现有逻辑已经退出 77 并打印 `NUMERICAL_NOT_RUN`；`MORTRED_CI_REQUIRE_WEIGHTS=1` 时同一处失败且退出 1。`MORTRED_UPDATE_GOLDEN=1` 的跳过文案不是 `weights not available`，也不会被计成数值通过，因此可以退出 0 且不打印 `NUMERICAL_PASSED`。缺黄金、类型不符、尺寸不符和重合低于 `0.90` 用断言失败即可得到退出码 1。验收 7 把抠图黄金改成只有 1 个前景像素时，该黄金既触发「前景像素数小于 2 则失败」，对真实掩码的前景 IoU 也会低于 `0.90`；失败信息写入 `mask overlap` 和 `0.90` 即可，不会被记成通过。

`scripts/golden_drift_check.py` 对黄金原始字节做 sha256。删掉这四份 png、换上同名 json，并只改基线里这四个键，`case_names`、`case_count` 30 和 `golden_count` 26 可以保持不变。基线上没有 `compare_scene_segmentation_overlap` 或 `compare_matting_overlap`。四份 `conf/ci` 已是 onnx/cpu 和方案中的 onnx 路径，对应产品 toml 仍是 `type = "tensorrt"` 且路径为 engine。比较发生在现有一次 `run` 之后，服务热路径不用改。
