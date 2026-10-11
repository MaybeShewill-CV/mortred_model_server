# 第3步签字

- 问题: P1-11
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 基线 `806813aa` 上，这四条已经在 `model_golden_test` 里经 `run_case` 走完缺权重判断、`load_model_cfg`、`force_cpu_backend`、现有 creator、`init`、读图和一次 `run`。`compare_scene_segmentation` 与 `compare_matting` 目前只调用 `expect_fingerprint`，而且只被 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting` 使用。改这两个函数不会碰到增强、SAM 或其他 fingerprint。

现有后处理已经给出方案要的掩码：`bisenetv2` 与 `pphuman` 的 `segmentation_result` 是 `CV_32SC1` 类别图；`modnet` 的 `matting_result` 是乘 255 后的 `CV_8UC1`；`ppmatting` 是乘 255 后的 `CV_32SC1`。比较按像素整数值读这两种单通道整型即可，不必改模型输出类型，也不必改 HTTP、产品 toml 或进程。四份 `conf/ci` 已是 onnx/cpu 和方案中的 onnx 路径；对应产品文件仍是 `type = "tensorrt"`，路径仍是 engine。

退出码不用新写。`MORTRED_SKIP_OR_FAIL_WEIGHTS` 仍负责 `weights not available`。四条都因缺权重跳过时，现有 `golden_process_exit` 打印 `NUMERICAL_NOT_RUN` 并返回 77。`MORTRED_CI_REQUIRE_WEIGHTS=1` 时同一处变成失败，退出码 1。`MORTRED_UPDATE_GOLDEN=1` 沿用现有 `GTEST_SKIP`，跳过文案表示黄金已更新；这种跳过既不是权重跳过，也不是数值通过，所以进程可以退出 0，且标准输出不含 `NUMERICAL_PASSED`。缺黄金、尺寸或类型不符、分割类别少于两个、抠图前景少于 2、重合低于 `0.90`，都用断言失败，退出码 1。只有平均重合或前景重合 `< 0.90` 的那条断言带上 `mask overlap` 和 `0.90`，其余失败文案不带这两个字，也不再调用 `expect_fingerprint`，因此不含 `fingerprint drift`。

验收 6 把每个类别加 1 后，类别集合整体错开，各类 IoU 为 0，平均重合低于 `0.90`，同时类别数不变，不会落到「少于两个类别」那条。验收 7 把前景改成与原前景不相交且至少 2 像素时，前景 IoU 为 0，会落到 `mask overlap`，而不是「前景少于 2」。

`scripts/golden_drift_check.py` 对黄金原始字节做 sha256。删掉这四份 png、换上同名 json，并只改基线里这四个键，`case_names`、`case_count` 30 和 `golden_count` 26 可以保持不变。基线上没有 `compare_scene_segmentation_overlap` 或 `compare_matting_overlap`。服务热路径不用改：比较发生在现有测试进程的一次 `run` 之后，不增加推理拷贝、色彩往返或第二套进程。
