# 实现说明

- 方案: `cowork/ledger/round-011/issues/p1-12/03-design-final.md`
- 方案版本: v1
- 分支: `cowork/r11-p1-12-trt-product`
- 工作树: `G:\Codex\mortred-cowork-r11-p1-12`
- commit: 未提交
- 偏离: `mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`yolov7`、`libface`、`enlightengan`、`dinov2_vitb14` 这 9 条仍是退出码 1，标准输出含该 id 的 `product toml` 行，不含 `NUMERICAL_PASSED`。阈值、黄金茎名和失败文案未改。

核对结果：这 9 条与对应 CPU 用例调用同一个比较函数、读同一字段。CPU 用例在 `run()` 之后没有另做一步后处理。分类读 `class_id` 和 `scores`（`expect_scores`），检测读框（`expect_boxes`），`libface` 同时读 landmarks，`enlightengan` 读 `enhancement_result`，`dinov2_vitb14` 读 `embedding`。产品 toml 与对应 CPU 配置的分数阈值、NMS、输入边长，除 `libface` 产品输入为 320×240、CPU 用例为 640×480 外，其余一致。`libface` 这一尺寸差写在产品 toml 里，本轮未改 toml。

因此不是测试比错字段，也不是漏了 CPU 用例已有的后处理。过不了的是产品 engine 经现有 `run()` 得到的输出：

- `mobilenetv2`、`resnet`、`densenet`：`class_id` 是 230，黄金是 170；分数是可正可负的 O(1)，黄金分数约 `1e-4` 且总和约 1。绝对差约 0.2～6，超过 `1e-3`，失败文案含 `score mismatch at`。argmax 已经不同，所以不是测试里少做了一次 softmax。
- `yolov5l`：`detection count changed for yolov5_detection`，预测 5 框，黄金 7 框。
- `yolov6s`：`detection count changed for yolov6_detection`，预测 7 框，黄金 5 框。
- `yolov7`：分数差约 0.065～0.38，超过 `0.01`。
- `libface`：关键点距离约 3.4～25.8，超过 `3.0`。
- `enlightengan`：`fingerprint drift for enlightengan_enhancement`，均值绝对差约 7.000，超过 `1.0`。
- `dinov2_vitb14`：`embedding cosine similarity low for dinov2_feature_embedding`，余弦约 0.126，低于 `0.999`。

影响：这 9 个 id 仍达不到方案自测表要求的退出码 0。按「产品 engine 的输出本身过不了现有门时停下」处理，没有放宽阈值，没有提交。

## 行为变化

本轮没有改测试或产品代码。推理热路径没有动：比较仍在 `run()` 返回之后，现有 `init` / `run` 够用。

## 文件

无。工作树里上一轮未提交的 `test/model_golden_test.cc`、`test/model_golden_registry.h` 和 18 份 `test/golden/trt_<id>.json` 保持原样。

## 验证

未重新构建。二进制是工作树已有的 `_bin/model_golden_test`。每条一个进程、一个 `--gtest_filter`。未设置 `MORTRED_UPDATE_GOLDEN` 与 `MORTRED_CI_REQUIRE_WEIGHTS`。`PATH` 含 `/usr/local/cuda/bin`。`LD_LIBRARY_PATH` 含工作树 `_lib` 与 `3rd_party/libs`。

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.mobilenetv2` | 1 | 含 `product toml conf/model/classification/mobilenetv2/mobilenetv2.toml type=tensorrt device=gpu model_file_path=weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32` 和 `score mismatch at`；`class_id` 230 对 170；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.resnet` | 1 | 含对应 `product toml` 行和 `score mismatch at`；`class_id` 230 对 170；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.densenet` | 1 | 含对应 `product toml` 行和 `score mismatch at`；`class_id` 230 对 170；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.yolov5l` | 1 | 含对应 `product toml` 行和 `detection count changed for yolov5_detection`（5 对 7）；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.yolov6s` | 1 | 含对应 `product toml` 行和 `detection count changed for yolov6_detection`（7 对 5）；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.yolov7` | 1 | 含对应 `product toml` 行；分数差约 0.065～0.38，超过 `0.01`；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.libface` | 1 | 含对应 `product toml` 行；关键点距离约 3.4～25.8，超过 `3.0`；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.enlightengan` | 1 | 含对应 `product toml` 行和 `fingerprint drift for enlightengan_enhancement`，均值绝对差约 7.000；不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=TrtProductGolden.dinov2_vitb14` | 1 | 含对应 `product toml` 行和 `embedding cosine similarity low for dinov2_feature_embedding`，余弦约 0.126；不含 `NUMERICAL_PASSED` |

## 未做

没有放宽阈值，没有把这 9 条的期望退出码改成 1，没有改产品 toml、`src/` 或 `Yolov8TrtDetection`，没有补 P1-8 的 CPU 用例。没有跑 `TrtProductGolden.msocrnet`，没有跑 `TrtProductGolden.*`。因为这 9 条仍是退出码 1，没有提交。

## 新发现

无。
