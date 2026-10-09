# 测试报告

- 结论: pass
- 方案版本: v2
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: G:\Codex\mortred-cowork-r10-p1-5
- commit: 无
- 环境: WSL。`/dev/dxg` 存在。engine 为工作树符号链接 `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16`，指向基线同名文件，24,590,332 字节。`LD_LIBRARY_PATH` 含基线 `3rd_party/libs` 与 `_lib`，`ldd` 无缺失库。二进制 `_bin/model_golden_test`（2026-10-09 16:25）含 `Yolov8TrtDetection` 与 `NUMERICAL_PASSED`。
- 缺的环境: 无
- 恢复条件: 无
- 未跑的验收: 无

## 用例
| 验收标准 | 类型 | 场景 | 命令 | 退出码 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 黑盒 | 别名可列出 | `ctest --test-dir build/full -N -R Yolov8TrtDetection` | 0 | 不少于 1 条 | `Test #86: Yolov8TrtDetection`，`Total Tests: 1` | pass |
| 2 | 黑盒 | engine 在且 GPU 可计算 | `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$' --output-on-failure -V` | 0 | Passed | `Passed` 2.20 sec。gtest `[  PASSED  ] 1 test.`，另有一行 `NUMERICAL_PASSED` | pass |
| 3 | 黑盒 | golden 中 score >= 0.5 的框同类且 IoU >= 0.5 | `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$' --output-on-failure -V` | 0 | 五框都匹配，通过 | 同上一次执行。golden 5 框 score 均 >= 0.5（class 0：0.9140625、0.91259765625、0.8759765625、0.55712890625；class 5：0.8984375）。gtest 通过并打印 `NUMERICAL_PASSED`，输出无 IoU 失败 | pass |
| 4 | 黑盒 | 产品 toml 未改 | `git diff HEAD -- conf/model/object_detection/yolov8/yolov8s.toml` | 0 | `type=tensorrt`，engine 路径不变 | diff 为空。`type = "tensorrt"`，`model_file_path = "../weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16"` | pass |
| 5 | 黑盒 | CPU 用例可单独列出 | `ctest --test-dir build/full -N -R yolov8_detection` | 0 | 列出名称含 `yolov8_detection` 的那一条，配置为 `conf/ci/yolov8_onnx_hosted.toml` | `Total Tests: 1`。唯一名称 `Test #87: yolov8_detection`。命令为 `_bin/model_golden_test --gtest_filter=model_golden.yolov8_detection`。该用例登记配置为 `conf/ci/yolov8_onnx_hosted.toml` | pass |
| 6 | 黑盒 | 藏起 engine 后跳过、要求权重时失败、改回后通过 | 同一条 `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$' --output-on-failure -V`：未设 `MORTRED_CI_REQUIRE_WEIGHTS`；再设 `MORTRED_CI_REQUIRE_WEIGHTS=1`；改回原名后再跑 | 0；8；0 | 退出码 0 且 gtest SKIPPED；设了则为 FAIL；改回后通过。跳过文案为 `tensorrt engine not available`，进程退出码 0 | 未设变量：退出码 0，gtest `[  SKIPPED ]`，文案 `tensorrt engine not available`，无 `NUMERICAL_NOT_RUN`。设了变量：退出码 8，gtest `[  FAILED ]`，同一文案。改回原名后：退出码 0，`Passed` 1.54 sec，`NUMERICAL_PASSED`。符号链接已恢复，隐藏名已不存在 | pass |
| 白盒 | 白盒 | CPU 仍走 force_cpu；TRT 不走；比对为同类 IoU；全文件一份 main | 核对 `test/model_golden_test.cc` 与 `test/model_golden_registry.h` | 0 | CPU 的 `yolov8_detection` 走 `force_cpu`；TRT 用例没有 `force_cpu`；比对不是数量全等加 score 近等；一份 `main` | `MAIN_COUNT 1`。CPU 用例经 `run_case` → `load_model_cfg` → `force_cpu_backend`。TRT 函数体只有 `fix_toml_paths`，按 score >= 0.5、同类、`k_box_iou_thresh`（0.5）匹配。函数体内无 `force_cpu`、`load_model_cfg`、`EXPECT_NEAR`、数量全等 | pass |

## 缺陷

## 不适用

- 正确 catalog id、错误 id、未就绪模型：方案不改 catalog、HTTP 和进程模型。
- 空输入、极端尺寸、通道或色彩：方案只规定输入 `bus.jpg`。
- 输出结构、坐标、mask、类别分数的 HTTP 形状和值域：方案不改 HTTP；数值门是现有 golden 的同类 IoU。
- 错误码、超时、参数缺失：本条只规定 engine 缺失时的 SKIP/FAIL，以及数值不匹配时为 FAIL。
- 声明后端之外的可比较结果：CPU golden 行为明确不改；验收 5 只要求列出用例和配置路径。
- 整份已有 golden 与 CI：本轮重跑了 `Yolov8TrtDetection`，并列出 `yolov8_detection`。golden JSON 未改。

## 残余风险

- `yolov8_detection` 本轮核对了 ctest 名称和 `conf/ci/yolov8_onnx_hosted.toml`，没有重跑它的 CPU 数值。
- TRT 门不要求 score 近等，分数偏离 golden 仍可通过。
- 数值门只覆盖 `bus.jpg` 一张图。
