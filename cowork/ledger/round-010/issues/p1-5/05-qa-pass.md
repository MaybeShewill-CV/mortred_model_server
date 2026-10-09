# 测试报告

- 结论: pass
- 方案版本: v2
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: G:\Codex\mortred-cowork-r10-p1-5
- commit: 无
- 环境: WSL。工作树 HEAD `3f3f607a9b096e33004dcc9d3e3c5194ef0d0610`。nvidia-smi 列出 RTX 2070 SUPER。`/dev/dxg` 存在。`/dev/nvidia*` 不存在。engine `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16` 在工作树，测完已改回原名。libMNN.so 来自主仓库 `3rd_party/libs`。
- 缺的环境: 无
- 恢复条件: 无
- 未跑的验收: 无

## 用例
| 验收标准 | 类型 | 场景 | 命令 | 退出码 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 黑盒 | 别名可列出 | `ctest --test-dir build/full -N -R Yolov8TrtDetection` | 0 | 不少于 1 条 | Test #86 Yolov8TrtDetection，Total Tests: 1 | pass |
| 2 | 黑盒 | engine 在且 GPU 可计算 | `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$' --output-on-failure -V` | 0 | Passed，gtest 为通过 | ctest Passed。gtest `[  PASSED  ] 1 test`。日志 Loaded engine size: 23 MiB | pass |
| 3 | 黑盒 | IoU 门 | 同上 | 0 | golden 中每个 score >= 0.5 的框同类且 IoU >= 0.5，失败为 FAIL | golden 5 框分数均 >= 0.5。gtest `[       OK ] Yolov8TrtDetection.MatchesCpuGoldenBoxes` | pass |
| 4 | 白盒 | 产品 toml 未改 | `git diff HEAD -- conf/model/object_detection/yolov8/yolov8s.toml` | 0 | type=tensorrt，engine 路径不变 | diff 为空。`type = "tensorrt"`，`model_file_path` 为 `../weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16` | pass |
| 5 | 黑盒 | CPU 用例可被 ctest -N 单独列出 | `ctest --test-dir build/full -N -R yolov8_detection` | 0 | 名称含 yolov8_detection 的 CPU 用例，配置为 conf/ci/yolov8_onnx_hosted.toml | Test #87 yolov8_detection，Total Tests: 1。命令为 `--gtest_filter=model_golden.yolov8_detection`。该用例登记配置为 `conf/ci/yolov8_onnx_hosted.toml` | pass |
| 6 | 黑盒 | engine 缺失可 SKIP；设了 REQUIRE 则 FAIL；改回后须通过 | 藏起 engine 后跑同一条 TRT 测试；再以 `MORTRED_CI_REQUIRE_WEIGHTS=1` 跑；改回原名后再跑 | 0；8；0 | 未设变量时 SKIP 且进程退出码 0；设了则为 FAIL；engine 在则通过 | gtest `[  SKIPPED ]`，退出码 0；随后 Failed，退出码 8；改回后 gtest `[  PASSED  ] 1 test`，退出码 0 | pass |
| 白盒 | 白盒 | CPU 走 force_cpu，TRT 不走，比对方式不同 | 核对 `test/model_golden_test.cc` 与 `test/model_golden_registry.h` | 0 | CPU 的 yolov8_detection 经 `load_model_cfg` 调用 `force_cpu_backend`；TRT 用例不调用；TRT 比对为同类且 IoU >= 0.5 | CPU 用例走 `run_case` → `load_model_cfg` → `force_cpu_backend`，`expect_boxes` 含数量全等与 score 近等。TRT 用例只做 `fix_toml_paths`，按 score >= 0.5、同类、IoU >= 0.5 匹配，函数体内无 `force_cpu`、`load_model_cfg`、`EXPECT_NEAR` | pass |

## 缺陷
无

## 不适用
- 正确 catalog id、错误 id、未就绪模型：本方案只给产品 yolov8s 加 TRT 检测框门，不改 catalog。
- 空输入、极端尺寸、通道或色彩：验收未要求，用例输入固定为 `bus.jpg`。
- mask、类别分数形状与值域：本方案只核对检测框的同类与 IoU，不改输出契约。
- 错误码、超时、参数缺失：除 engine 缺失（验收 6）外，方案未要求。
- 后端分数近等：方案明确不要求 score 近等，TRT 与 CPU 的可比结果是验收 3 的 IoU 门。
- 其他家族 golden 与 HTTP：方案明确不改。

## 残余风险
隐藏 engine 且未设 `MORTRED_CI_REQUIRE_WEIGHTS` 时，ctest 汇总行写 Passed，进程退出码 0；gtest 正文是 `[  SKIPPED ]`。判定以 gtest 正文为准，与验收 6 一致。
