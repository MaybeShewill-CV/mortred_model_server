# 测试报告

- 结论: fail
- 方案版本: v2
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: G:\Codex\mortred-cowork-r10-p1-5
- commit: 无
- 环境: WSL2。nvidia-smi 列出 RTX 2070 SUPER。没有 `/dev/nvidia*`。`/dev/dxg` 存在。cuInit 返回 0，设备数 1。TensorRT 10.3。CUDA 12.6。本机 engine 为 `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16`。
- 缺的环境: 无
- 恢复条件: 无
- 未跑的验收: 无

这是挂起恢复后的第 5 步重跑。验收 2、3 已通过。验收 5 失败，不合入。上一份 `05-qa.md` 仍是当初的 blocked 记录。

## 用例
| 验收标准 | 类型 | 场景 | 命令 | 退出码 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 黑盒 | 别名可列出 | `ctest --test-dir build/full -N -R Yolov8TrtDetection` | 0 | 不少于 1 条 | Test #86 Yolov8TrtDetection，Total Tests: 1 | pass |
| 2 | 黑盒 | engine 在且 GPU 可计算 | `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$' --output-on-failure` | 0 | Passed，不是 Skipped | ctest Passed。gtest `[  PASSED  ] 1 test` | pass |
| 3 | 黑盒 | IoU 门 | 同上 | 0 | 每个 score >= 0.5 的框同类且 IoU >= 0.5 | golden 5 框均 >= 0.5，gtest OK | pass |
| 4 | 白盒 | 产品 toml 未改 | `git diff HEAD -- conf/model/object_detection/yolov8/yolov8s.toml` | 0 | type=tensorrt 不变 | diff 为空 | pass |
| 5 | 黑盒 | CPU 用例可被 ctest -N 单独列出 | `ctest --test-dir build/full -N -R yolov8_detection` | 0 | 列出名称含 yolov8_detection、且不是 Yolov8TrtDetection 的 CPU 用例 | Total Tests: 0。CPU 用例的 ctest 名是 `model_golden_test` | fail |
| 6 | 黑盒 | engine 缺失可 SKIP；设了 REQUIRE 则 FAIL；engine 在不得 SKIP | 藏起 engine 后重跑；`MORTRED_CI_REQUIRE_WEIGHTS=1` 再跑 | 0；8 | 未设变量时 SKIP；设了则 FAIL | SKIPPED 后 REQUIRE=1 为 Failed | pass |

## 缺陷
`ctest --test-dir build/full -N -R yolov8_detection` 退出码 0，`Total Tests: 0`。全表里含 yolov8 的只有 `Yolov8TrtDetection`。gtest 里已有 `model_golden.yolov8_detection`，登记配置已是 `conf/ci/yolov8_onnx_hosted.toml`。缺的是这个 ctest 名。

## 残余风险
验收 5 未过，不合入。
