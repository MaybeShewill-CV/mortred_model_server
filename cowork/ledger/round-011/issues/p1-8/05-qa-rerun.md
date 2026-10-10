# 测试报告

- 结论: blocked
- 方案版本: v2
- 分支: cowork/r11-p1-8-cpu-golden
- 工作树: G:\Codex\mortred-cowork-r11-p1-8
- commit: 无
- 环境: WSL，可见内存约 19GiB。二进制是工作树 `_bin/model_golden_test`。回修提交 `aedd09c2dc3c1ffdab5d2fca3850075eca687e15` 只新增 `test/golden/depth_vitl14.json` 和 `test/golden/metric3d_512.json`。
- 缺的环境: 没有 `./build/full/test/model_golden_test`。`model_golden.msocrnet` 在这台机器上单独推理会涨到约 18GB 后被杀掉，`test/golden/msocrnet.json` 仍不存在。
- 恢复条件: 用 `cmake --preset full` 建成 `./build/full/test/model_golden_test` 后，只重跑该二进制的 `--gtest_list_tests`，确认 `Yolov8TrtDetection` 仍在，且新增数值用例只有 `model_golden.` 下这 19 个名字。msocrnet 要在能容下这次推理内存的机器上，由 CPU onnx 生成并提交 `test/golden/msocrnet.json`，再单进程跑 `model_golden.msocrnet`，退出码 0 且标准输出含 `NUMERICAL_PASSED`。
- 未跑的验收: 验收 3 的 msocrnet，以及把 19 条放进同一进程的那条过滤。验收 11 的 `./build/full/test/model_golden_test --gtest_list_tests`。

## 用例
| 验收标准 | 类型 | 场景 | 命令 | 退出码 | 预期 | 实际 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 黑盒 | 列出全部用例 | `./_bin/model_golden_test --gtest_list_tests` | 0 | 19 个新名字和六个旧名字都在 | 沿用 05-qa.md，本次未重跑 | pass |
| 2 | 黑盒 | CI toml 与产品 toml | 读取 19 份 CI toml 与产品 toml | 0 | onnx/cpu，产品仍是 tensorrt | 沿用 05-qa.md，本次未重跑 | pass |
| 3 | 黑盒 | 数值复跑 | 16 条沿用上一份；本次单独跑 `depth_vitl14` 与 `metric3d_512` | 16 条 0；本次两条 0；msocrnet 未跑 | 黄金在时退出码 0 且含 `NUMERICAL_PASSED` | 16 条沿用上一份通过。本次 `depth_vitl14` 退出码 0，含 `NUMERICAL_PASSED`，15521 ms。本次 `metric3d_512` 退出码 0，含 `NUMERICAL_PASSED`，12097 ms。`msocrnet` 未跑 | blocked |
| 4 | 黑盒 | 权重全缺 | 移走 19 个 onnx 后跑 19 条过滤 | 77 | 退出码 77 且含 `NUMERICAL_NOT_RUN` | 沿用 05-qa.md，本次未重跑 | pass |
| 5 | 黑盒 | 要求权重 | `MORTRED_CI_REQUIRE_WEIGHTS=1` 跑同一过滤 | 1 | 退出码 1，不含 `NUMERICAL_NOT_RUN`，含 `weights not available` | 沿用 05-qa.md，本次未重跑 | pass |
| 6 | 黑盒 | `nanodet_416` 改类别再移框 | 两次破坏并分别改回 | 1，0，1，0 | 失败输出分别含 `class_id` 与 `low IoU` | 沿用 05-qa.md，本次未重跑 | pass |
| 7 | 黑盒 | 分割与抠图降低重合 | `pphuman_lite` 与 `ppmatting_v2` | 1 然后 0 | 含 `mask overlap` 与 `0.90` | 沿用 05-qa.md，本次未重跑 | pass |
| 8 | 黑盒 | embedding 取反 | `dinov2_vits14` | 1 然后 0 | 含 `embedding cosine similarity low` | 沿用 05-qa.md，本次未重跑 | pass |
| 9 | 黑盒 | 深度乘 2 | `depth_vits14` | 1 然后 0 | 含 `depth error` 与 `0.02` | 沿用 05-qa.md，本次未重跑 | pass |
| 10 | 黑盒 | 涂色 fingerprint png | `bisenetv2_segmentation` | 1 | 含 `fingerprint drift` | 沿用 05-qa.md，本次未重跑 | pass |
| 11 | 黑盒 | 测试列表 | full-cpu 列表沿用；full 列表未跑 | full-cpu 0；full 未跑 | full-cpu 没有新 TensorRT 套件；full 里 `Yolov8TrtDetection` 仍在 | full-cpu 沿用上一份。`./build/full/test/model_golden_test` 仍不存在 | blocked |
| 白盒 | 白盒 | 两份新深度黄金 | 解析 `depth_vitl14.json` 与 `metric3d_512.json` | 无 | 有限 float 单通道，且有绝对值大于 `1e-6` 的像素 | 两份都是 375×1242、长度 465750。`depth_vitl14` 有 432977 个像素绝对值大于 `1e-6`。`metric3d_512` 全部像素绝对值都大于 `1e-6` | pass |

## 缺陷
无。

## 不适用
方案没有 HTTP 变化。产品 TensorRT 数值比较留给 P1-12。

## 残余风险
16 条通过是分进程结果。19 条放在同一个进程里没有跑。`msocrnet.json` 仍不存在。
