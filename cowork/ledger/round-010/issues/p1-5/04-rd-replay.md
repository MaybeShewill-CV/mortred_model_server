# 实现说明

- 方案: G:\Codex\mortred_model_server\cowork\ledger\round-010\issues\p1-5\03-design-final.md
- 方案版本: v2
- 分支: cowork/r10-p1-5-trt-golden
- 工作树: G:\Codex\mortred-cowork-r10-p1-5
- commit: 55f812fca57f2744d3350c9018cf38e7c30d2a4c
- 偏离: 无

## 行为变化
`Yolov8TrtDetection` 仍加载产品 toml `conf/model/object_detection/yolov8/yolov8s.toml`，不调用 `force_cpu`。输入 `bus.jpg`。golden 里每个 score >= 0.5 的框必须匹配到同类且 IoU >= 0.5 的预测框。engine 缺失时仍是 `MORTRED_SKIP_OR_FAIL_WEIGHTS("tensorrt engine not available")`。这段测试在 `golden_process_exit` 与全文件唯一的 `main` 之前。退出码 77、`NUMERICAL_NOT_RUN`、`NUMERICAL_PASSED` 仍在。`model_golden_test` 不链接 `gtest_main`，CTest 该项 `SKIP_RETURN_CODE` 为 77。ctest 名 `Yolov8TrtDetection` 与 `yolov8_detection` 都保留。`yolov8_detection` 的过滤仍是 `model_golden.yolov8_detection`，配置仍是 `conf/ci/yolov8_onnx_hosted.toml`。推理热路径未改：比对发生在 `model->run` 返回之后。

## 文件
- `test/model_golden_test.cc`
- `test/CMakeLists.txt`

合并的另一父提交是 `afff3131`。除上述两文件的冲突结果外，基线相对原分支的其余文件按合并进入，没有另改。

## 验证
| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `cmake --build build/full --target model_golden_test` | 0 | 重编 `model_golden_test` |
| `ctest --test-dir build/full -N -R Yolov8TrtDetection` | 0 | Test #86 Yolov8TrtDetection，Total Tests: 1 |
| `scripts/run_tests.sh build/full -R '^Yolov8TrtDetection$' --output-on-failure` | 0 | Passed，2.36 sec，不是 Skipped |

工作树 `git status --porcelain` 为空。未 push。基线 HEAD 仍是 `afff3131`。

## 未做
没有在基线上合入。没有改产品 toml、golden JSON、`force_cpu`、IoU 门限、推理热路径。

## 新发现
工作树没有 `3rd_party`，而 `build/full` 的缓存指向它。重配前临时把该目录链到基线依赖，测完已删除，未入库。

`Yolov8TrtDetection` 缺 engine 时的跳过文案是 `tensorrt engine not available`。`golden_process_exit` 只把含 `weights not available` 的跳过计为全部缺权重。只选中这一条且 engine 不在时，进程不会退出 77，也不会打印 `NUMERICAL_NOT_RUN`。本机 engine 在，自测是 Passed。两侧正文按要求原样保留。
