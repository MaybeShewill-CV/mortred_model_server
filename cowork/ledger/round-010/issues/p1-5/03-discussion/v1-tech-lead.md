# 技术方案

- 问题: P1-5
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只给产品模型 yolov8s 增加一条 TRT 检测框数值门。加载 `conf/model/object_detection/yolov8/yolov8s.toml`，不调用 `force_cpu_backend`。输入 `demo_data/model_test_input/object_detection/bus.jpg`。对照 `test/golden/yolov8_detection.json`。不改 CPU golden。不改其他家族。不新增 StatusCode。不改进程模型。不改产品 toml 的 type 和 engine 路径。

engine 在磁盘上时必须真跑并比对，不得 SKIP。engine 不在时沿用 `MORTRED_SKIP_OR_FAIL_WEIGHTS`。本机验收时 engine 已在，结果必须是 Passed。

## 做法

### 推荐

在 `model_golden_test` 里新增套件 `Yolov8TrtDetection`。`add_test(NAME Yolov8TrtDetection ... --gtest_filter=Yolov8TrtDetection.*)`。仅 `MORTRED_HAS_TRT` 时编译。

对 golden 里每个 `score >= 0.5` 的框，预测中必须有一个未占用框，`class_id` 相同且 IoU >= 0.5。不要求框数量相等，不要求 score 近等。当前 golden 五框都要匹配。

### 否决

放开全体 CPU golden 走 GPU。只做 session 形状测试。另存一份逐位 TRT golden。

## 改动面

- `test/model_golden_test.cc` 或同二进制的新测文件
- 必要时在 registry 加「不强制 CPU」的加载辅助，不得改现有 `load_model_cfg` / `force_cpu_backend`
- `test/CMakeLists.txt` 的别名

## 明确不改

`force_cpu_backend`、现有 `yolov8_detection` 用例、`yolov8_detection.json` 数值、产品 toml、HTTP、进程模型。

## 契约

产品 toml 仍是 `type=tensorrt`，engine 仍是 `yolov8s.engine.static_bs1.fp16`。比对门限固定为同类 IoU >= 0.5，覆盖全部 golden score >= 0.5 的框。

## 性能与资源

单进程短时加载一份 engine，一张图。不拉起 gateway。

## 验收标准

1. `ctest --test-dir build/full -N -R Yolov8TrtDetection` 列出不少于 1 条。0 条失败。
2. 本机 engine 在且 GPU 可见时，`scripts/run_tests.sh build/full -R Yolov8TrtDetection --output-on-failure` 退出码 0，结果是 Passed，不是 Skipped。
3. 该结果对 golden 中每个 score >= 0.5 的框都有同类且 IoU >= 0.5 的匹配。失败是 FAIL，不是 SKIP。
4. `yolov8s.toml` 仍是 type=tensorrt，engine 路径不变。
5. CPU 的 `yolov8_detection` 仍走 onnx hosted toml 和 force_cpu。
6. engine 不存在且未设 `MORTRED_CI_REQUIRE_WEIGHTS` 时可以 SKIP；设了则为 FAIL。engine 已存在时不得 SKIP。

## 测试义务

QA 在本机有 engine 时执行验收 1–3。白盒确认没有走 force_cpu，且不是数量全等加 score 近等。不要求其他家族或 HTTP。

## 自测命令

```bash
ctest --test-dir build/full -N -R Yolov8TrtDetection
scripts/run_tests.sh build/full -R Yolov8TrtDetection --output-on-failure
```

第一条期望退出码 0 且条数不少于 1。第二条期望退出码 0 且 Passed。

## RD 不得自行决定

不得改 force_cpu。不得只交 session 测试。不得换模型。不得在 engine 存在时 SKIP。不得改 IoU 门限。不得省略别名。
