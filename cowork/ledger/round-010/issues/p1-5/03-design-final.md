# 技术方案

- 问题: P1-5
- 版本: v2
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只给产品模型 yolov8s 增加一条 TRT 检测框数值门。加载产品 toml，不调用 force_cpu。输入 bus.jpg。对照现有 golden。不改 CPU golden 行为。不改产品 toml。engine 在磁盘上时必须真跑，不得 SKIP。

## 做法

### 推荐

在 model_golden_test 新增套件 Yolov8TrtDetection，并 add_test 别名。仅 MORTRED_HAS_TRT 时编译。golden 里每个 score >= 0.5 的框必须匹配到同类且 IoU >= 0.5 的预测框。五框都要匹配。不要求 score 近等。

### 否决

全体 CPU golden 改走 GPU。只做 session 形状测试。另存逐位 TRT golden。

## 改动面

测试文件与 CMake 别名。不得改现有 load_model_cfg / force_cpu_backend。

## 明确不改

force_cpu、现有 yolov8_detection 用例、golden JSON 数值、产品 toml、HTTP、进程模型。

## 契约

产品 toml 仍是 type=tensorrt，engine 仍是 yolov8s.engine.static_bs1.fp16。门限固定为同类 IoU >= 0.5。

## 性能与资源

单进程短时加载一份 engine，一张图。不拉起 gateway。

## 验收标准

1. `ctest --test-dir build/full -N -R Yolov8TrtDetection` 列出不少于 1 条。0 条失败。
2. 本机 engine 在且 GPU 可见时，`scripts/run_tests.sh build/full -R Yolov8TrtDetection --output-on-failure` 退出码 0，结果是 Passed，不是 Skipped。
3. 该结果对 golden 中每个 score >= 0.5 的框都有同类且 IoU >= 0.5 的匹配。失败是 FAIL，不是 SKIP。
4. yolov8s.toml 仍是 type=tensorrt，engine 路径不变。
5. `ctest --test-dir build/full -N` 能单独列出名称含 yolov8_detection、且不是 Yolov8TrtDetection 的 CPU 用例；该用例登记的配置路径是 conf/ci/yolov8_onnx_hosted.toml，不是产品 yolov8s.toml。
6. engine 不存在且未设 MORTRED_CI_REQUIRE_WEIGHTS 时可以 SKIP；设了则为 FAIL。engine 已存在时不得 SKIP。

## 测试义务

白盒确认 CPU 的 yolov8_detection 仍走 force_cpu，TRT 用例没有走 force_cpu，且比对不是数量全等加 score 近等。

## 自测命令

```bash
ctest --test-dir build/full -N -R Yolov8TrtDetection
scripts/run_tests.sh build/full -R Yolov8TrtDetection --output-on-failure
```

第一条期望退出码 0 且条数不少于 1。第二条期望退出码 0 且 Passed。

## RD 不得自行决定

不得改 force_cpu。不得在 engine 存在时 SKIP。不得改 IoU 门限。不得省略别名。不得换模型。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v2 | `cowork/ledger/round-010/issues/p1-5/03-discussion/v2-sign-pm.md` |
| cv-rd | 同意 | 无 | v2 | `cowork/ledger/round-010/issues/p1-5/03-discussion/v2-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v2 | `cowork/ledger/round-010/issues/p1-5/03-discussion/v2-sign-tech-lead.md` |
