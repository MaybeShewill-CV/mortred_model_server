# 技术方案

- 问题: P1-3 检测缺 dtype / NaN / Inf 拒绝矩阵
- 版本: v1
- 对应列表: `cowork/ledger/round-008/02-p0-p1-final.md` P1-3

## 边界

只补 YOLOv8 的无权重拒绝矩阵。`validated_f32_named_output` 已对 dtype 与非有限值返回 `MODEL_OUTPUT_CONTRACT_FAILED`。现有 `YoloV8RejectsMalformedOutput` 覆盖了 dtype 与 NaN，没有 +Inf。不改检测解码，不加载权重，不加 `GTEST_SKIP`。不做 P1-4。

## 做法

### 推荐

在 `test/object_detection_output_contract_unittest.cc` 增加 `YoloV8RejectsInfinity`：shape `{1,84,1}` 的 f32 输出填 `+infinity`，`postprocess` 返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`。dtype 与 NaN 继续由已有用例覆盖。目标已在 FULL 的 `TEST_LIST` 里，不新挂 tests-only。

### 否决

为了矩阵去改 `f32_output.h`。用缺权重的 golden SKIP 代替。

## 改动面

- `test/object_detection_output_contract_unittest.cc`

## 明确不改

`src/`、golden、P1-4、其它检测家族。

## 契约

坏 dtype、NaN、+Inf 都是 `MODEL_OUTPUT_CONTRACT_FAILED`。

## 性能与资源

无权重、不启服务。

## 验收标准

1. **AC1** `YoloV8RejectsMalformedOutput` 仍断言 dtype 与 NaN 为 `MODEL_OUTPUT_CONTRACT_FAILED`。
2. **AC2** `YoloV8RejectsInfinity` 对 `+infinity` 返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`。
3. **AC3** 源码无 `GTEST_SKIP`，不读权重。
4. **AC4** FULL 上 `scripts/run_tests.sh build/full -R 'object_detection_output_contract'` 退出码 0，且实际执行该测试，不是 0 tests。

## 测试义务

WSL FULL。不要求 GPU。

## 自测命令

```bash
scripts/run_tests.sh build/full -R 'object_detection_output_contract'
```

期望退出码 0，且测试条数大于 0。

## RD 不得自行决定

不得改解码逻辑来放行 Inf。不得用 SKIP 冒充通过。不得改 P1-4。
