# 技术方案

- 问题: P1-2 深度族缺不依赖权重的假 tensor 后处理契约测试
- 版本: v1
- 对应列表: `cowork/ledger/round-004/02-p0-p1-final.md` P1-2

## 边界

只补 DepthAnything 的无权重假 tensor 后处理契约测试。不改 `postprocess` 实现，不改 MNN/ORT/TRT，不做 P1-3，不测 Metric3D。一 catalog 一进程不变。测试在 WSL。

## 做法

### 推荐

仿 `test/enlightengan_output_contract_unittest.cc`：新建 `test/mono_depth_output_contract_unittest.cc`，子类暴露 `DepthAnything::postprocess`，喂空 outputs 和错误 shape 的假 tensor。不 init、不跑 session、不读权重、不 `GTEST_SKIP`。

在 `test/CMakeLists.txt` 的 `MORTRED_BUILD_FULL` 块里，与 enlightengan 同级登记 `mono_depth_output_contract_unittest`。默认 ctest 在 FULL 构建里会跑到它。`-R` 匹配 0 条不算通过。

### 否决

只测 helper 不调用 `DepthAnything::postprocess`。为了塞进 tests-only 去改 models 链接。

## 改动面

- 新增 `test/mono_depth_output_contract_unittest.cc`
- `test/CMakeLists.txt` 登记该 target

## 明确不改

`src/models/mono_depth_estimation/**`、backend、conf、P1-3。

## 验收标准

1. **AC1** 在已构建该 target 的 FULL 目录上，`scripts/run_tests.sh <BUILD> -R 'mono_depth_output_contract'` 退出码 0，且 ctest 实际执行了该测试（不是 0 tests）。
2. **AC2** 空 outputs 断言状态非 OK，且日志或捕获文本含 `depth anything`。
3. **AC3** 错误 rank/shape 的假 tensor 断言状态非 OK，且文本含 `depth anything` 或 `shape`。
4. **AC4** 源码无 `GTEST_SKIP`，不读权重文件。
5. **AC5** `rg -n mono_depth_output_contract_unittest test/CMakeLists.txt` 退出码 0。

## 自测命令

```text
scripts/run_tests.sh <BUILD> -R 'mono_depth_output_contract'
rg -n mono_depth_output_contract_unittest test/CMakeLists.txt
```

期望退出码都是 0。`<BUILD>` 必须是 `MORTRED_BUILD_FULL=ON` 且已编出该 target 的目录。

## RD 不得自行决定

不得改 postprocess。不得 SKIP。不得用匹配 0 条冒充通过。ctest 名必须是 `mono_depth_output_contract_unittest`。
