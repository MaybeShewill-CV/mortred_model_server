# 技术方案

- 问题: P1-1 AttentiveGan / HRNet 在 shape/体积不符时仍回 MODEL_EMPTY_OUTPUT
- 版本: v1
- 对应列表: `cowork/ledger/round-007/02-p0-p1-final.md`

## 边界

只改这两处误用的返回码，并加 FULL 契约单测。空 outputs 仍是 `MODEL_EMPTY_OUTPUT`。已是 `MODEL_OUTPUT_CONTRACT_FAILED` 的路径不改语义。不新增 StatusCode。不改 RealESRGAN、drain、进程模型、HTTP。

## 做法

### 推荐

- `attentive_gan_derain_net.inl` 约 113–114 与 126–129 行改为 `MODEL_OUTPUT_CONTRACT_FAILED`。约 76–78 行空 outputs 不动。
- `hrnet_segmentation.inl` 约 100–102 行 mask 小于输入图改为 `MODEL_OUTPUT_CONTRACT_FAILED`。空 outputs 不动。
- 两头文件的 `postprocess` 现为 private。只升到 protected，供子类 `using`。不得改成 public。
- 新增 `test/attentive_gan_output_contract_unittest.cc` 与 `test/hrnet_output_contract_unittest.cc`，挂进 `MORTRED_BUILD_FULL`，并加 ctest 名 `AttentiveGanOutputContract`、`HRNetOutputContract`。

### 否决

只改代码不挂 ctest。顺手改 RealESRGAN 或其他家族。新增 StatusCode。

## 改动面

- `src/models/enhancement/attentive_gan_derain_net.inl`
- `src/models/enhancement/attentive_gan_derain_net.h`
- `src/models/scene_segmentation/hrnet_segmentation.inl`
- `src/models/scene_segmentation/hrnet_segmentation.h`
- `test/attentive_gan_output_contract_unittest.cc`
- `test/hrnet_output_contract_unittest.cc`
- `test/CMakeLists.txt`

## 明确不改

RealESRGAN、drain、HTTP、`status_code.h`、其他模型。

## 契约

空 outputs → `MODEL_EMPTY_OUTPUT`。shape/体积/HWC 转换失败/mask 过小 → `MODEL_OUTPUT_CONTRACT_FAILED`。

## 性能与资源

无额外进程或 engine。无权重、不启 gateway。

## 验收标准

1. **AC1** AttentiveGan 空 outputs 返回 `MODEL_EMPTY_OUTPUT`。
2. **AC2** AttentiveGan shape/体积或短 buffer 返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 EMPTY。
3. **AC3** AttentiveGan 的 HWC 转换失败或体积不符分支返回 `MODEL_OUTPUT_CONTRACT_FAILED`。
4. **AC4** HRNet 空 outputs 返回 `MODEL_EMPTY_OUTPUT`。
5. **AC5** HRNet mask 元素数小于 `network_size.area()` 返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 EMPTY。
6. **AC6** HRNet 已有 rank/dtype 契约失败仍是 `MODEL_OUTPUT_CONTRACT_FAILED`。
7. **AC7** 不新增 StatusCode。
8. **AC8** 不改 RealESRGAN、drain、HTTP。
9. **AC9** FULL 上 `ctest -R 'AttentiveGanOutputContract|HRNetOutputContract'` 匹配不少于 1 条且退出码 0。匹配 0 条失败。稳定 gtest 名：`AttentiveGanOutputContract.EmptyOutputsStayModelEmptyOutput`、`AttentiveGanOutputContract.ShapeOrVolumeMismatchReturnsOutputContractFailed`、`HRNetOutputContract.EmptyOutputsStayModelEmptyOutput`、`HRNetOutputContract.MaskSmallerThanNetworkReturnsOutputContractFailed`。

## 测试义务

WSL、`cmake --preset full`、目录 `build/full`。不加载权重。

## 自测命令

```bash
cmake --preset full
cmake --build --preset full --target attentive_gan_output_contract_unittest --target hrnet_output_contract_unittest
ctest --test-dir build/full -R 'AttentiveGanOutputContract|HRNetOutputContract' --output-on-failure
```

期望退出码 0，且实际跑到测试。

## RD 不得自行决定

不得把空 outputs 改成契约失败。不得新增 StatusCode。不得把 `postprocess` 改成 public。不得改 RealESRGAN。不得用匹配 0 条冒充通过。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | `cowork/ledger/round-007/issues/p1-1/03-discussion/v1-sign-pm.md` |
| cv-rd | 同意 | 无 | v1 | `cowork/ledger/round-007/issues/p1-1/03-discussion/v1-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v1 | `cowork/ledger/round-007/issues/p1-1/03-discussion/v1-sign-tech-lead.md` |
