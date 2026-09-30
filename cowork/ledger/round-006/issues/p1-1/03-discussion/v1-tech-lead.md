# 技术方案

- 问题: P1-1 RealESRGAN 输出 shape/体积不符时与空 tensor 共用 `MODEL_EMPTY_OUTPUT`，排障分不清
- 版本: v1
- 对应列表: `cowork/ledger/round-006/02-p0-p1-final.md`（v2，仅 P1-1）

## 边界

- 只动 RealESRGAN `postprocess` 在契约通过后仍把 shape/体积失败映射成 `MODEL_EMPTY_OUTPUT` 的路径，以及相应白盒契约单测与 FULL 测试挂载。
- 空 `outputs`（约 74–76 行）继续 `MODEL_EMPTY_OUTPUT`，日志继续含 `real esrgan output tensor is empty`；语义不得改。
- `validated_f32_first_output` / `f32_output.h` / `status_code.h` 既有语义不动：契约失败已是 `MODEL_OUTPUT_CONTRACT_FAILED`（文案 `model output contract failed`）。
- 不改 MNN / ORT / TRT；不改一 catalog 一进程、gateway/supervisor；不做未采纳的 drain / 三家空输出测试；不新开 StatusCode；不改 HTTP 映射（两类失败今天都是 500，区分靠 StatusCode 与日志）。

## 做法

### 推荐

删掉契约已覆盖的二次 rank/channel 检查；若保留几何/体积兜底，一律改为 `MODEL_OUTPUT_CONTRACT_FAILED` + 非 empty 文案日志；空 outputs 路径原样保留。新增 FULL 链接的 `realesrgan_output_contract_unittest`，用子类暴露已有 `protected` `postprocess`（头文件已是 protected，禁止为测试改访问级别）。

代价：改动面小（`.inl` + 一个单测 + `test/CMakeLists.txt` 挂载）；无新状态码、无进程/显存/engine 副作用；契约失败路径多数已由 `{1,3,-1,-1}` NCHW 契约挡住，落地主要是清掉错码与补可 `-R` 的黑盒用例。

### 否决

1. 只改约 87–95 行的 return，不补专用契约单测 / 不挂 FULL `TEST_LIST`。WSL 上无法用稳定 `ctest -R` 黑盒验收；匹配 0 条不算通过。
2. 改 `f32_output.h`、扩 StatusCode、或改 HTTP 层分流两类 500。超出本条。

## 改动面

- `src/models/enhancement/real_esrgan.inl`：约 87–89 行删除或改为 `MODEL_OUTPUT_CONTRACT_FAILED`；约 93–95 行体积不符改为 `MODEL_OUTPUT_CONTRACT_FAILED`，日志不得写成 empty。74–76 行不改。
- 新建 `test/realesrgan_output_contract_unittest.cc`。
- `test/CMakeLists.txt`：`MORTRED_BUILD_FULL` 下加入 `TEST_LIST`，链接与其他 `*_output_contract_unittest` 一致。

## 明确不改

- `real_esrgan.h` 访问级别。
- `f32_output.h`、`tensor_contract.h`、`status_code.h`、`http_status.h`。
- 预处理、产品 toml、其他 enhancement 家族、后端、drain、三家空输出测试。

## 契约

- 空 `outputs` → `MODEL_EMPTY_OUTPUT`。
- 契约失败与手工 shape/体积拒绝 → `MODEL_OUTPUT_CONTRACT_FAILED`。
- HTTP 仍可同为 500。布局仍是 NCHW `[1,3,H,W]` f32。

## 性能与资源

无常驻进程、显存或 engine 变更。成功路径解码不变。

## 验收标准

1. **AC-1** 空 outputs 返回 `MODEL_EMPTY_OUTPUT`，日志含 `real esrgan output tensor is empty`。
2. **AC-2** shape `{1,4,H,W}` 返回 `MODEL_OUTPUT_CONTRACT_FAILED`，不得为 `MODEL_EMPTY_OUTPUT`。
3. **AC-3** rank 不是 4 返回 `MODEL_OUTPUT_CONTRACT_FAILED`，不得为 `MODEL_EMPTY_OUTPUT`。
4. **AC-4** 声明 `{1,3,H,W}` 但 buffer 偏短，返回 `MODEL_OUTPUT_CONTRACT_FAILED`，不得为 `MODEL_EMPTY_OUTPUT`。
5. **AC-5** 合法 `{1,3,H,W}` 有限 f32 返回 `OK`，结果图尺寸 `(W,H)`、`CV_8UC3`。
6. **AC-6** FULL 构建 `ctest -R 'RealEsrGanOutputContract'` 执行条数不少于 1 且通过。匹配 0 条失败。

## 测试义务

稳定用例名：

- `RealEsrGanOutputContract.EmptyOutputsStayModelEmptyOutput`
- `RealEsrGanOutputContract.WrongChannelReturnsOutputContractFailed`
- `RealEsrGanOutputContract.WrongRankReturnsOutputContractFailed`
- `RealEsrGanOutputContract.ShortBufferReturnsOutputContractFailed`
- `RealEsrGanOutputContract.ValidNchwReturnsOk`

WSL、FULL 构建，不加载权重，不启 gateway。

## 自测命令

```bash
cmake --preset full && cmake --build --preset full --target realesrgan_output_contract_unittest
ctest --test-dir build/full -R 'RealEsrGanOutputContract' --output-on-failure
```

期望退出码都是 0，且 ctest 实际跑到测试。本机用 `full` 而不是 `full-cpu`。

## RD 不得自行决定

不得改空 outputs 的状态码或 empty 日志。不得新增 StatusCode。不得改 `postprocess` 访问级别。不得扩大到其他模型。用例名不得私自改。
