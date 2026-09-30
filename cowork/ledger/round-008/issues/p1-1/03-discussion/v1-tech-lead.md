# 技术方案

- 问题: P1-1 lightglue 与 openai clip 非空故障错成 MODEL_EMPTY_OUTPUT
- 版本: v1
- 对应列表: `cowork/ledger/round-008/02-p0-p1-final.md`

## 边界

同一条收完 lightglue 与 openai clip。dtype 不符、尺寸不符、embedding 形状不兼容改为 `MODEL_OUTPUT_CONTRACT_FAILED`。lightglue 输出名缺失、clip 空文本保持 `MODEL_EMPTY_OUTPUT`。不新增 StatusCode。不加 OCR。不做 P1-2/P1-3/P1-4。不改进程模型。

## 做法

### 推荐

- lightglue：输出名缺失保持 EMPTY；dtype 与相对尺寸不符改为 `MODEL_OUTPUT_CONTRACT_FAILED`。契约判定入口只升到 protected，不得 public。
- openai clip：空文本保持 EMPTY；`embeddings_compatible` 失败改为 `MODEL_OUTPUT_CONTRACT_FAILED`。承载该映射的入口只升到 protected。
- 新增 FULL 单测，ctest 别名 `LightGlueOutputContract` 与 `OpenAiClipOutputContract`。不加载权重。

### 否决

只改 `postprocess` 桩。把入口改成 public。新增 StatusCode。只做其中一个家族。

## 改动面

- `src/models/feature_point/lightglue.h`、`lightglue.inl`
- `src/models/clip/openai_clip.h`、`openai_clip.inl`
- `test/lightglue_output_contract_unittest.cc`
- `test/openai_clip_output_contract_unittest.cc`
- `test/CMakeLists.txt`

## 明确不改

其他家族、`status_code.h`、产品 toml、`conf/`、HTTP、进程模型。clip 的 `normalize_embedding` 失败和 softmax 非有限分支保持原码。

## 契约

非空 dtype/尺寸/embedding 形状失败 → `MODEL_OUTPUT_CONTRACT_FAILED`。输出缺失与空文本 → `MODEL_EMPTY_OUTPUT`。

## 性能与资源

无额外进程或 engine。测试不跑 session。

## 验收标准

1. **AC1** lightglue 缺 keypoints/scores/descriptors（matcher 缺 matches0/mscores0）返回 `MODEL_EMPTY_OUTPUT`。
2. **AC2** lightglue 输出齐全但 dtype 不符返回 `MODEL_OUTPUT_CONTRACT_FAILED`。
3. **AC3** lightglue 相对尺寸不符返回 `MODEL_OUTPUT_CONTRACT_FAILED`。
4. **AC4** clip 空文本返回 `MODEL_EMPTY_OUTPUT`。
5. **AC5** clip embedding 形状不兼容返回 `MODEL_OUTPUT_CONTRACT_FAILED`。
6. **AC6** `build/full` 上 `ctest -R '^(LightGlueOutputContract|OpenAiClipOutputContract)$'` 匹配不少于 1 条且退出码 0。匹配 0 条失败。
7. **AC7** 不改其他家族，不新增 StatusCode，相关符号最多 protected。

## 测试义务

WSL FULL，目录 `build/full`。不加载权重。

## 自测命令

```bash
scripts/run_tests.sh build/full -R '^(LightGlueOutputContract|OpenAiClipOutputContract)$' --output-on-failure
```

期望退出码 0，且实际跑到测试。

## RD 不得自行决定

不得只收一个家族。不得把空文本或输出缺失改成契约失败。不得把入口改成 public。不得做另外三条 P1。
