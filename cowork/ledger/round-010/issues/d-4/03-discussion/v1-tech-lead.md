# 技术方案

- 问题: D-4 LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵
- 版本: v1
- 对应列表: `cowork/ledger/round-010/02-p0-p1-final.md` 版本 v2 的 D-4

## 边界

本条没有上一版方案，没有待写入的审查条件。

同级标准是检测族已经落地、且不依赖权重的两则：`test/object_detection_output_contract_unittest.cc` 的 `YoloV8RejectsMalformedOutput` 对错 dtype（标本为 `I32`）和 `quiet_NaN` 断言 `MODEL_OUTPUT_CONTRACT_FAILED`；`YoloV8RejectsInfinity` 对 `+infinity` 断言同一状态码，且不是 `MODEL_EMPTY_OUTPUT`。`validated_f32_named_output` 默认拒绝非有限 f32。`test/model_contract_test_util.h` 的七格宏只挂在 RTDETR 脚手架上，并接受 `MODEL_NOT_IMPLEMENTED`，不作为本条的通过标准。

现在 LightGlue 的 `classify_extractor_outputs` 只有 keypoints 错 dtype 和描述子元素个数不符两条代表性契约失败；matcher 没有 dtype 用例。`scores`、`descriptors`、`mscores0` 上的 NaN 与 `+Inf` 不会在分类入口被拒绝。CLIP 读视觉塔和文本塔的 `"output"` 时未要求有限值；非有限 embedding 会进到 `normalize_embedding`，非有限范数回 `MODEL_EMPTY_OUTPUT`。现有 CLIP 用例只有空文本和两条 embedding 长度不兼容。

本条只让这两族下面列出的 dtype / `quiet_NaN` / `+infinity` 格子达到上述两则的可观察结果。缺输出名和空文本仍是 `MODEL_EMPTY_OUTPUT`。

不涉及 D-5、D-6，以及已关闭的 P0-1、P1-1 至 P1-6、D-1、D-2、D-3。

## 做法

### 推荐

在现有分类入口上补齐矩阵，测试继续注入假 tensor，不加载权重，不跑 session。

LightGlue 的非有限检查写进已经由 `extract_feature_points`、`match_feature_points` 调用的 `classify_extractor_outputs` 与 `classify_matcher_outputs`，并且发生在分数阈值比较之前。f32 张量是 `scores`、`descriptors`、`mscores0`。i32 张量只补尚未覆盖的 dtype：`matches0` 应为 `I32`。keypoints 的错 dtype 保持现有用例。

CLIP 增加一个 protected 分类入口，`encode_image` 与 `encode_text` 在对应 session 返回成功之后、把数据写入 embedding 之前、调用 `normalize_embedding` 之前，都走这一个入口。入口检查名为 `"output"` 的张量：`F32`、秩 2、形状 `[1, D]`（`D > 0`）。错 dtype、任一元素 `quiet_NaN`、任一元素 `+infinity` 都回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不写入 embedding。有限且合约的张量回 `OK`，embedding 等于读出的 f32，不做归一化。入口不要求模型已经 init，也不打开 session。

新用例放进现有套件 `LightGlueOutputContract` 与 `OpenAiClipOutputContract`，沿用已有 CTest 别名。每个格子一个 `TEST`。

代价：原先会被 LightGlue 收进特征点或匹配的 NaN 与 `+Inf` 改为整段契约失败。原先在 CLIP 归一化阶段变成空输出的 NaN 与 `+Inf` 改为契约失败。CLIP 多一个 protected 入口。无新进程、无新权重、无 engine、无额外显存。

### 否决

套 `POSTPROCESS_CONTRACT_TEST` 的七格（缺输出、错 dtype、错秩、错形状、短 buffer、NaN、Inf），或去改这两族现在固定返回 `MODEL_RUN_SESSION_FAILED` 的 `postprocess`。代价是缺输出名会被改成契约失败，和已经关闭的空输出错码冲突；宏对 `MODEL_NOT_IMPLEMENTED` 也算拒绝，RTDETR 脚手架会显得“同级通过”。秩、形状、短 buffer 不在本条冻结句里。

只加测试、不改拒绝，或只给一个代表张量补 NaN。代价是 `descriptors`、`mscores0` 或 CLIP 的读出仍会收下 NaN/`+Inf`，CLIP 仍把它们报成 `MODEL_EMPTY_OUTPUT`。代表性错码还在，完整矩阵跑不通。

## 改动面

- `src/models/feature_point/lightglue.inl`：两个已有 classify 函数。
- `src/models/clip/openai_clip.h`、`src/models/clip/openai_clip.inl`：一个 protected 分类入口，以及视觉塔、文本塔两处调用。
- `test/lightglue_output_contract_unittest.cc`
- `test/openai_clip_output_contract_unittest.cc`

`lightglue.h` 的现有 protected 声明保持。不新增测试二进制，不改 `test/CMakeLists.txt`。

## 明确不改

- 产品 toml 的 `type`，以及 `type=mnn` / `type=tensorrt` 约定。
- `status_code.h`。不新增状态码。
- 分类入口的可见性：保持 protected，不改为 public。
- LightGlue 缺 `keypoints` / `scores` / `descriptors`，以及缺 `matches0` / `mscores0`，仍为 `MODEL_EMPTY_OUTPUT`。
- CLIP 空文本仍为 `MODEL_EMPTY_OUTPUT`，且不是 `MODEL_OUTPUT_CONTRACT_FAILED`。两条 embedding 长度不兼容仍为 `MODEL_OUTPUT_CONTRACT_FAILED`。
- `normalize_embedding` 对有限但范数过小（含全零）的向量仍返回失败，调用方仍为 `MODEL_EMPTY_OUTPUT`。softmax 的非有限分支保持原码。
- `classify_feature_layout` 的现有尺寸判定。
- 检测族源码与 `test/object_detection_output_contract_unittest.cc`。
- `test/model_contract_test_util.h` 与 RTDETR 契约测试。
- 其他模型家族、HTTP、OCR、TRT 数值门、新架构接入、gateway `:8080`、supervisor `:8787`、一 catalog 一进程、engine 生成。

## 契约

无新的 HTTP 路径、请求体、状态码或 toml 键。tensor 布局不变。

合法对照样本，其余格子只改注明的那一处：

- LightGlue extractor：1 个点。`keypoints` 为 `I32`、2 个元素；`scores` 为 `F32`、1 个元素且值为 `1`；`descriptors` 为 `F32`、256 个元素且全部有限。
- LightGlue matcher：`matches0` 为 `I32`、2 个元素；`mscores0` 为 `F32`、1 个元素且值为 `1`。
- CLIP：名为 `"output"`，`F32`，形状 `[1, 4]`，值为 `2, 0, 0, 0`。

错 dtype 的标本：期望 `F32` 的张量改成 `I32`；期望 `I32` 的 `matches0` 改成 `F32`。元素个数仍与上表一致。NaN 标本是该张量一个元素的 `quiet_NaN`，同张量其余元素保持有限。Inf 标本是该张量一个元素的 `+infinity`，同张量其余元素保持有限。

| 格子 | 注入 | 状态码 |
| --- | --- | --- |
| extractor `scores` 错 dtype | `scores` 为 `I32` | `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT` |
| extractor `descriptors` 错 dtype | `descriptors` 为 `I32` | 同上 |
| extractor `scores` NaN | `scores[0] = quiet_NaN` | 同上 |
| extractor `descriptors` NaN | `descriptors[0] = quiet_NaN` | 同上 |
| extractor `scores` +Inf | `scores[0] = +infinity` | 同上 |
| extractor `descriptors` +Inf | `descriptors[0] = +infinity` | 同上 |
| matcher `matches0` 错 dtype | `matches0` 为 `F32` | 同上 |
| matcher `mscores0` 错 dtype | `mscores0` 为 `I32` | 同上 |
| matcher `mscores0` NaN | `mscores0[0] = quiet_NaN` | 同上 |
| matcher `mscores0` +Inf | `mscores0[0] = +infinity` | 同上 |
| CLIP 有限对照 | `[1, 4]` 值为 `2, 0, 0, 0` | `OK`；embedding 长度为 4，首元素为 `2` |
| CLIP 错 dtype | `"output"` 为 `I32`，形状 `[1, 4]` | `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`；embedding 长度为 0 |
| CLIP NaN | `output[1] = quiet_NaN` | 同上 |
| CLIP +Inf | `output[1] = +infinity` | 同上 |

i32 的 `keypoints` 与 `matches0` 不做 NaN/Inf 格子。keypoints 错 dtype 仍由已有用例覆盖。

## 性能与资源

无新常驻进程，不增加 catalog id。不生成 engine，不占额外显存。有限性检查只扫描已经在宿主内存里的 f32 输出，不复制第二份权重。cpu profile 不因此链接 TensorRT。测试不启动 gateway 或 supervisor。

## 验收标准

下列名字都在套件 `LightGlueOutputContract` 或 `OpenAiClipOutputContract` 里。判定方式是同一次 `-V` 日志出现对应的 `[       OK ]` 行。出现 `SKIPPED`，或该名字没有执行，则该条不通过。

1. **AC1** 已有七个用例仍为 OK：`MissingExtractorOutputsStayEmpty`、`ExtractorDtypeMismatchReturnsContractFailed`、`ExtractorSizeMismatchReturnsContractFailed`、`MissingMatcherOutputsStayEmpty`、`FeatureLayoutMismatchReturnsContractFailed`、`EmptyTextsStayModelEmptyOutput`、`IncompatibleEmbeddingReturnsContractFailed`。
2. **AC2** `ExtractorScoresWrongDtype` 为 OK，对应上表 extractor `scores` 错 dtype。
3. **AC3** `ExtractorDescriptorsWrongDtype` 为 OK。
4. **AC4** `ExtractorScoresNan` 为 OK。
5. **AC5** `ExtractorDescriptorsNan` 为 OK。
6. **AC6** `ExtractorScoresPositiveInf` 为 OK。
7. **AC7** `ExtractorDescriptorsPositiveInf` 为 OK。
8. **AC8** `MatcherMatchesWrongDtype` 为 OK。
9. **AC9** `MatcherScoresWrongDtype` 为 OK。
10. **AC10** `MatcherScoresNan` 为 OK。
11. **AC11** `MatcherScoresPositiveInf` 为 OK。
12. **AC12** `EmbeddingFiniteAccepted` 为 OK，对应 CLIP 有限对照。
13. **AC13** `EmbeddingWrongDtype` 为 OK。
14. **AC14** `EmbeddingNan` 为 OK。
15. **AC15** `EmbeddingPositiveInf` 为 OK。
16. **AC16** 自测命令退出码为 0。输出里有通过的测试，通过数大于 0。出现 `No tests were found` 则本条不通过。

## 测试义务

在已经配置好的 FULL 构建目录 `build/full` 上跑。不要求 GPU，不加载权重，不生成 engine，不启动 gateway 或 supervisor。

白盒核对：

- LightGlue 的非有限拒绝写在 `classify_extractor_outputs` 与 `classify_matcher_outputs` 内，位于阈值比较之前。两条生产路径仍先调用这两个函数。
- CLIP 的 `encode_image` 与 `encode_text` 在 session 成功返回后调用同一个分类入口，且在 `normalize_embedding` 之前。失败时 embedding 保持为空。有限全零向量仍走原来的空输出，不在本条改成契约失败。
- AC1 的七个 `TEST` 里，缺名与空文本的期望仍是 `MODEL_EMPTY_OUTPUT`，其余三条 LightGlue 与 embedding 长度不兼容的期望仍是 `MODEL_OUTPUT_CONTRACT_FAILED`。这些断言没有改松。
- 本条通过不作为 D-5、D-6 或其他已关闭 id 的关闭证据。

## 自测命令

```bash
scripts/run_tests.sh build/full -R '^(LightGlueOutputContract|OpenAiClipOutputContract)$' --output-on-failure -V
```

期望退出码 0。`-V` 日志里 AC1 至 AC15 的每个用例名都有 `[       OK ]`。日志中没有这些用例的 `SKIPPED`。输出不是 `No tests were found`。

## RD 不得自行决定

- 不得只做 LightGlue 或只做 CLIP，也不得只给一个 f32 张量补 NaN。
- 不得把缺名输出或空文本改成 `MODEL_OUTPUT_CONTRACT_FAILED`。
- 不得把有限全零 embedding 改成契约失败。
- 不得把分类入口改成 public，不得新增状态码，不得改产品 toml 的 `type`。
- 不得用 `GTEST_SKIP`、缺权重跳过或未执行的过滤冒充通过。
- NaN 标本必须是 `quiet_NaN`，Inf 标本必须是 `+infinity`。不得用一个很大的有限数代替 Inf。
- 不得把秩、形状、短 buffer、HTTP OCR、其他家族的 TRT 数值门，或新架构零 C++，写进本次改动。
- CLIP 分类入口不得要求已初始化的 session 或权重。失败时不得留下非空 embedding。
