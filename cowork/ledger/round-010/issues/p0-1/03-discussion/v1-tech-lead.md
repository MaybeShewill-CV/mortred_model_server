# 技术方案

- 问题: P0-1
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

- 只修 `SamVitEncoder` 与 `SamPromptDecoder` 在「已有输出、但 dtype/形状/尺寸不合约」时把错码从 `MODEL_EMPTY_OUTPUT` 改成 `MODEL_OUTPUT_CONTRACT_FAILED`；真正空输出（无输出、缺名输出）仍为 `MODEL_EMPTY_OUTPUT`。
- 不新增 `StatusCode`；不改进程模型、HTTP 响应形状、其他模型家族、AMG decoder、权重/golden。
- 本轮门禁必须是无权重契约测，能在 WSL 上经 `scripts/run_tests.sh` 跑通；`ctest -R` 靠 `add_test` 别名选中，选中 0 条算失败。

## 做法

### 推荐

在现有 encode/decode 路径上按「空 vs 非空不合约」分流错码，并抽出可对假 tensor 调用的契约分类入口（对齐 LightGlue 的 `classify_*` + Callable 子类），再加 `sam_output_contract_unittest` 与 CTest 别名 `SamOutputContract`。

代价：只动 SAM prediction 两个实现文件及其头、加一份契约测与 CMake 登记；运行时无额外推理，仅多几条分支与日志。

### 否决

1. 只改 return 码、不暴露可测入口、靠 golden/权重测。本轮要求无权重契约测。
2. 把 SAM 全面改接到 `validated_f32_*` 并重构 decode。full mask 允许 3/4 维、low-res 固定 4 维是 SAM 专用规则。
3. 新增 StatusCode 或改 HTTP 映射。约束禁止新枚举。wire 码 5 与 6 已能区分。

## 改动面

- `sam_vit_encoder.cpp`：输出向量为空 → `MODEL_EMPTY_OUTPUT`；已有输出但非 F32 或 `element_count() <= 0` → `MODEL_OUTPUT_CONTRACT_FAILED`。
- `sam_prompt_decoder.cpp`：full-mask 非法 rank（非 3 且非 4）、low-res 非法 rank（非 4）→ `MODEL_OUTPUT_CONTRACT_FAILED`；缺 `iou_predictions`，或 masks 与 low_res 皆缺，仍为 `MODEL_EMPTY_OUTPUT`。
- 为无权重测暴露与 LightGlue 同级的可调用契约入口。入口必须是生产路径实际调用的同一套判断。
- `test/sam_output_contract_unittest.cc` 与 `test/CMakeLists.txt` 的 `add_test(NAME SamOutputContract ...)`。

## 明确不改

- `status_code.h` 枚举与 wire 值；HTTP 映射；进程模型。
- 其他模型家族；`sam_amg_decoder`；FastSAM；encoder init 静态 IO；decoder `fill_input`；HTTP JSON 字段。
- 产品 toml、engine、权重、golden 数值门。

## 契约

- 真正空输出仍是 `MODEL_EMPTY_OUTPUT`（wire 5）。
- 非空不合约返回 `MODEL_OUTPUT_CONTRACT_FAILED`（wire 6）。
- HTTP 与 tensor 对外形状不改。配置不改。

## 性能与资源

无新增常驻进程、session、engine 或显存峰值。AMG 若共用 `SamVitEncoder`，encoder 错码一并纠正，但不把 AMG decoder 纳入本条验收。

## 验收标准

1. encoder 输出向量为空时，返回 `MODEL_EMPTY_OUTPUT`。
2. encoder 有输出但 dtype 不是 F32 时，返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`。
3. encoder 有 NamedTensor 但 `element_count() <= 0` 时，返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`。
4. decoder full-mask rank 非法（非 3 且非 4）时，返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`。
5. decoder low-res mask rank 不是 4 时，返回 `MODEL_OUTPUT_CONTRACT_FAILED`，且不是 `MODEL_EMPTY_OUTPUT`。
6. decoder 缺 iou，或 masks 与 low_res 皆缺时，返回仍为 `MODEL_EMPTY_OUTPUT`。
7. `ctest -R SamOutputContract` 选中条数不少于 1，且全部通过。选中 0 条视为失败。
8. 无产品权重时，`scripts/run_tests.sh <build> -R SamOutputContract --output-on-failure` 退出码为 0。

## 测试义务

QA 对照验收 1–6 确认空与非空分支分别回 wire 5 与 6，且测试调用的是生产路径同一分类逻辑。黑盒执行验收 7–8。不要求 SAM golden、GPU/TRT 或 AMG 用例。

## 自测命令

```bash
scripts/run_tests.sh build/full -R SamOutputContract --output-on-failure
```

期望退出码 0，且选中不少于 1 条。选中 0 条或非 0 退出码即未过。

## RD 不得自行决定

- 不得新增或改名 StatusCode。不得把缺输出或空输出向量改成契约失败码。
- 不得用 golden 或权重测代替无权重契约门禁。不得省略 `SamOutputContract` 的 `add_test` 别名。
- 不得改 HTTP 形状、进程模型、其他家族或 `sam_amg_decoder` 错码。
- 不得写与生产路径不一致的测试旁路。
