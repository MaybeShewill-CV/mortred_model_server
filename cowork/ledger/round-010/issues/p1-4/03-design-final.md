# 技术方案

- 问题: P1-4
- 版本: v1（编排方决策版）
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md P1-4

## 边界

只处理 `src/models/backend/tensor.h` 的 3 处 `CHECK`/`CHECK_EQ`（`make` → `:239`、`data<T>()` → `:310`/`:316`）。不改 `invoke_nothrow` 那道边界，不改各 session 的既有校验，不改 `Tensor` 的内存布局与零拷贝字段。

## 做法

### 推荐

把三处"断言即 abort"改为**抛出 `std::invalid_argument`**，由 `src/models/base_model.h:28-39` 既有的 `invoke_nothrow` 栅栏接住并映射为 `MODEL_RUN_SESSION_FAILED`（HTTP 500）：

- `Tensor::make`：`checked_shape_nbytes` 失败时 `throw std::invalid_argument("tensor shape must be concrete, non-empty, and fit in size_t: <shape>")`。
- `data<T>()` 与 `data<T>() const`：dtype 不匹配时 `throw std::invalid_argument("tensor dtype is <actual>, requested <T>")`。
- 在 `tensor.h` 顶部注释写明：这是**唯一**允许以异常表达契约违规的地方，理由是该头被 `.inl` 解码路径广泛使用，逐点返回状态码需改动 30+ 个调用点且会引入静默空指针风险；栅栏已存在且被文档化为"把异常挡在模型边界"。

已核验的前提（编排方逐行确认）：`invoke_nothrow` 捕获 `std::exception` 与 `...` 并返回 `MODEL_RUN_SESSION_FAILED`；`Tensor::make`/`data<T>()` 到该栅栏之间**没有** `noexcept`（全仓 `noexcept` 仅出现在 TRT logger 回调与 `DeviceBuffer` 移动赋值）。

### 否决

- **log + 返回空 buffer / nullptr（技术负责人线原推荐）**：`invoke_nothrow` 兜不住空指针解引用，需要给 12 个 `data<T>()` 生产调用点逐个补判空（`sam_vit_encoder.cpp`、`sam_prompt_decoder.cpp`、`sam_amg_decoder.cpp`、`msocrnet.inl`、`hrnet_segmentation.inl`、`superpoint.inl`、`lightglue.inl`、`ddpm_unet.inl`、`cls_cond_ddpm_unet.inl`）。漏改一处就从"可诊断的 abort"变成"不可诊断的段错误"，且这些模型在本机缺权重/GPU，无法全部验证——风险高于收益。
- 给 `make` 增加返回 `optional/Status` 的新 API 并迁移 24 个调用点：改动面最大。
- 保留 abort、只在会话层加契约：覆盖不到 `.inl` 解码路径。

## 改动面

- `src/models/backend/tensor.h`（3 处 + 头注释）
- `test/tensor_contract_unittest.cc`（新增：坏 shape 的 `make` 抛异常且进程存活；dtype 不匹配的 `data<T>()` 抛异常；`from_mat` 既有约定不变）
- 如需模型级证据：`test/model_output_contract_unittest.cc` 增加一条"坏形状输出 → `MODEL_OUTPUT_CONTRACT_FAILED` → envelope 5xx"

## 明确不改

- `invoke_nothrow`、状态码映射（`http_status.h` 里 `MODEL_OUTPUT_CONTRACT_FAILED` → 500 保持）
- `Tensor` 的 `device_data`/`device_ready_event` 零拷贝语义与默认拷贝/移动
- 各 session、`make` 的 24 个调用点签名、packing/max_batch_size

## 契约

- 形状或 dtype 不符**不再 abort 进程**；表现为 `MODEL_RUN_SESSION_FAILED`（HTTP 500）或调用点既有状态码；同 id 其它在途请求不受影响。

## 性能与资源

成功路径只把 `CHECK` 换成 `if`，失败分支多一次构造异常对象；无分配、无锁，对 TRT static shape batch=1 热路径无可测影响。

## 验收标准

- A1 `./_bin/tensor_contract_unittest` 退出 0，且含：非 concrete/0 维/溢出 shape 的 `make` 抛异常、F32 上取 `data<int32_t>()` 抛异常、用例在同进程内跑完后续断言（证明未 abort）。
- A2 `grep -rn 'CHECK(checked_shape_nbytes\|CHECK_EQ(dtype' src/` 无输出（`echo $?` = 1）。
- A3 `./_bin/model_output_contract_unittest` 与 `./_bin/http_contract_test` 退出 0（500 映射未变）。
- A4 `./_bin/object_detection_output_contract_unittest`、`./_bin/backend_unittest` 退出 0（调用点未破坏）。
- A5 提交只含 `tensor.h` 与测试文件；不含 `cowork/ledger/`、构建产物、`src/common/mortred_version.h`。

## 测试义务

- 上述 5 条，给出真实命令与退出码。
- 白盒：确认没有调用点依赖"abort 语义"（例如 try/catch 之外的 `noexcept` 包装）。
- 反例：`EXPECT_THROW(..., std::invalid_argument)` 必须精确匹配异常类型，不允许 `EXPECT_ANY_THROW` 这种弱断言。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r10b
cmake --build build --target tensor_contract_unittest model_output_contract_unittest -j8   # 期望 0
./_bin/tensor_contract_unittest          ; echo $?   # 期望 0
./_bin/model_output_contract_unittest    ; echo $?   # 期望 0
grep -rn 'CHECK(checked_shape_nbytes\|CHECK_EQ(dtype' src/ ; echo $?   # 期望 1
```

## RD 不得自行决定

- 是否额外提供不抛的 `try_*` 变体；异常类型（须为 `std::invalid_argument`，不得用 `std::runtime_error`）；是否把 `data<T>()` 标 `[[nodiscard]]`。

## 实施后的事实更正（编排方，第 5 步 QA 实测）

- 本文件原先写 "`Tensor::make` 24 个调用点 / `data<T>()` 12 个调用点"。QA 实测：`make` = **24**（正确），`data<T>()` = **23** 处（`.data<T>()` 与 `.template data<T>()` 合计，`src/` 内排除 `tensor.h`，分布在本文点名的同 9 个文件：lightglue 8、sam_prompt_decoder 3、cls_cond_ddpm_unet 2、ddpm_unet 2、msocrnet 2、sam_amg_decoder 2、superpoint 2、hrnet_segmentation 1、sam_vit_encoder 1）。数字更正只加强"否决 log + nullptr 方案"的理由（漏判空的代价更大），不改变验收结论。
- 本文「自测命令」里的 `cd /mnt/g/Codex/mortred_wt_r10b` 是复制笔误，本项实际在 `/mnt/g/Codex/mortred_wt_r10d` 执行。
- QA 另发现 `packed_batch_nvi_unittest` 因 `link.txt` 只链 `libcommon.so`（符号在 `libmodels.so`）而在两个构建目录都无法链接，故 `BackendCvModel::run_batch` 那道 packed 栅栏只有代码走查；该二进制在本次操作前即不存在，与本项无关。已记入 `discovered.md` 与 `05-qa.md` 残余风险。
