# 第2步表决

- 轮次: 6
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| C1 | 保留 | `real_esrgan.inl` 在 shape/`element_count` 不符时与空 tensor 同回 `MODEL_EMPTY_OUTPUT`，换导出权重后运维只看状态码会分不清；单文件改码即可交付，不是已关闭项改写。 |
| C2 | 删除 | `resnet.inl` 日志写「classification model」属措辞；一 catalog 一进程下配置/进程已点名家族，不挡交付、也不构成换机易错手工。 |
| C3 | 改优先级 | 三家 `outputs.empty()` 已 fail-closed；`model_output_contract_unittest` 已有 modnet/bisenet 无权重 shape 契约，缺口只是空输出 ctest，达不到 P1「换机必做易错手工」；补测可交付但不重开 DepthAnything，应降附录而非进实现队列。 |
| C4 | 删除 | drain 已有 `kDrainTimeoutMs` 截止属已关闭项；超时后继续析构并靠外层 SIGKILL 是现行一进程一 id 停服约定，再改等于推翻进程模型，不能为凑分保留。 |

## 不同意的条目
- C4：不同意把「超时后仍析构」写成新 P1（改写已关闭的 drain 截止与 SIGKILL 模型）。
- C3：不同意按「三家全无无权重契约」保留为 P1（shape 契约已有；仅空输出 ctest 应降级）。
