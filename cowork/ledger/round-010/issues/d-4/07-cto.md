# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、D-1、D-2、D-3、D-4
- 挂起: 无

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 8.4 | 无 | 无 |
| src-server-apps | 15 | 9.3 | 无 | 无 |
| src-control | 10 | 8.7 | 无 | 无 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 7.9 | 无 | 无 |
| ops | 15 | 8.6 | 无 | 无 |
| test | 15 | 7.5 | LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 | 无 |
| docs | 10 | 9.1 | 无 | D-5：deployment.md §15 FAQ 结构损坏；D-6：中文 README 路径 B 与现行 full 构建入口不一致 |
| ci | 5 | 8.0 | 无 | 无 |

## 附录
D-4 合入 `b183893685b835f4db76f80c2bbe03635136f994`。问题分支提交 `c293e9d31585561ef16e1eff89e41fa6b419ed55`。QA 结论 pass。偏离无。RD 新发现无。QA 残余风险无。关闭范围：LightGlue 的 scores、descriptors、mscores0 上的错 dtype、quiet_NaN、+infinity，以及 matches0 的错 dtype；CLIP output 上的错 dtype、quiet_NaN、+infinity。这些格子是 MODEL_OUTPUT_CONTRACT_FAILED，不是 MODEL_EMPTY_OUTPUT。缺名与空文本仍是空输出。

未改模块的分数照 `cowork/ledger/round-010/01-modules/` 九份。此前已关闭范围照 `cowork/ledger/round-010/08-writeoff.md`：P1-1 的关闭范围是已合入的可复制模板，任意新架构零 C++ 不另开。P1-5 合入 `1f267ea20c125f40e674242a5908a2583a18e331`，关闭范围是产品 yolov8s 的 TRT 检测框门，同类且 IoU ≥ 0.5；CPU 用例 ctest 名 `yolov8_detection` 仍登记 `conf/ci/yolov8_onnx_hosted.toml`；其余家族的 TRT 数值门不另开。P1-6 合入 `a9e6284bae832996c0d0c52ec39bd69da1d6fe05`，关闭范围是 `cpu=success` 且 `gpu=skipped` 时退出码 2，并打印 `NOT_A_TRT_PROOF`；本条不含 GPU/TRT 数值黑盒。D-1 合入 `8f3a0d101070737f63694024e94eab968f2c80f4`，关闭范围是 deadline 返回后未就绪 child 的 `state` 为 `failed`；deadline 分支不发信号；外层 SIGKILL 仍属不计毛病。D-2 合入 `03d604d766da55792494c5dadfc8998dbc671a76`，关闭范围是三份 attentive_gan toml 的 `model_input_image_size` 改为 `[240, 360]`，并删掉同一句 stale 注记；preprocess 仍用乘号 `240×360`；不改推理路径。D-3 合入 `f43d544810918d9d9b831e0273cf1890951e3a2b`，关闭范围是选中用例全部因缺权重跳过时，`model_golden_test` 退出码 77，标准输出恰好一行 `NUMERICAL_NOT_RUN`；单条缺权重仍可以跳过。

## 新准入
无
