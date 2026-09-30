# P0 与 P1

- 轮次: 6
- 版本: v2
- 综合分上限: 有 P1 时不超过 8.9

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | RealESRGAN 输出 shape/体积不符时与空 tensor 共用 `MODEL_EMPTY_OUTPUT`，排障分不清 | 客户换导出权重或后端改 layout 后增强接口失败，运维只看 HTTP/状态码 | `validated_f32_first_output` 对契约失败会回 `MODEL_OUTPUT_CONTRACT_FAILED`；随后手工 shape/`element_count` 检查仍回 `MODEL_EMPTY_OUTPUT`，与空输出同码 | 错误 shape/体积与空输出返回不同状态码（或契约失败码），日志能直接区分两类故障 | `src/models/enhancement/real_esrgan.inl`（约 74–96 行）；`src/models/backend/f32_output.h`；`src/common/status_code.h` |

## 附录
无

## 未采纳
| id | 来源表决 | 原因 |
| --- | --- | --- |
| P1-2 | 技术负责人 C4 删除 | 已关闭 drain/`kDrainTimeoutMs` 与外层 SIGKILL 约定的改写；再改等于动一进程一 catalog 停服进程模型，不能为凑分保留 |
| P1-3 | PM C3 删除；技术负责人 C3 改优先级 | 双方不同意按「三家都没有无权重契约」开单；modnet/bisenet 已有 shape 契约，仅空输出 ctest 缺口达不到 P1，且不得借壳重开 DepthAnything |
| cand-log-family | 双方 C2 删除 | 分类日志写「classification model」属措辞；一 catalog 一进程下进程/配置已点名家族，本轮不升 P1 |

## 签字
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| cv-pm | 同意 | v2 | `cowork/ledger/round-006/02-discussion/v2-sign-pm.md` |
| cv-tech-lead | 同意 | v2 | `cowork/ledger/round-006/02-discussion/v2-sign-tech-lead.md` |
| cv-cto | 同意 | v2 | `cowork/ledger/round-006/02-discussion/v2-sign-cto.md` |
