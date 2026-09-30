# 交付队列

- 轮次: 10
- 版本: v2
- 综合分上限: P0 上限 7.9

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | SAM 非空契约故障仍报成空输出 | 现场 SAM 权重或后端输出 dtype/形状非法时排障 | `sam_prompt_decoder.cpp` full/low-res mask shape 非法、`sam_vit_encoder.cpp` 非 F32/零元素与空输出同回 `MODEL_EMPTY_OUTPUT` | 非空契约失败回契约失败码；真正空输出仍为 `MODEL_EMPTY_OUTPUT` | `src-models` 劣势「SAM 错码」 |

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 接新架构/客户自定义导出仍要写 C++ | 上新检测/分割以外的架构，或接入客户自定义导出权重 | 实现以公开头 + `.inl` 为主，新架构或自定义导出需改 C++ | 新架构或客户自定义导出可在既有接法内完成，或提供可复制模板；是否已接入可用黑盒请求/契约判定 | `src-models` 劣势「写 C++」 |
| P1-2 | CLIP/LightGlue/SAM/diffusion 配错时未在启动或契约层点名 | 配置这些模型时缺段或会话绑定错误 | 配错时失败表现笼统 | 配错在启动或契约层立刻点名缺段或错会话 | `src-models` 劣势「多 session/diffusion」 |
| P1-3 | conf 遗留 engine 清单与 `onnx_sources` 相对产品 toml 漂移，且对齐检查未进日常门禁 | 按遗留清单 convert，或对照 onnx_sources 对产品路径 | depth/metric3d/hrnet 仍旧 engine 名；onnx_sources 与产品 toml 漂移；对齐脚本未挂 doctor/init/CI | 路径一致，或读取方失败并点名产品真源；日常门禁能拦住同类漂移 | `conf` 与 `ops` 劣势 |
| P1-4 | pack 覆盖偏样板，occupancy 依赖声明的 `gpu_mem_mib` | 多客户同机或换 GPU 机 | 仅 4 份 pack，多为 seed 占用；spawn 按声明门禁 | 交付用 pack 含本机校准结果；声明与实测不一致时有可观测文案 | `conf` 与 `src-control` 劣势 |
| P1-5 | 无与 CPU golden 同级的 GPU/TRT 数值黑盒 | 本机本卡确认 TRT 检测框/分数 | Golden 强制 CPU；TRT 用例只断言 session | 选定产品模型在 TRT 路径上有可重复的数值/IoU 门 | `test` 劣势 |
| P1-6 | 默认 PR 下 GPU/TRT 不是 fail-closed | 无自托管 GPU runner 时合入 | skipped 的 gpu-pr-gate 仍让 inference-paths 退出 0 | 无 GPU runner 时 required 检查明确失败，或降级为不可当 TRT 证明 | `ci` 劣势 |

## 劣势
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| D-1 | models deadline 到期后 child 可能仍停在 starting | 模型波超时后看 supervisor 状态 | 只写 models_failed，未把未 running 子进程置终态 | 超时未就绪的 child 不再长期 starting | `src-control` 劣势 |
| D-2 | attentive_gan.toml 输入尺寸仍 stale | 按 toml 配去雨预处理 | toml 写 [240, 320]，notes 为 240×360 | toml 与 session 一致，或启动点名不一致 | `conf` 劣势 |
| D-3 | 缺权重时 golden 默认仍 SKIP | 本机无权重跑 golden | 套件可全绿却未校验数值 | 无权重时能区分「未跑数值」与「已通过」 | `test` 劣势 |
| D-4 | LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 | 回归这两族的契约 | 仅有代表性错码 | 与检测族同级的拒绝矩阵可跑通 | `test` 劣势 |
| D-5 | deployment.md §15 FAQ 结构损坏 | 读 FAQ 找离线拷贝 weights | 残留无 summary 段落并多一个 details 结束标签 | §15 FAQ 可折叠结构完整 | `docs` 劣势 |
| D-6 | 中文 README 路径 B 与现行 full 构建入口不一致 | 按路径 B 从源码构建 | 仍写 setup_full_deps.sh 与手写 cmake | 路径 B 与 deployment/preset 一致 | `docs` 劣势 |

## 附录
不计毛病不进队列：HTTP OCR 仅检测、RTDETR 脚手架、无 MOT HTTP、一 catalog 一进程、超时后外层 SIGKILL、不推翻 HTTP/后端、跨机拷贝 engine、训练/标注/多云。

## 签字
| 角色 | 结论 | 针对版本 | 记录 |
| --- | --- | --- | --- |
| cv-pm | 同意 | v2 | `cowork/ledger/round-010/02-discussion/v2-sign-pm.md` |
| cv-tech-lead | 同意 | v2 | `cowork/ledger/round-010/02-discussion/v2-sign-tech-lead.md` |
| cv-cto | 同意 | v2 | `cowork/ledger/round-010/02-discussion/v2-sign-cto.md` |
