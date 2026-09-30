# 项目评分

- 轮次: 10
- 自算原始分: 8.5
- 自算加权分: 8.5
- 上限规则: P1 上限 8.9
- 相对上轮综合分: 高于 8.4

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 8.4 | `cowork/ledger/round-010/01-modules/src-models.md` |
| src-server-apps | 15 | 9.3 | `cowork/ledger/round-010/01-modules/src-server-apps.md` |
| src-control | 10 | 8.7 | `cowork/ledger/round-010/01-modules/src-control.md` |
| src-factory-common | 5 | 9.3 | `cowork/ledger/round-010/01-modules/src-factory-common.md` |
| conf | 10 | 7.9 | `cowork/ledger/round-010/01-modules/conf.md` |
| ops | 15 | 8.6 | `cowork/ledger/round-010/01-modules/ops.md` |
| test | 15 | 7.5 | `cowork/ledger/round-010/01-modules/test.md` |
| docs | 10 | 9.1 | `cowork/ledger/round-010/01-modules/docs.md` |
| ci | 5 | 8.0 | `cowork/ledger/round-010/01-modules/ci.md` |

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-1 | 接新架构/客户自定义导出仍要写 C++；多 session / diffusion 比单图路径更难一次配稳 | 上新检测/分割以外的架构，或接 CLIP/LightGlue/SAM/diffusion | 实现以公开头 + `.inl` 为主，`CMakeLists.txt` 只编少量源；CLIP/LightGlue/SAM 走 `MultiSessionModel`，diffusion 拆多组件，现场比 YOLO/增强更易配错 | 常见换图/新架构可在既有接法内完成或有可复制模板；多引擎/diffusion 配错能在启动或契约层立刻点名缺段/错会话 | `src-models` 劣势「写 C++」「多 session/diffusion」 | 新证据 |
| P1-2 | SAM 非空契约故障仍报成空输出 | 现场 SAM 权重或后端输出 dtype/形状非法时排障 | `sam_prompt_decoder.cpp` full/low-res mask shape 非法、`sam_vit_encoder.cpp` 非 F32/零元素与空输出同回 `MODEL_EMPTY_OUTPUT` | 非空契约失败回契约失败码；真正空输出仍为 `MODEL_EMPTY_OUTPUT` | `src-models` 劣势「SAM 错码」 | 新证据 |
| P1-3 | conf 遗留 engine 清单与 `onnx_sources` 相对产品 toml 漂移，且对齐检查未进日常门禁 | 按遗留清单 convert、或对照 `onnx_sources` 排 conf 与产品路径不一致 | `trt_engines.json` 中 depth/metric3d/hrnet 仍旧名；`onnx_sources.json` 大量 `product_type=mnn`/旧 runtime；对齐脚本存在但 doctor/init/CI 无调用 | 遗留/onnx 与产品路径一致，或读取方失败并点名产品真源；doctor/init 或 CI 能拦住同类漂移 | `conf` 劣势「遗留 trt_engines」「onnx_sources 漂移」；`ops` 劣势「对齐脚本未挂」 | 新证据 |
| P1-4 | pack 覆盖偏样板，occupancy 依赖声明的 `gpu_mem_mib`，换机/共卡易错 | 多客户同机或换 GPU 机拉起多模型 | 仅 4 份 pack，GPU 侧多为 seed 占用；spawn 按声明与 `nvidia-smi` 门禁 | 交付用 pack 含本机校准结果；声明与实测不一致时有可观测校准/拒启文案 | `conf` 劣势「pack 偏样板」；`src-control` 劣势「occupancy 声明」 | 新证据 |
| P1-5 | 无与 CPU golden 同级的 GPU/TRT 数值黑盒 | 交付前在本机本卡确认 TRT 检测框/分数与预期一致 | Golden 强制 CPU；TRT 用例只断言 session run/形状/dtype | 选定产品模型在 TRT 路径上有可重复的数值/IoU 门 | `test` 劣势「无 TRT 数值对照」 | 新证据 |
| P1-6 | 中文 README 路径 B 与现行 full 构建入口不一致 | 中文读者按「路径 B：full build」从源码构建 | `README.zh-cn.md` 仍写 `setup_full_deps.sh` 与手写 `cmake ..` | 路径 B 与 deployment/preset 一致 | `docs` 劣势「路径 B」 | 新证据 |
| P1-7 | 默认 PR 下 GPU/TRT 不是 fail-closed | 无自托管 GPU runner 时合入改动 | `gpu-pr-gate` 未设变量则 skipped；`inference-paths` 对 skipped 直接绿 | 无 GPU runner 时 required 检查明确失败或降级为不可当 TRT 证明 | `ci` 劣势 | 新证据 |

## 劣势
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| D-1 | models deadline 到期后 child 可能仍停在 `starting` | 模型波超时、gate 已失败时看 supervisor 状态 | 只写 `models_failed` 并 return，未把未 `running` 子进程置终态 | 超时未就绪的 child 不再长期 `starting` | `src-control` 劣势「models deadline」 | 新证据 |
| D-2 | `attentive_gan.toml` 输入尺寸仍 stale | 接 ATTENTIVE_GAN 去雨时按 toml 配预处理 | toml 写 `[240, 320]`，notes 已注 session 实为 240×360 | toml 与 session 尺寸一致，或启动点名不一致 | `conf` 劣势「attentive_gan 尺寸」 | 新证据 |
| D-3 | 缺权重时 golden 默认仍 SKIP | 本机无权重跑 golden | 默认 skip，套件可全绿却未校验数值 | 无权重时结果明确区分「未跑数值」与「已通过」 | `test` 劣势「缺权重 SKIP」 | 新证据 |
| D-4 | LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 | 回归契约分支覆盖面 | 仅有代表性空输出与部分 dtype/size | 与检测族同级的拒绝矩阵可跑通 | `test` 劣势「LG/CLIP 矩阵不全」 | 新证据 |
| D-5 | `deployment.md` §15 FAQ 结构损坏 | 读 FAQ 找离线拷贝 weights | 残留无 summary 段落并多一个 `</details>` | §15 FAQ 可折叠结构完整 | `docs` 劣势「§15 FAQ」 | 新证据 |

## 附录
不计毛病不进队列：HTTP OCR 仅检测、RTDETR 脚手架、无 MOT HTTP、一 catalog 一进程、超时后外层 SIGKILL、不推翻 HTTP/后端、跨机拷贝 engine、训练/标注/多云。

## 未读模块
无
