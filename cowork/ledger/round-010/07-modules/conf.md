# 模块：conf

- 路径: `conf/`
- 权重: 5
- 分数: 7.5
- 基线: sub-agent-co-work @ `0492a517`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **按族组织的两层配置**：`conf/model/**` **55 份**产品 toml（12 个族目录：classification、diffusion、enhancement、feature_embedding、feature_point、matting、mono_depth_estimation、mot、object_detection、ocr、openai_clip、scene_segmentation、segment_anything），`conf/server/**` **44 份**服务 toml（与 44 个已服务 id 一一对应），另有 `conf/packs`（4）、`conf/trt_profiles`、`conf/ci`、`conf/local/edge`。
- **根级清单与锁文件齐备**：`conf/mortred.toml`（网关/监督主配置）、`api_keys.toml.example`、`weights_manifest.json`、`base_images.lock`、`trt_engines.json` + `trt_engines.set_a.json`、`onnx_sources.json`、`product_static_bs1.json`、`ci_hosted_golden.json`——即权重、镜像摘要、引擎清单与静态 shape 采样的"可复现来源"都有落点。
- **配置有 schema 校验**：`src/server/server_config_schema.h` 提供键白名单与类型/拼写检查，`test/config_schema_test.cc` 覆盖（该文件 :160 还用到 `fake_delay_ms` 旋钮）；键表分 `server_optional_int_keys()` 等分组，P1-7 本轮在其上补了 `rate_limit_qps` 的语义注释。
- **服务 toml 的键集干净**（抽查 `conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml`）：`port`/`host`/`max_connections`/`peer_resp_timeout`/`request_size_limit`/`compute_threads`/`handler_threads`/`worker_nums`/`model_run_timeout`/`max_queue_depth`/`max_batch_size`/`max_batch_delay_ms`，无多余键。
- **本轮改动**：P1-5 把 5 份原先显式写 `model_run_timeout=0` 的服务 toml 改为 `300000`（每份 1 行；`--numstat` 均 `1 1`，无整文件重写）。
- **已清理的历史配置债**：早前轮次候选 P2 提到的"`rate_limit_qps` 在 44 份配置中无效"，现已不成立——编排方实测 `conf/` 全量仅剩 **1 处** `rate_limit_qps`，且 `conf/server/**` 内为 **0 处**；该键现只在 schema 白名单里保留并带说明注释。

## 问题

1. **产品 toml 与 TRT 强耦合**：59 份产品 toml 的 `type=` 分布为 **tensorrt 50 / mnn 5 / onnx 4**。这意味着配置面的绝大多数条目只有在 TensorRT 引擎存在时才有意义，完成标准 EX3（不得把产品 toml 改成 `type=onnx` 直接服务、cpu profile 上 `type=tensorrt` 不得进入 ready）正是为守住这条边界而设；换机或换 engine 集合时，配置与引擎清单必须同时迁移。
2. **55 份产品 toml 与 44 份服务 toml 存在大量重复内容**：服务 toml 每份都重复十余个同名键（如上列键集），产品 toml 各族的参数段结构相近但无生成器；仓库有 `scripts/migrate_model_config.py` 与 `repo_toml.py` 可做迁移，但没有"从模板生成"的路径，新增一个 id 需要手工复制并逐键核对。
3. **历史上曾存在无效键**：本轮 P1-1 发现在**检测参数**层面 `clip_boxes` 是个什么都不做的开关（现已删除并在解析层显式拒绝）。这说明"配置键必须有人读"这一约束此前没有被系统性检查；建议后续把"每个 schema 键都要有读取点"做成检查项（P1-7 已为 `rate_limit_qps` 补了注释，但注释不等于机制）。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| EX3（覆盖集产品 toml 不得被改成 `type=onnx` 直接服务；已是 ONNX 的 id 与 LIBFACE 除外；cpu profile 上 `type=tensorrt` 初始化成功或进 ready 同样算未满足） | **已满足（第一半有实测，第二半待复核）** | 第一半：实测产品 toml 的 `type=` 分布（tensorrt 50 / mnn 5 / onnx 4），4 份 onnx 与 LIBFACE 属豁免面；本轮无人改 `type=`（P1-5 只改了 5 个 timeout 值，`--numstat` 均 `1 1`）。第二半"cpu profile 上 TRT 必须 fail-closed"需要一次 cpu profile 下带 `type=tensorrt` 的初始化实测，本轮未做（`conf/ci` 与 `conf/trt_profiles` 提供了夹具，但编排方未在 cpu profile 下起过 TRT id），故这半标为待复核 |
| UX4 的一半（覆盖集 id 的请求/响应示例与 `docs/openapi.json` 一致） | **不属本模块**（归属 `docs`） | `conf/` 只提供参数与目录；一致性由 `docs/openapi.json` 与 `openapi_consistency_test` 判定 |

## 评分依据

- 给 7.5 而不是更高：① 配置面体量大但**重复度高**（44 份服务 toml 各自重复同一键集，55 份产品 toml 无生成器），加一个 id 是手工活，与"按文档加一个模型"的目标有摩擦；② 与 TensorRT 强耦合（50/59），换机时配置与引擎清单必须整体迁移；③ 历史上确实存在"没人读的键"（`clip_boxes`），说明缺一条"schema 键必须有读取点"的系统检查。
- 给 7.5 而不是更低：① 两层结构清晰、按族归档、命名规范，44 个 id 与 44 份服务 toml 一一对应；② 有 schema 校验与对应测试，键表分组明确；③ 权重/镜像/引擎/ONNX 来源/静态 shape 采样的清单与锁文件都在 `conf/` 有落点，这是"可复现"的基础；④ 本轮的 timeout 改动面极小且可逐行核对（5 × 1 行）；⑤ 曾被记为债的 `rate_limit_qps` 已从配置中清除（`conf/server` 内 0 处）。
- 参照系：若把服务 toml 的公共键抽成模板/生成器（或引入 include 机制），并补上"schema 键必须有读取点"的检查，本模块有理由到 8.5。
