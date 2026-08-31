# 统一请求契约（Contract v1）代码改造计划

> 目标：单一契约（`images[]` / `params` / `options` / `results[]`），编码与语义分离，
> 为 base64 与原始字节（P0-4）预留同一接缝。本文件是进度追踪清单：每个 ID 对应
> 一个可独立编译、可独立验证的提交，提交信息建议带 ID（如 `[M4.1] ...`）。

## 总原则

1. **每个 ID 一步**：小步提交，任一步之后 `check` 目标全绿
2. **M1–M2 不改变任何现有行为**（新类型/新解析器并行就位，未接线）
3. **M4 是契约翻转点**：`task_request` 改形（M3.5）必须与 M4.1 在同一提交窗口合入
4. **永不回头原则**：`img_data` 永不成功（422 + migration 提示）；未知键永远 422；
   `images` 恒为数组（单图也是 `images[0]`）
5. 回滚策略：契约翻转是一个提交系列，出问题整体 revert，无配置开关、无中间态发布

---

## M0 基线（0.5 天）

- [x] **M0.1** 全量构建 + 测试基线通过，记录 golden 用例清单（迁移对照用）
      ✅ 2026-08-31：CI 档（/tmp/mortred-build-ci, FULL=OFF, -j6）**34/34 全绿**；
      golden 档（full-werror, GPU）**26/27**，唯一失败 realesrgan_enhancement 为
      存量问题（MNN 动态形状未解析，详见 test/contract_baseline/M0_baseline_report.md）。
      ⚠️ 运行 e2e/golden 需 `LD_LIBRARY_PATH` 同时含 build/lib 与 3rd_party/libs。
      验证：`cmake --build <build-dir> --target check`
- [x] **M0.2** 固化当前 v1 HTTP 契约样本（复用 `test/http_contract_test.cc` 的
      请求/响应 fixture，存为迁移 diff 的"改动前"基准）
      ✅ 产出 `test/contract_baseline/v1_http_contract.md`：请求/响应信封、11 类任务
      data 形态、状态码→HTTP 映射、异步端点、公开端点、v1→统一契约变更对照表。
      （`base_server_impl.h` 已 include `models/base_model.h`，需确认 .so 链接）
- [x] **M0.3** 确认 `src/server/CMakeLists.txt` 对 models 目标的链接关系
      （`base_server_impl.h` 已 include `models/base_model.h`，需确认 .so 链接）
      ✅ `target_link_libraries(server common models vendored::workflow glog::glog)`
      —— server 直接链接 models，M1.x 新类型挂入 models 目标即可被 server 引用。
      （`base_server_impl.h` 已 include `models/base_model.h`，需确认 .so 链接）

---

## M1 类型层：模型侧新增，零行为变化（2–3 天）

- [x] **M1.1** `src/models/io/common_input.h`：新增类型
      ```cpp
      struct byte_source { enum class kind { base64_text, raw_bytes }; kind origin; std::string data; };
      struct image_input { byte_source image; const jinq::models::backend::ParamSet* params = nullptr; };
      ```
      （`ParamSet` 前向声明，避免包含环）
- [x] **M1.2** 新增 `src/models/backend/param_spec.h`：
      - `ParamSpec` 流式构建器：`f32/i32/boolean/string`、`range/values/desc/request_overridable`
        （风格对齐既有 `SessionSpec`/`IoSpec`）
      - `ParamValue`（variant）与 `ParamSet`：≤16 项扁平 kv 存储（非 map，无逐请求节点分配），
        `get_f32/get_i32/get_bool/get_str(key, default)`、`contains/keys`
      - `validate(请求 kv) → vector<{pointer, message}>`：未知键 / 类型 / 越界 / 枚举 / 不可覆盖键
      - 新增 `test/param_spec_unittest.cc`（表驱动全覆盖）+ 挂入 `src/models/CMakeLists.txt`、`test/CMakeLists.txt`
- [x] **M1.3** `src/common/status_code.h`：X-macro 追加
      `INVALID_REQUEST_PARAMETER=66`、`REQUEST_ITEM_LIMIT=67`、`DEADLINE_EXCEEDED_PARTIAL=68`；
      `src/server/http_status.h` 补映射（66→422，67→413，68→200+partial）；
      `test/status_code_unittest.cc` 补断言
- [x] **M1.4** `src/models/cv_image_input.h`：新增 `load_image(const image_input&, limits, status, error)`
      - `base64_text` 分支复用现有 base64 解码逻辑（不动）
      - `raw_bytes` 分支：字节直接 `imdecode`（零 base64 解码；P0-4 接缝现在就位）
      - `ImageInputLimits`（max_pixels/max_side）对两种 origin 同样强制
      - `test/cv_image_input_unittest.cc` 补两种 origin + 超限用例
- [x] **M1.5** `src/models/backend/inference_context.h`：`InferenceContext` 增加
      `const ParamSet* params = nullptr`
- [x] **M1.6** `src/models/backend/backend_cv_model.h`：`prepare_inputs` 默认实现对
      `image_input` 输入填充 `prepared.context.params = input.params`；
      `preprocess/postprocess` 钩子签名不变；自定义输入模型（CLIP/SAM/LightGlue/扩散）不受影响

**M1 验收**：`check` 全绿；现有 golden 测试逐字节不变（模型行为完全未动）。

---

## M2 解析层：新解析器就位，未接线（2 天）

- [x] **M2.1** 新增 `src/server/output_options.h`：`OutputOptions`
      `{encoding: png|jpeg|webp, include_image, max_results, echo_params}`，
      严格已知键校验 + 各任务无关的默认值
- [x] **M2.2** 新增 `src/server/request_envelope.h`：
      `parse_request(body, ParamSchema, OutputSchema) → ParsedRequest{req_id, items[], params, options}`
      或结构化错误 `{code, pointer, message, migration}`：
      - 未知键严格拒绝；`img_data` 识别 → 66 + `migration: "img_data → images[0]"`
      - `images` 必须为非空字符串数组（元素将来允许 string|object，纯附加）
      - rapidjson，沿用 never-throw 纪律；`req_id` 透传
- [x] **M2.3** 新增 `test/request_envelope_unittest.cc`（替代/吸收
      `json_request_parser_unittest.cc`）：合法单图/多图、未知参数、越界、类型错、
      枚举错、`img_data`、空数组、畸形 JSON、超长字符串、params 条数上限
- [x] **M2.4** `src/common/http_response.h`：新增 v2 响应信封结构
      `{status, status_str, task_id, model{name,version}, results[], server_time_ms, partial}`
      （仅新增结构，不改 `build_response_body` 行为）

**M2 验收**：编译 + 新单测全绿；线上行为零变化。

---

## M3 装配层：worker 输入类型切换 + 工厂挂参数模式（3–4 天）

- [x] **M3.1** `src/server/generic_cv_server.h`：
      - `Base64Input` 别名更名 `ImageInput` = `io_define::common_io::image_input`
      - `CvServerSpec` 增加 `std::vector<ParamSpec> param_specs`（默认空 = 该模型不接受请求参数）
      - `CvResponseFiller` 签名追加 `const OutputOptions&`
- [x] **M3.2** `src/server/response_serializers.h`：全部 `fill_*` 机械追加 options 参数
      （多数现阶段忽略；图像输出类任务读取 `encoding/include_image`）
- [x] **M3.3** `src/factory/*_task.h`（15 个任务文件）：输入别名切换（机械）；
      `obj_detection_task.h` 逐 entry 挂 `param_specs`（score_threshold/nms_threshold/top_k）
- [x] **M3.4** `src/models/object_detection/detector_common.h` + `detection_params.h`：
      后处理从 `InferenceContext.params` 读取覆盖值，缺省回落 TOML 配置值
      （yolov5/6/7/8/nanodet/centerface/libface 共用同一路径，一处改动覆盖全族）
- [x] **M3.5** `src/server/async_job_table.h`：`task_request` 改形
      `{task_id, is_valid, parse_status, items[], params(shared_ptr), options, deadline(steady_clock)}`
      ⚠️ **必须与 M4.1 同一提交窗口**（`serve_process` 同步切到新解析器）
- [x] **M3.6** CMake：新文件挂入 models/server 目标与测试目标

**M3 验收**：与 M4.1 合并窗口后 `check` 全绿。

---

## M4 内核翻转：契约切换点（4–5 天，原子提交系列）

- [x] **M4.1** `src/server/base_server_impl.h`：
      - `parse_task_request` → `parse_request`（信封解析 + 严格校验）
      - `img_data` → 422 + migration；未知键 → 422 + pointer
      - `max_request_items`（默认 16，`[SERVER]` 可覆写）→ `REQUEST_ITEM_LIMIT`
      - **背压按 item 计数**：`_m_waiting_jobs`/EWMA/`Retry-After` 全部以 item 为单位
      - **deadline 传播**：`deadline = steady_now + model_run_timeout`，存入 `task_request`
- [x] **M4.2** `do_work` 单请求路径：一次 worker 领取跑完该请求全部 items，
      逐项检查剩余预算，逐项产出结果，用完归还 worker
- [x] **M4.3** 批路径：每 item 一个 `batch_entry` + 请求级完成闩
      （`shared_ptr<atomic<int>>`，请求方超时消失不 UAF）；
      收集窗口取 `min(剩余 deadline, max_batch_delay_ms)`
- [x] **M4.4** 响应组装：`results[]` 与 `images[]` 下标对齐、每项独立 status、
      顶层聚合 status + `partial` 标志；`reply_json`/`build_response_body` 切新信封
      （`req_id` 进 `task_id` 出，命名延续）
- [x] **M4.5** 异步端点：`/jobs` 存 items 形态 `task_request`；`async_run_job` 复用
      与同步相同的 run-items 核心；`/result` 返回统一信封
- [x] **M4.6** `src/control/gateway/gateway_app.cpp`：**修复 Content-Type 写死 bug**，
      原样透传客户端 `Content-Type` 与 `Accept`（内部 token 注入逻辑不变）
- [x] **M4.7** metrics：received/finished 按 item 计数；encoding label 预留（现恒 `"json"`）；
      参数值**永不**进 label（约定写入代码注释）
- [x] **M4.8** 测试重写为统一契约：
      `http_contract_test.cc`、`server_e2e_contract_test.cc`、`gateway_e2e_test.cc`、
      `response_schema_test.cc`、`openapi_consistency_test.cc`、`model_output_contract_unittest.cc`、
      `fake_model_server.cc`、`backpressure_unittest.cc`（item 计数语义）、
      `async_job_unittest.cc`/`async_job_stress_test.cc`、`request_size_limit_unittest.cc`
- [x] **M4.9** `conf/server/**`：`max_request_items` 写入 1–2 个示例文件并注释默认值
      （代码有默认，避免 27 文件机械改动）
- [x] **M4.10** golden 参数扫描：同图不同 `score_threshold` 断言
      `count(0.9) ≤ count(0.35)` 单调 + 固定阈值框集哈希稳定（对引擎噪声免疫）

**M4 验收**：
1. 全链路（直连 + 过网关 + 异步）只说统一契约；`img_data` 请求得到 422 + migration
2. 多图请求中注入 1 张损坏图，其余项结果正确（隔离）
3. 16 图请求与 16 个单图请求对 `max_queue_depth` 占用等价（背压公平）
4. deadline 中途耗尽 → 部分结果 + `partial:true`
5. 仓库内无 v1 旧断言残留（grep `img_data` 仅存在于 migration 提示与测试）

---

## M5 契约出口：代码生成，消灭漂移（2 天）

- [ ] **M5.1** 新增 `scripts/contract_dump.cc` + CMake target `contract_dump`：
      实例化各任务目录，输出 `{task, model, params, options, io}` JSON
      （先例：`scripts/trt_engine_inspect.cc`）
- [ ] **M5.2** `scripts/gen_openapi.py` 消费 dump 产物；重新生成 `src/server/openapi_doc.h`
- [ ] **M5.3** 新增 `scripts/check_contract_sync.py` + 接入 CI
      （`.github/workflows/ci.yml`）：再生成结果必须 == 仓库内 spec
- [ ] **M5.4** `scripts/server/test_server.py`、smoke 脚本、`README`/`README.zh-cn`、
      `CHANGELOG.md` 更新到统一契约示例

**M5 验收**：OpenAPI 与 C++ 声明零漂移（CI 强制）；`contract_dump` 输出即文档源。

---

## M6 raw 编码接入（P0-4，2–3 天，可与 M5 并行）

- [ ] **M6.1** `serve_process` 编码三分支：`application/json` |
      `image/*` 与 `application/octet-stream`（body 即 `images[0]`，raw `byte_source`）|
      其余 415 + 支持清单响应头
- [ ] **M6.2** `X-Mortred-Params` / `X-Mortred-Options` 头解析（值为紧凑 JSON），
      走**同一个** ParamSpec 校验器（422 语义与 JSON 路径逐字节一致）
- [ ] **M6.3** 限额：raw 按原始字节计 `request_size_limit`（天然比 base64 宽 ~33%）；
      `ImageInputLimits` 同样强制；chunked 由 workflow 层兜底
- [ ] **M6.4** metrics encoding label 启用（`json | raw`，两取值）；e2e 断言：
      同一请求两种编码的 `results[]` **逐字节一致**
- [ ] **M6.5** 基准交付：同模型同图 json/base64 vs raw 的 p50/p99/吞吐/内存峰值对比表

**M6 验收**：双编码语义等价有测试钉死；基准数据入库存档。

---

## 总体验收矩阵

 | 能力 | 验证方式 |
 |---|---|
 | 单图/多图/参数严格校验 | request_envelope 单测 + e2e |
 | 每项失败隔离 | e2e 损坏图注入 |
 | 按 item 背压 + Retry-After | backpressure 单测 + 压测 |
 | deadline 部分结果 | 单测 + e2e |
 | 异步 = 同一契约 | `/jobs` → `/result` 信封一致 |
 | 网关透传 Content-Type | gateway e2e |
 | OpenAPI 零漂移 | CI 再生成比对 |
 | 双编码等价 | M6.4 断言 |
 | 参数语义进 golden | M4.10 |

## 工期与依赖

- 串行总量 ≈ 3.5–4 周；M5 与 M6 可并行
- 关键路径：M1 → M3(除3.5) → M2 → [M3.5+M4.1 原子] → M4 其余 → M5/M6
- 风险最高的两步：M4.1（翻转）与 M4.3（批路径闩）；各配独立单测先行
