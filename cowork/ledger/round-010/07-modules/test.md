# 模块：test

- 路径: `test/`
- 权重: 10
- 分数: 7.0
- 基线: sub-agent-co-work @ `05c74314`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **规模**：顶层 **84 个 `.cc` / 19,932 行**，另有 `test/golden/`(62 文件)、`test/contract_baseline/`(2)、`test/testdata/`(1)。按用途分布：`*unittest*` **70**、`*output_contract*` **14**、`*e2e*` **3**、golden **1**。
- **按族设立的输出契约套件（14 个）**是本仓库测试面最厚的部分：attentive_gan、dinov2、enlightengan、hrnet、lightglue、model（通用）、mono_depth、msocrnet、object_detection、openai_clip、realesrgan、rtdetr_detector、sam、superpoint。
- **本轮实跑通过的目标（约 20 个二进制，全部退出码 0）**：`request_envelope_unittest` 17、`server_e2e_contract_test` 45→**47**、`gateway_e2e_test` 49、`detection_params_unittest` 8、`detector_common_unittest` 15、`object_detection_output_contract_unittest` 13、`prometheus_metrics_test` 7→**11**、`tensor_contract_unittest` **14**、`http_contract_test` 6、`model_output_contract_unittest` 10、`backend_unittest` 3（+5 SKIP）、`openai_clip_output_contract_unittest` 6、`lightglue_output_contract_unittest` 15、`server_runtime_config_unittest` 5→**6**、`supervisor_unittest` 24→**26**、`supervisor_malformed_e2e_test` 1、`request_admission_unittest` 5、`log_buffer_unittest` 1 等。SKIP 均有明确原因（MNN/ONNX 权重缺失）。
- **四个条目的测试侧改造都在本模块**：P1-7（3 个契约测试对齐产品行为，含把 `{"images":["ok",5]}` 从单违规改为更强的双违规断言）、P1-1（新增拒绝用例 + 邻位守卫）、P1-6（4 条归一化用例，原 `method="POST"` 断言 0 删除行）、P1-4（3 条 `EXPECT_THROW` 精确断言 + 存活证明）、P1-5（三态配置用例 + 白盒访问器）、P1-2（`/proc/<pid>/environ` 用例，经**变异验证**）、P1-3（`fake_model_server` 新增孙进程模式 + 正例/反例两条用例）。
- **测试注册机制清晰**：`add_executable(EXCLUDE_FROM_ALL)`（`test/CMakeLists.txt:324`）、`add_test(... WORKING_DIRECTORY ${PROJECT_ROOT_DIR})`（`:344`，保证以仓库根为 CWD）、`add_dependencies(check ${src})`（`:348`）、`EXTRA_SRCS` 为需要额外 TU 的目标补源（`:163/188/261`）、`LABELS "sanitizer"`（`:354`）与 `LABELS "valgrind"`（`:498`）。
- **本轮没有实现者改坏测试基础设施**：7 个条目共 7 次提交，无一次触碰 `test/CMakeLists.txt`（逐条 `git show --name-only` 核对）。

## 问题

1. **`check` 聚合目标无法完成，58 个注册用例因此从未在一次运行里跑过**（用户已移出范围的"构建/门禁断裂"类，仅留档）：`test/CMakeLists.txt:348` 对 `TEST_LIST` 每一项都 `add_dependencies(check ${src})`，而 `packed_batch_nvi_unittest`(`:54`)、`cv_image_input_unittest`(`:68`) 链接失败、`enlightengan_output_contract_unittest` 编译失败。**编排方在主检出直接复现（非转述）**：三者 `build_exit=2`（前两者 `undefined reference to gpu_jpeg::GpuDecodeSlot::{ctor,dtor}`，后者 `enlightengan.inl:169,170: error: ‘named_tensor_as’ is not a member of ‘jinq::models::backend’`）。
2. **`model_golden_test` 的 `SKIP_RETURN_CODE 77` 让"跳过"与"通过"在退出码上不可区分**（同属用户已移出的"数值门禁可跳过"类）：`test/CMakeLists.txt:346` 只对这一个目标设了 `SKIP_RETURN_CODE`，配合"缺权重一律 skip"，一个未真正校验数值的运行与一次真正的通过，在 CI 汇总里都表现为绿。
3. **`sanitizer` 标签与实际构建目标不匹配（7 vs 6）**：`test/CMakeLists.txt:349-355` 给 7 个目标打标签（含 `rate_limit_gcra_unittest`），而 `.github/workflows/ci.yml:328` 只构建 6 个，`:335` 却用 `ctest -L sanitizer` 选择——被选中但未构建的目标没有可执行文件，该步失败。编排方实测三个构建目录下 `ctest -N -L sanitizer` 均登记 **7** 个。归 `ci` 模块记录（`07-modules/ci.md` 问题 1）共同持有。
4. **权重缺失导致 5 例长期 SKIP**：`backend_unittest` 的 `MnnSession.*`/`OrtSession.*`/`MultiBackend.CoexistInOneProcess` 因权重缺失跳过，使完成标准 AV1（mnn/onnx id 能单独拉起并完成一次合法推理）在本机无法判定（见 `07-modules/src-models.md` 的 AV1 行）。

**关于"注册数与可运行数"的口径**（编排方实测，防止误读）：`build` 91 注册 / 34 有二进制、`build/full-cpu` 96/35、`build/probe-cpu` 95/34、`build/probe-ci` 63/33。**"二进制缺失"不等于"无法构建"**——这些目录都是部分构建的产物；可确证无法构建的只有问题 1 里的三个具名目标。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| PF3（一次提交 N>1 张图须返回 N 条逐项结果、batch 观测为 1；单张失败不得拖垮同批或报成空成功） | **已满足** | `server_e2e_contract_test` 47/47（含多图与逐项结果用例）；批次内单条失败由 `backend_cv_model.h:208-220` 广播 `item_status`（P1-4 之后不再 abort 整进程）；`http_contract_test` 6/6 锁定错误码映射 |
| ST1（未知字段/空图/错误 token 不得得到成功；健康检查除外） | **已满足** | `request_envelope_unittest` 17/17（含 `images_shape_errors` 的 `{"images":[""]}` 与未知字段指针）；`server_e2e_contract_test` 的 `strict_envelope_rejects_unknown_fields_and_bad_params`(`:505`) 与 `async_submit_unknown_field_metrics_use_http_422`(`:1116`)；401 用例见 `gateway_e2e_test`/`server_e2e_contract_test` |
| ST3（429 + `Retry-After`；拒绝后仍可成功） | **已满足** | `queue_limit_returns_429_with_retry_after`(`:937-965`) 与 `async_queue_full_returns_429`(`:1306`)；"拒绝后仍能成功"由 P1-5 在 `max_queue_depth=1` 下以第二次请求 200 证明 |
| AV1（mnn/onnx id 能单独拉起并完成一次合法推理） | **不可在本机判定**（见 `src-models` 的 AV1 行） | 权重缺失 → 5 例 SKIP；本模块不代替判定 |
| UX3（CI 仍检查 OpenAPI 一致；GPU 跳过不得写成 TensorRT 证明） | **不属本模块** | 归属 `ci`/`docs`；本记录只提供"相关测试存在且可运行"的事实（`openapi_consistency_test` 断言 `additionalProperties == false`，`ci.yml:272/:460` 调用生成/同步检查） |

## 评分依据

- 给 7.0 而不是更高：① **本模块的三条基础设施完整性都有实际缺陷**——`check` 聚合目标跑不完（58 个注册用例从未在一次运行里整体跑过）、`SKIP_RETURN_CODE 77` 让"跳过"与"通过"同码、`sanitizer` 标签把未构建的目标选进 TSan 门；测试模块的价值就在于"能不能拿它当门"，这三条都直接削弱这一点；② 权重缺失导致 5 例长期 SKIP，使 AV1 这类判定在本机不可完成；③ 84 个顶层测试文件平铺在一个目录里（另有 62 个 golden 文件），查找与归属成本高。
- 给 7.0 而不是更低：① 覆盖**广且按族组织**（14 个输出契约套件 + 3 个 e2e + 70 个单测 + golden 脚手架），本轮实跑约 20 个二进制全绿，是"能讲清一次失败"的基础；② 注册机制规范（固定 CWD = 仓库根、`EXCLUDE_FROM_ALL`、`EXTRA_SRCS` 显式补源、`check` 聚合、标签分组），不是散装脚本；③ 本轮 7 个条目的测试侧改造都在此模块落地，且**无一次触碰 `test/CMakeLists.txt`**（没有为了让测试通过而改基础设施）；④ SKIP 都有明确原因并写在测试里（权重缺失），不是静默跳过。
- 参照系：修好问题 1（补 `libmodels` 链接 / 补 include）与问题 2（让跳过在汇总里可区分，例如非零专用退出码并在 CI 里单独统计）之后，本模块有理由到 8.0 以上；问题 3 是 `ci` 与本模块的共同项。
