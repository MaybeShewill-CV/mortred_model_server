# 模块：src-models

- 路径: `src/models`
- 权重: 15
- 分数: 7.5
- 基线: sub-agent-co-work @ `d05bdd98`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **规模与结构**：137 个源文件、**22,939 行**（`find` + `wc -l` 实测），按 16 个族目录组织：`backend`、`catalog`、`classification`、`clip`、`diffusion`、`enhancement`、`feature_embedding`、`feature_point`、`io`、`matting`、`mono_depth_estimation`、`mot`、`object_detection`、`ocr`、`scene_segmentation`、`segment_anything`。最大文件是 `mot/byte_tracker/byte_tracker.cpp`(942)、`segment_anything/.../sam_amg_decoder.cpp`(902)、`backend/backend_cv_model.h`(790)、`backend/model_runtime.cpp`(782)、`backend/trt_session.cpp`(769)、diffusion 的三个 sampler `.inl`(615/604/595)。
- **三个后端在同一会话接口下**：`backend/` 提供 `session.h`/`session_io.h` 抽象与 `mnn_session.cpp`、`ort_session.cpp`、`trt_session.cpp` 三个实现，外加 `multi_session_model.h`、`inference_context.h`、`param_spec.h`、`f32_output.h`、`nchw_host.h`；GPU 路径另有 `gpu_preprocess.cu`、`gpu_preprocess_desc.h`、`gpu_jpeg_decoder.{h,cpp}`，请求几何与回映射集中在 `request_geometry.h`（`unmap_letterbox_bbox` 的恒定夹取在 `:139-142`）。
- **每个族都有输出契约测试**：`test/` 下 **14 个** `*_output_contract_unittest.cc`（attentive_gan、dinov2、enlightengan、hrnet、lightglue、model（通用）、mono_depth、msocrnet、object_detection、openai_clip、realesrgan、rtdetr_detector、sam、superpoint），覆盖形状与 dtype 契约、类别数、坐标域等。本轮由编排方实跑通过的有：`object_detection_output_contract_unittest` **13/13**、`model_output_contract_unittest` **10/10**、`openai_clip_output_contract_unittest` **6/6**、`lightglue_output_contract_unittest` **15/15**、`tensor_contract_unittest` **14/14**、`backend_unittest` **3 passed / 5 skipped**（权重缺失），全部退出码 0。
- **本轮在此模块完成的两项改造（均已合入并复跑）**：
  - P1-1（`db231f5c`）：删除无效且与实现相反的 `clip_boxes` 开关，改为按本文件既有约定显式拒绝；`min_box_area_px`、`passes_min_box_area`（`detection_params.h:31`）、各检测器过滤位置与 `unmap_letterbox_bbox` 的恒定夹取**零改动**（QA 用 `--numstat` 与 `git diff --stat` 证明）。
  - P1-4（`19c649fa`）：`tensor.h` 的 3 处 `CHECK`/`CHECK_EQ` 改为 `throw std::invalid_argument`，由既有 `invoke_nothrow`（`base_model.h:29-39`）与 `backend_cv_model.h:208-220` 栅栏映射为 `MODEL_RUN_SESSION_FAILED` → 500。**合入后编排方穷举复核：整个 `src/models` 内 `CHECK(`/`CHECK_EQ(`/`LOG(FATAL)`/`std::abort` 0 命中**，模块内已无运行时 abort 路径；仅存 3 处编译期 `static_assert` 与 `mot/byte_tracker/lapjv.*` 的 `#if 0` 关闭的 `ASSERT` 宏族。
- **维护度信号**：整个 `src/models` 只有 **6 处** `TODO/FIXME/XXX`（23k 行规模下密度很低）。
- 参数契约集中在 `backend/param_spec.h` 与各族 `detection_params.h` 一类的小文件里，请求级 override 与配置级参数分开，便于按族审查。

## 问题

1. **同族后处理顺序不一致（N-7，P2 候选）**：v5/v6/v7 是 `unmap → NMS`（`yolov5_detector.inl:88-91`、`yolov6:85-88`、`yolov7:136-141`），v8 是 `NMS → unmap`（`yolov8:75-85`）。由于 `unmap_letterbox_bbox` 会把框恒定夹到源图（`request_geometry.h:139-142`），四条链在**不同坐标系、不同框集合**上计算 NMS IoU，同一张图同族模型可能给出不同框集合。这是"同族行为不一致"，中型企业按族切换检测器时会遇到。
2. **两个 god-file**：`backend/backend_cv_model.h`(790) 与 `backend/model_runtime.cpp`(782) 同时承担会话编排、预处理、形状推导与批处理，是模块内最难按单元审查的两处。
3. **rtdetr 不参与检测参数契约（P1-1 的 QA 残余）**：`rtdetr_detector.inl:65-73` 的 `on_init` 是 `(void)params;` + TODO，不使用 `DetectionParams`，因此本轮对 `clip_boxes` 的显式拒绝不作用于它；改动前后一致（它从未接受该键），不构成回归，但"检测参数契约"实际覆盖 7 个检测器而非全部 8 个。
4. **运行时契约以异常表达**（P1-4 的既定选择）：形状/dtype 违规抛 `std::invalid_argument`，依赖栅栏存在。已在 `tensor.h` 顶部写明"仅因该栅栏才这样抛、栅栏外未捕获仍 terminate"，但这是**契约上的强约束**：将来任何在栅栏外调用 `Tensor::make`/`data<T>()` 的新代码都会把 500 变成进程终止（`src/server/generic_cv_server.h:170` 的 `worker->init()` 已在 init 路径上处于栅栏之外，见 `discovered.md`）。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| AV1（`type=mnn`/`type=onnx` 的已服务 id 能单独拉起并完成一次合法推理；失败不得报成成功） | **未满足（本机缺权重，无法验证）** | 本机无 MNN/ONNX 权重，`backend_unittest` 的 5 例（`MnnSession.InitAndRunMobilenetv2`、`MnnSession.ConfiguredOutputNameOrder`、`OrtSession.InitAndRunDdpmUnet`、`OrtSession.ConfiguredOutputNameOrder`、`MultiBackend.CoexistInOneProcess`）全部 SKIPPED；本轮没有任何一次 mnn/onnx id 的真实推理记录。按 `rubric.md:95`，没有出处的行不得写"已满足"；若评审认为它属"engine/权重行"，可按 `rubric.md:91` 改判阻塞并写恢复条件（备好权重后跑上述 5 例 + 一次真实 HTTP 推理） |
| PF3（一次提交 N>1 张合法图片须返回 N 条逐项结果、batch 观测为 1；单张失败不得拖垮同批或报成空成功） | **已满足** | P1-7 与 P1-2 的重跑中 `server_e2e_contract_test` 45/45 与 47/47 均绿，其中包含多图与逐项结果用例；`model_output_contract_unittest` 覆盖"坏形状 → `MODEL_OUTPUT_CONTRACT_FAILED` → 500"的行级映射（P1-4 QA 引用 `http_status.h` 与 `http_contract_test.cc:43-44`）；P1-4 之后批次内单条失败由 `backend_cv_model.h:208-220` 的 catch 广播 `item_status`，不再 abort 整进程 |
| EX3（覆盖集 toml 不得改成 `type=onnx` 直接服务；cpu profile 上 `type=tensorrt` 不得初始化成功/进入 ready） | **待编排方判定（归属 `conf`/`src-factory-common`）** | 本记录只提供模块侧事实：`backend_config.h` 与 `param_spec.h` 是后端选择与参数契约的落点，三个后端的 `on_init` 在 `src/models` 内；"cpu profile 上 TRT 必须 fail-closed"的判定需要 `conf` 与 factory 侧证据，本记录不代替 |
| ST1 的一半（错误 token / 空图不得得到成功） | **已满足（与本模块相关的一半）** | 空图在信封层被拒（`images[i] must be non-empty`，`request_envelope_unittest` 的 `images_shape_errors` 含 `{"images":[""]}`）；模型侧 dtype/形状违规不再被报成空成功（P1-4：500 + 进程存活） |

## 评分依据

- 给 7.5 而不是更高：① N-7 是**同一任务族内四条链的语义分叉**，而检测是这个项目最常用的族——按 `rubric.md:11` 的"能不能放进生产机"口径，一个小组在 v5 与 v8 之间切换时会看到不一致的框集合，这需要文档解释或统一；② 两个 780+ 行的 god-file 让"看清一次请求的时间构成"这类目标的代码审查成本显著上升；③ 运行时契约依赖"栅栏必须包住所有调用点"这一隐性前提，而 init 路径已破例（`generic_cv_server.h:170`）；④ AV1 在本机无法验证，按标准不能算已满足。
- 给 7.5 而不是更低：① 16 个族、22.9k 行、**14 个按族设立的输出契约测试套件**——这是本仓库测试面最强的模块，本轮编排方实跑的 6 个套件（13/10/6/15/14/3+5）全绿；② 三个后端收敛在同一会话接口下，参数契约小文件化，便于按族增量接入（EX2 的活体证据：`--list` 44 个 id 且网关零硬编码 id）；③ `TODO/FIXME` 只有 6 处；④ 本轮两项改造（P1-1 删除无效开关、P1-4 消灭 abort）都在此模块落地，且合入后穷举确认**模块内已无运行时 abort**。
- 参照系：把 N-7 的顺序统一（或写进文档并用例固定）、并把两个 god-file 拆开之后，本模块有理由到 8.5；若 AV1 在具备权重的机器上跑通，则再上一档。
