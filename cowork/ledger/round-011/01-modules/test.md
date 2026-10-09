# 模块评分

- id: test
- 权重: 15
- 分数: 6.0
- 读过的路径: `test/`（目录清单）、`test/CMakeLists.txt`、`test/model_golden_test.cc`、`test/model_golden_registry.h`、`test/ci_require_weights.h`、`test/golden/yolov8_detection.json`、`test/golden/mobilenetv2_classification.json`、`test/contract_baseline/v1_http_contract.md`、`test/backend_unittest.cc`（TRT 段）

## 优势
- `test/CMakeLists.txt` 在 `MORTRED_BUILD_FULL` 下挂上 `model_golden_test`，该项 `SKIP_RETURN_CODE` 为 77，且二进制自带 `main`，不链接 `gtest_main`。
- `test/model_golden_test.cc` 的 `golden_process_exit`：选中用例全部因文案含 `weights not available` 而跳过时，标准输出一行 `NUMERICAL_NOT_RUN` 并返回 77。`MORTRED_CI_REQUIRE_WEIGHTS=1` 时同一处改为 `GTEST_FAIL`。
- 已有 CPU 数值门，且比较规则对得上 H2：分类 `expect_scores` 的 `class_id`（`mobilenetv2`、`resnet`、`densenet`）；检测 `expect_boxes` 同类且 IoU ≥ 0.5（`yolov5l`、`yolov6s`、`nanodet_1x5`、`yolov7`、`yolov8s`）；人脸同类框（`libface`、`centerface`）；`dbnet` 检测框 IoU ≥ 0.5；增强指纹均值差 ≤ `k_fingerprint_diff`（`attentive_gan`、`enlightengan`、`realesrgan`）；`superpoint` 关键点距离；`dinov2_vitb14` 余弦 ≥ `k_embedding_cos_thresh`。
- `Yolov8TrtDetection` 只加载产品 toml `conf/model/object_detection/yolov8/yolov8s.toml`，不调用 `force_cpu`，对 `test/golden/yolov8_detection.json` 里 score ≥ 0.5 的框要求同类且 IoU ≥ 0.5。CPU 用例 `yolov8_detection` 仍指向 `conf/ci/yolov8_onnx_hosted.toml`（权重为 `yolov8s.onnx`）。

## 劣势
- 这些 id 没有满足 H2 的 CPU 数值比较：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088` 在 `test/model_golden_test.cc` 里没有用例。`bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 虽有 CPU 用例，但 `compare_scene_segmentation` / `compare_matting` 只走 `expect_fingerprint`（64×64 均值差 ≤ 1.0），没有掩码重合阈值。`yolov8_detection` 与 `yolov8_onnx_detection` 都加载 `yolov8s.onnx`。现在：`model_golden_test --gtest_list_tests` 列不出上述无用例 id 的数值门；指纹用例失败文案是 fingerprint drift。改完后：每个点名 id 有一条 CPU 用例，检测失败于同类且 IoU < 0.5，分割和抠图失败于掩码重合低于测试写明的阈值，embedding 和深度失败于测试写明的距离或误差阈值。
- 产品 toml 为 `type=tensorrt`、但没有同规则 TRT 比较的 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。各 id 的 `conf/server` 指向的 `conf/model` 均为 `type = "tensorrt"`。`test/model_golden_test.cc` 里唯一不 `force_cpu` 的数值门是 `Yolov8TrtDetection`。现在：TRT 构建下 `model_golden_test --gtest_list_tests` 只有这一条做同类 IoU。改完后：每个点名 id 多一条加载该产品 toml、且不把 backend 改成 cpu 的用例，失败条件与该任务的 CPU 规则相同。

## 不计毛病
- SAM、diffusion、CLIP、LightGlue 不在最高标准范围里。`fastsam` / SAM / CLIP / DDPM 的 golden，以及 `test/backend_unittest.cc` 的 `TrtSession` 只断言 session 能跑和输出形状，不按 H2 记成劣势。
- `test/contract_baseline/v1_http_contract.md` 自述是改造前快照，不是现行契约。
- 缺权重全跳过退出 77 已在 `golden_process_exit` 落地，不记成劣势。`yolov8s` 的 TRT 同类 IoU 门只覆盖 `yolov8s`，不把其余 id 算进这一条的完成。

## 依据
- 构建与退出码：`test/CMakeLists.txt`（`model_golden_test`、`SKIP_RETURN_CODE 77`、ctest 名 `yolov8_detection` / `Yolov8TrtDetection`）；`test/model_golden_test.cc` 的 `golden_process_exit` 与 `Yolov8TrtDetection`。
- CPU 比较规则：`test/model_golden_registry.h` 的 `k_box_iou_thresh`、`k_fingerprint_diff`、`k_embedding_cos_thresh`、`expect_scores`、`expect_boxes`、`expect_text_regions`、`expect_keypoints`、`expect_fingerprint`；用例登记在 `test/model_golden_test.cc`。
- 权重文件与 id 对应：`conf/ci/yolov8_onnx_hosted.toml` 为 `yolov8s.onnx`；`conf/ci/nanodet_onnx_hosted.toml` 为 `nanodet_plus_m_1x5`；`conf/ci/yolov7_cpu_golden.toml` 为 `yolov7.mnn`；`conf/ci/pphuman_onnx_hosted.toml` 为 mobile 192；`conf/ci/ppmatting_onnx_hosted.toml` 为 human_512；`conf/ci/dinov2_vitb14_onnx_hosted.toml` 只被 `dinov2_feature_embedding` 使用。
- catalog id 与产品 toml：`src/factory/*_task.h` 的 `catalog()` / `face_catalog()`；各 `conf/server/**/*server_config.toml` 的 `model_config_file_path`；对应 `conf/model/**` 的 `type = "tensorrt"`。
- 样例：`test/golden/yolov8_detection.json`（5 框，含 class 0 与 class 5）；`test/contract_baseline/v1_http_contract.md`。

## 未读
无
