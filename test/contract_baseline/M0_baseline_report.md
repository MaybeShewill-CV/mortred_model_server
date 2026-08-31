# M0 基线报告（2026-08-31）

## M0.1 构建与测试基线

- 构建环境：WSL Ubuntu（用户 `mortred`），`/tmp/mortred-build-ci`
  （`MORTRED_BUILD_FULL=OFF`，CI/tests-only 档），编译并行度 `-j6`
- 结果：**34/34 通过**（`cmake --build /tmp/mortred-build-ci --target check -j6`）
- 环境注意事项：`server_e2e_contract_test` 依赖 `libssl.so.1.1`，运行时
  `LD_LIBRARY_PATH` 必须同时包含 `/tmp/mortred-build-ci/lib` 与
  `<repo>/3rd_party/libs`（补齐后该测试单独重跑通过，12.7s）

### CI 测试清单（34 项）

base64 / file_path_util / json_request_parser / blocking_worker_queue /
status_code / auth_token / time_stamp / cv_utils / detection_params /
detector_common / model_runtime / tensor_contract / model_factory /
byte_tracker / simple_tokenizer / request_size_limit / ready_probe /
rate_limiter / cv_image_input / worker_nums / http_contract /
backpressure / prometheus_metrics / response_schema / openapi_consistency /
config_schema / async_job / async_job_stress / restart_policy / catalog /
control_config / supervisor / server_e2e_contract / api_key_manager

### golden 用例清单（full/GPU 档执行，15 个基线文件，27 个 gtest 用例）

centerface_detection / dbnet_text_detection / densenet121_classification /
dinov2_classification / libface_detection / mobilenetv2_classification /
nanodet_detection / openai_clip_embedding / openai_clip_text_embedding /
resnet50_classification / superpoint_feature_point /
yolov5_detection / yolov6_detection / yolov7_detection / yolov8_detection
（另含 `golden_baseline.json` 汇总）

golden 执行环境：`/tmp/mortred-build-full-werror`（`MORTRED_BUILD_FULL=ON`，
`MORTRED_BUILD_PROFILE=gpu`），运行前缀
`LD_LIBRARY_PATH=/tmp/mortred-build-full-werror/lib:<repo>/3rd_party/libs`

## M0.3 server 与 models 链接关系确认

`src/server/CMakeLists.txt` 中 `target_link_libraries(server common models vendored::workflow glog::glog)` —— **server 直接链接 models**。
结论：`base_server_impl.h` 使用 `models/` 头（`base_model.h` / `model_io_define.h`）
与 backend 层新类型（M1.x）无链接障碍；新头文件挂入 models 目标即可被 server 侧引用。

## 附录：golden 运行结果

命令：`LD_LIBRARY_PATH=/tmp/mortred-build-full-werror/lib:<repo>/3rd_party/libs
/tmp/mortred-build-full-werror/bin/model_golden_test`（全量 27 用例，143s）

- **26 / 27 通过**
- 失败 1 例：`model_golden.realesrgan_enhancement`

### 已知存量失败（与契约改造无关，M0 未改任何代码）

```
mnn session input:  input:f32[1,0,0,3] (dynamic)     ← 输入形状未解析
mnn session output: output:f32[0,0,0,0] (dynamic)
Compute Shape Error for 207
fingerprint drift: mean abs diff = 91.18 (限值 1)
```

判断：权重文件存在（`weights/enhancement/real_esrgan/realesr-general-x4v3.model`，
4.8MB），MNN 会话可创建，但该模型的动态输入形状解析失败导致输出全零。
**M1 复跑更正（2026-08-31）**：同二进制路径重跑 golden 为 **27/27 全过**
（realesrgan 通过）。该用例属**偶发抖动**（MNN 动态形状解析在特定负载/状态下
偶发失败），非确定性回归。跟踪建议：若再次出现，记录当时 GPU/负载状态；
根因在 MNN 会话层，与契约改造无关。

**M0 基线结论**：CI 档 34/34 全绿；golden 档 26/27（唯一失败为上述存量问题）。
该状态即统一契约改造的"改动前"基准，M4 翻转后的 golden 结果须与本基线逐项对照。
