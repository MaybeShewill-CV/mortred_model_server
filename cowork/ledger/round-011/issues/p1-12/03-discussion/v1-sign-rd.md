# 第3步签字

- 问题: P1-12
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 契约表 38 份产品 toml 的 `backend.type` 都是 `tensorrt`、`device` 都是 `gpu`，`model_file_path` 与表中 engine 一致。`Yolov8TrtDetection` 已经只做 `fix_toml_paths`、不调用 `force_cpu_backend` / `load_model_cfg` 就 `init`/`run`；`InferenceSession::create` 按 `type=tensorrt` 且 `device=gpu` 建 `TrtSession`，名单里的 creator 都走这条 `BackendCvModel`。已有 CPU 比较的阈值和失败文案与方案一致；没有 CPU 用例的检测、embedding、深度可以在 `model_golden_registry.h` 另写会打印阈值的比较，不必改现有 `expect_*`。`depth_map` 已是拉回原图尺寸的 `CV_32FC1`。`golden_process_exit` 与 `MORTRED_SKIP_OR_FAIL_WEIGHTS` 已能给出 0、1、77 以及 `NUMERICAL_PASSED` / `NUMERICAL_NOT_RUN`。不改进程模型、HTTP 和产品 toml。
