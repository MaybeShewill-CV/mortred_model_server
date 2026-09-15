# 模型开发者指南

本仓库**添加 / 加固 CV 模型**的唯一入口。把模型挂上 HTTP（`conf/server`、OpenAPI、
consistency）见 [how_to_add_new_server.zh-cn.md](how_to_add_new_server.zh-cn.md)。
契约评审清单：[model-contract-governance.md](model-contract-governance.md)。

英文全文与本页同步维护：[model-developer-guide.md](model-developer-guide.md)。

---

## 层概览

所有 CV 模型继承
[`BackendCvModel<INPUT, OUTPUT>`](../src/models/backend/backend_cv_model.h)：

```text
init:      解析 [SECTION.backend] -> 创建 InferenceSession -> on_init([SECTION.params])
run_impl:  prepare_inputs -> session.run -> postprocess(context)
```

标准单图模型只需实现 **preprocess** 与 **postprocess**。后端细节在
[`src/models/backend/`](../src/models/backend/)。优先使用 runtime toolkit：
`ImagePipeline` / `OutputReader` / `SessionIoValidator`。

参考实现：mobilenetv2（MNN）、yolov8_detector（TRT）、ddpm_unet（ORT，非图像输入）。
请求几何放进 `InferenceContext`，不要放模型成员。畸形 tensor 必须返回
`MODEL_OUTPUT_CONTRACT_FAILED`。

配置字段见 [about_model_configuration.zh-cn.md](about_model_configuration.zh-cn.md)。
Catalog 在 `src/factory/<task>_task.h` 加一行 `CvModelEntry`。

---

## 路径 1：十分钟加分类模型

```bash
python scripts/new_model.py --list-tasks
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet --backend mnn --dry-run
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet --backend mnn
```

脚手架生成头文件 / `.inl` / TOML / 契约单测，并打印需手贴的 catalog 与 CMake
片段。未实现的 hook 返回 `MODEL_NOT_IMPLEMENTED`，不会被误上线。
`rtdetr_detector.*` 是入库的脚手架样例。

钩子写法对齐 `mobilenetv2.inl`，然后粘贴 catalog 行与测试 target。

## 路径 2：十分钟加检测模型

`--task object_detection`。注意 YOLO 用 letterbox 几何；人脸与通用检测分属
`face_catalog()` / `catalog()`。详见英文 Path 2。

## 路径 3–6

输出契约宏 `POSTPROCESS_CONTRACT_TEST`、golden 宏与 `MORTRED_UPDATE_GOLDEN`、
`golden_drift_check.py`、shape/dtype 排错表、以及 helper 适用范围 —— 见英文指南
Path 3–6（与本页同一文件族，内容更全）。

## 验证

```bash
cmake --preset full && cmake --build --preset full
scripts/run_tests.sh build/full -R model_golden_test --output-on-failure
python3 scripts/check_consistency.py
```
