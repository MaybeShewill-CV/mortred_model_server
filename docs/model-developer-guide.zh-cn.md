# 模型开发者指南

本仓库**添加 / 加固 CV 模型**的唯一入口。把模型挂上 HTTP（`conf/server`、OpenAPI、
consistency）见 [how_to_add_new_server.zh-cn.md](how_to_add_new_server.zh-cn.md)。

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
仓库默认限制解码图 `max_image_pixels` / `max_image_side`；模特类全尺寸输入可在模型 TOML 提高并写明（如 MODNet / PPMatting）。
Catalog 在 `src/factory/<task>_task.h` 加一行 `CvModelEntry`。

---

## 路径 1：十分钟加分类模型

官方复制样例（classification）。验收路径不要加 `--dry-run`，也不要加
`--apply-catalog`——catalog 注册在钩子写完、准备上线前再手填。

```bash
python3 scripts/new_model.py --list-tasks
python3 scripts/new_model.py --task classification \
    --name p1_1_cls_probe --class P11ClsProbe --backend mnn
```

生成头文件 / `.inl` / 模型与 server TOML / 契约单测 / 文档草稿，并打印需手贴的
catalog 与 CMake 片段（脚手架不会自动改共享文件）。未实现的 hook 返回
`MODEL_NOT_IMPLEMENTED`，不会被误上线。`rtdetr_detector.*` 仍是入库的检测脚手架
样例（不进 factory / `--list`）。

### 生成后仍须手填

1. 在 `src/factory/classification_task.h` 增加 catalog `Entry` 与 creator（仅当钩子已实现且准备上线）。
2. 在 `test/CMakeLists.txt` 登记契约测 target（`list(APPEND TEST_LIST …)` 与对应 `EXTRA_LIBS` `elseif`）。
3. 放入真实权重 `weights/classification/<name>/`，补 golden，并改掉契约测里的脚手架期望。
4. 实现 `preprocess` / `postprocess` / `on_init`。

### 黑盒判定（探针不得进 catalog）

固定探针身份：`--task classification --name p1_1_cls_probe --class P11ClsProbe --backend mnn`。
一键验收（推荐）：

```bash
python3 scripts/verify_cls_scaffold_probe.py
```

逐步等价命令（必须清理；不要把探针留在产品树）：

```bash
python3 scripts/new_model.py --task classification \
    --name p1_1_cls_probe --class P11ClsProbe --backend mnn
# 临时把 p11_cls_probe_output_contract_unittest 挂进 test/CMakeLists.txt
cmake --preset full
cmake --build build/full --target p11_cls_probe_output_contract_unittest \
    mortred-model-server.out -j$(nproc)
./_bin/p11_cls_probe_output_contract_unittest
# 期望退出码 0；未填钩子为 MODEL_NOT_IMPLEMENTED
./_bin/mortred-model-server.out --list
# 输出不得含 p1_1_cls_probe / P1_1_CLS_PROBE
# 删除生成物；还原 test/CMakeLists.txt；确认 classification_task.h 无探针 Entry
python3 scripts/check_consistency.py
```

真模型上线时再粘贴 catalog 行与测试 target。验收探针 `p1_1_cls_probe` 的生成物不要提交。

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

ONNX 交换格式导出与缺口：[onnx-interchange.zh-cn.md](onnx-interchange.zh-cn.md)。
