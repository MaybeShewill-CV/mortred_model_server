# 模块评分

- id: src-factory-common
- 权重: 5
- 分数: 9.3
- 读过的路径: `src/factory/model_catalog.h`、`src/factory/cv_catalog.h`、`src/common/status_code.h`、`src/common/response_envelope.h`

## 优势
- `model_catalog.h` 与 `cv_catalog.h` 把无 HTTP 面的进程内模型与可挂载 CV 服务条目拆开，避免给尚无 HTTP 面的家族硬塞 server section。
- `CvModelEntry` 带 `make_worker` / `fill_response` / `param_specs` / `family`，`create_server` 能直接拼出 `CvModelServer`，小团队按 catalog id 接模型路径清楚。
- `status_code.h` 用 X-macro 把枚举、wire 整型与文案绑在一处，并标明 wire code 属稳定 HTTP 契约，出错时客户侧可对号定位。
- `response_envelope.h` 统一 encode/decode（含 `partial`、`errors`、未知字段忽略），信封字段与 payload 边界写明，利于联调与向前兼容。

## 劣势
无

## 不计毛病
- 不因「想换一套状态码编号或重写信封形状」而推翻现有 HTTP wire 契约（`status_code.h` 明确 wire codes 须保持稳定；`response_envelope.h` 已固定字段名与编码）。

## 依据
- `src/factory/model_catalog.h`：`ModelCatalogEntry` / `find_entry` / `create_model`；注释说明与 `CvModelEntry` 分离的原因。
- `src/factory/cv_catalog.h`：`CvModelEntry`、`create_server` 注册并构造 `CvModelServer`；未命中时 `LOG(ERROR)` 并返回 `nullptr`。
- `src/common/status_code.h`：`MORTRED_STATUS_CODE_LIST`、`to_underlying`、`status_code_to_str`。
- `src/common/response_envelope.h`：`UnifiedResponse`、`envelope::encode`、`envelope::decode_response` 及文件头契约说明。

## 未读
无
