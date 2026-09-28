# 模块评分

- id: src-factory-common
- 权重: 5
- 分数: 7.6
- 读过的路径:
  - `src/factory/model_catalog.h`
  - `src/factory/cv_catalog.h`
  - `src/common/status_code.h`
  - `src/common/response_envelope.h`

## 优势
- HTTP 可挂载与进程内/基准调用拆成两套 catalog：`CvModelEntry` 带 `make_worker` / `fill_response` / `param_specs` / `family`，`ModelCatalogEntry` 只做 `make_model`，避免无 HTTP 面的模型被强行塞进 generic CV server（`cv_catalog.h`、`model_catalog.h`）。
- Status 用 X 宏单点定义枚举、wire 整型与文案，并写明 wire code 属 HTTP 契约须稳定，小团队对接客户与排障有统一码表（`status_code.h`）。
- 响应信封字段名、encode/decode 集中在一处，文档写明 wire JSON 形状与「客户端忽略未知字段」；支持 `partial`、`errors[].pointer/message`、`server_time_ms`，便于交付后看超时部分结果与字段级错误（`response_envelope.h`）。
- Catalog 按 `model_section` 查找，命中后组装 `CvServerSpec` 并经 `ServerFactory` 创建，接新检测/分割类服务的注册路径短、可读（`cv_catalog.h`）。

## 劣势
- Catalog 未命中或 `make_model` 为空时只 `LOG(ERROR)` 并返回 `nullptr`，不落到 `StatusCode`，小团队在起服/接模型失败时难从同一套 wire 码定位是 section 写错还是工厂未注册（`model_catalog.h`、`cv_catalog.h`）。
- 后端相关码偏 TensorRT（`TRT_CUDA_ERROR` 等），MNN/ORT 无对等细分；多后端混部时客户侧「CUDA/TRT 报错」与 CPU 路径失败粒度不对齐（`status_code.h`）。
- `decode_response` 对顶层缺 `status`/`status_str` 等仍可 `ok=true`，联调时半残信封不易在公共层被拦住（`response_envelope.h`）。

## 依据
- 分数 7.6：四份头文件表明本模块能支撑「挂模型 + 稳定 HTTP 契约」交付，但工厂失败面与状态码/解码严格度不够「出错即定位」到接近 9–10；出处见上列优势/劣势对应路径。

## 未读
无
