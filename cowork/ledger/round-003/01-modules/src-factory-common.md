# 模块评分

- id: src-factory-common
- 权重: 10
- 分数: 9.2
- 读过的路径: `src/factory/model_catalog.h`、`src/factory/cv_catalog.h`、`src/factory/classification_task.h`、`src/factory/obj_detection_task.h`、`src/factory/scene_segmentation_task.h`、`src/factory/enhancement_task.h`、`src/factory/feature_point_task.h`、`src/common/status_code.h`、`src/common/response_envelope.h`

## 已具备

- 各家族 `catalog()` 以小写 `model_section` 注册覆盖集与教程相关 id：`mobilenetv2`、`yolov5l`、`bisenetv2`、`attentive_gan`、`superpoint`、`densenet` 等。
- EX2：在已有家族 `catalog()` 增加带 `fill_response` 与 `server_section` 的行后，`--list` 与 gateway 转发不需改路由函数。
- ST1 在本模块：未知字段、空 `images`、空字符串、错误 token 得不到 `StatusCode::OK` 成功载荷。

## 问题

无

## 范围外

- 英文 EX1 点名的两份文档不要求每模型单独二进制；中文 how-to 曾有的大写示例已在第 2 轮关闭，不重复入列。
- `decode_response` 把可解析 JSON 的 `DecodeResult.status` 标成 OK 仍是实现细节；服务路径用 `encode`，一次 HTTP 验收看不到，不入列。

## 依据

分数 9.2：九个覆盖 id、统一服务器与请求拒绝路径有文件出处。本模块无未关闭问题。

## 未覆盖

无
