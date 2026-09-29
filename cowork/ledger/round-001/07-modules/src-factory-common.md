# 模块评分

- id: src-factory-common
- 权重: 5
- 分数: 7.6
- 读过的路径: 上轮四份头文件；本轮 diff 未触及 `src/factory` / `src/common`

## 优势
- HTTP catalog 与进程内 catalog 分开；Status 码表稳定；响应信封集中。

## 劣势
- Catalog 未命中只记日志并返回 nullptr，不落到 StatusCode。
- 后端错误码偏 TensorRT，MNN/ORT 无对等细分。
- `decode_response` 对缺字段仍可能 ok。

## 依据
- 分数 7.6：沿用上轮。本轮 diff 未改该模块。

## 未读
本轮未重读，沿用上轮记录。
