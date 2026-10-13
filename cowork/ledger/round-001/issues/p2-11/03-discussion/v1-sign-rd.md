# 第3步确认

- 问题: P2-11
- 方案版本: v1
- 结论: 不同意
- 条件: 模型进程 `/metrics` 的 `model` 标签是 `server_uri`，不是目录 id。`serve_process` 在渲染前执行 `set_model(_m_server_uri)`，九个 id 的实际标签分别是 `/mortred_ai_server_v1/classification/mobilenetv2`、`/mortred_ai_server_v1/obj_detection/yolov8s`、`/mortred_ai_server_v1/obj_detection/libface`、`/mortred_ai_server_v1/scene_segmentation/pphuman_mobile`、`/mortred_ai_server_v1/ocr/dbnet`、`/mortred_ai_server_v1/feature_embedding/dinov2_vits14`、`/mortred_ai_server_v1/feature_point/superpoint`、`/mortred_ai_server_v1/enhancement/realesrgan`、`/mortred_ai_server_v1/mono_depth_estimation/metric3d_512`。不存在 `mortred_workers_available{model="<id>"}` 或 `mortred_workers_busy{model="<id>"}`。v4 的 `_metric_series` 只匹配 `{model="<id>",le="..."}`；成功请求因此被标成 `metrics_missing`，`eligible` 为假，`occupancy_fields_from_report` 不写 `worker_nums` 和 `decode_auto`，解码投票也不会跑。方案禁止改 `scripts/calibrate_pack.py`，改动面也不包含这条标签。
- 理由: 端口、pack 单 id 自启动、gateway 五段 `stage` 标签、JPEG `backend` 标签、`server_time_ms` 和 `http_infer_rps.py` 的测量窗都能按 v1 接上。卡在 worker 数和写回 pack：按现有标签，calibrate 默认 30 秒的 `--write-pack` 不会写出 `[pack.<id>]` 的 `worker_nums` 与 `decode_auto`，部署后也读不到方案写的 `model="<id>"` 之和。这两项是验收第 1、2 条的勾销字段，现有进程给不出。
