# 本轮新增问题

| 来源 | 问题 | 依据 |
| --- | --- | --- |
| P1-7 第3步确认（cv-tech-lead） | 非法 base64 的 violation 指针 `/images/<i>` 是**压缩下标**：`parsed_request.h:130-131` 遍历的是 `decoded.value.images`（只含通过类型检查的字符串），因此 `{"images":[5,"ok"]}` 会报 `/images/0` 而用户看到的元素下标是 1。指针会指错元素，按契约应指向请求里的真实下标 | `src/server/parsed_request.h:126-140`；`src/common/request_envelope.h:118-124`；本用例 `["ok",5]` 恰好末位对齐，故未在测试中固化 |
| P1-1 第3步方案（技术负责人线） | 四条 letterbox 检测链的**后处理顺序不一致**：v5/v6/v7 是 `unmap → NMS`（`yolov5_detector.inl:88-91`、`yolov6_detector.inl:85-88`、`yolov7_detector.inl:136-141`），v8 是 `NMS → unmap`（`yolov8_detector.inl:75-85`）。由于 `unmap_letterbox_bbox` 会把框夹到源图（`request_geometry.h:139-142`），两者在**不同坐标系、不同框集合**上算 NMS IoU，同族模型对同一图可能给出不同框集合 | 上述四处 detector `.inl`；`src/models/backend/request_geometry.h:127-144` |
| P1-5 第3步方案（技术负责人线） | 5 份产品配置显式写 `model_run_timeout=0`（`conf/server/**`：`sam_amg`、`ddpm`、`ddim`、`ldm`、`cls_cond_ddim`），本意是让慢模型不受限。P1-5 一旦把 `<=0` 夹到有界值，这 5 个 id 会变成 504 机器——夹值必须覆盖这些模型的实际 p99，或同时把这 5 份配置改成显式有界值 | `grep -rn 'model_run_timeout=0' conf/server`；`src/server/server_runtime_config.h:151-157` |
| P1-7 第3步方案（技术负责人线） | 模型进程在 `[*_SERVER]` 里接受 `rate_limit_qps` 但永不生效，目前只有注释说明。建议在解析到 `> 0` 时补一行启动 WARNING，让"配了却无效"在日志里可见（与 `receive_timeout<=0`、`max_batch_size<1` 等既有 WARNING 风格一致） | `src/server/server_runtime_config.h:88-89`；`src/server/base_server_impl.h:343-345` |
