# 第3步确认

- 问题: P0-1
- 方案版本: v1
- 结论: 不同意
- 条件: `docs/benchmark-h3/libface.txt` 只按一种结果通过：命令含 `--shapes=input:1x3x480x640`，同一文件有硬件、日期和 `Throughput:`。高 480、宽 640 与 `model_input_image_size = [480, 640]`、profile 里 `input` 的 `[1, 3, 480, 640]` 相同。打开这一份文件就能判定。已拉起的 `libface` 在带上该进程正在使用的 Bearer 时，接受 `Content-Type: image/jpeg` 的 `demo_data/model_test_input/object_detection/face_wo_mask.jpg` 原始字节。这一次响应是 HTTP 200，JSON `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0。
- 理由: 验收标准 3 同时写下两套通过结果：未重编时日期为 2026-10-10、Throughput 为 1668.97 qps；重编后硬件、日期和 Throughput 要来自同一次退出码为 0 的 trtexec。文件里没有退出码，验收者也看不出这次有没有重编，一次打开文件确认不了按哪一套通过。验收标准 4 提交这张 JPEG 时没有该进程的 Bearer。`conf/server/object_detection/libface_det/libface_server_config.toml` 没有 `auth_token`，`docs/api-contract.md` 写明 token 为空时推理返回 401。这次 401 发生在会话之前，确认不了合法 JPEG 的成功状态。
