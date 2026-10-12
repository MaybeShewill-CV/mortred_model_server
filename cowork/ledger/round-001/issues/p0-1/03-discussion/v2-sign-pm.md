# 第3步确认

- 问题: P0-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 7 均已编号。打开 `libface.toml`、`conf/trt_profiles/set_a_libface.json` 和 `docs/benchmark-h3/libface.txt` 就能核对高 480、宽 640；基准文件只按一组同时成立：命令含 `--shapes=input:1x3x480x640`，同一文件有 `YYYY-MM-DD` 日期、含 `Throughput:` 的一行，以及一行硬件描述。已拉起的 `libface` 工作目录是仓库 `_bin`、可执行文件是 `_bin/mortred-model-server.out`，带上该进程环境里的 Bearer，以 `Content-Type: image/jpeg` 提交 `demo_data/model_test_input/object_detection/face_wo_mask.jpg` 的原始字节。这一次响应是 HTTP 200，JSON `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0。`python3 scripts/product_static_bs1.py --check` 的退出码为 0 也能直接判定。
