# 第3步确认

- 问题: P2-11
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 使用者是调用这一台推理服务器的业务后端。打开 `docs/perf/coverage_set_http_rps.md` 即可逐行判定，验收标准 1 至 5 都已编号。标准 1、2 覆盖锁定的九个 id：calibrate 后按 pack 部署，`--raw` 压测命令的并发等于已部署的 `worker_nums`、时长 30 秒，行内有硬件、日期、`decode_auto`、成功 RPS、p50、p99、不含 429 的错误率、Throughput 与比值，以及分开的排队直方图和推理直方图；`conf/server` 的 `worker_nums` 仍是 1。`yolov8s` 的分母 400.677 与 `docs/benchmark-h3/yolov8s.txt` 一致，其余八个 id 的 Throughput 文件也都在。错误率高于 1% 的行不勾销。`libface` 允许空白，其余写明字段的行仍勾销；其他 id 空白则不通过。标准 3 能读到同一次压测的 gateway 五段和模型 `server_time_ms` p50，并在阶段 p50 大于该值一半时写明主导段，且不创建 `/tmp/mortred_stage_timing`。这与评分复核里 PF4 仍挂在本条、P1-1 只关闭计时实现的范围一致。标准 4 把这张清单当作第一次合格 HTTP RPS 记录。标准 5 不要求九个 id 同时常驻，也不要求 `yolov8s` 与 `libface` 同时 ready。九个产品 toml 已是 `type=tensorrt`，九份 `conf/server` 的 `worker_nums` 已是 1，calibrator v4 的 `--raw` 与默认 30 秒写 pack、`http_infer_rps.py --raw`、gateway 五段和响应里的 `server_time_ms` 都是复用；新增的是这一张清单，以及压测输出里补上 `server_time_ms` 的 p50。引擎保持 static shape、batch=1。共享 context 不作为勾销条件。本方案不突破 `docs/unsupported-boundaries.md`，也不把训练或标注写进本仓库。
