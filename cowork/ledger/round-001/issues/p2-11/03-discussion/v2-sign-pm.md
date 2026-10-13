# 第3步确认

- 问题: P2-11
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 使用者是调用这一台推理服务器的业务后端。打开 `docs/perf/coverage_set_http_rps.md` 即可逐行判定，验收标准 1 至 5 都已编号。标准 1 覆盖锁定清单：九个 id 逐行勾销 calibrate 的 `--raw`、pack 里的 `worker_nums` 与 `decode_auto`、`conf/server` 的 `worker_nums` 仍为 1、部署后的进程 worker 数与 `MORTRED_DECODE_AUTO`、`--raw` 压测并发等于已部署 `worker_nums` 且不少于 30 秒，以及硬件、日期、成功 RPS、p50、p99、不含 429 的错误率、Throughput 与比值、分开的排队直方图和推理直方图、`mortred_jpeg_decode_total` 落点。标准 2 把计入这些字段的 `model` 标签定为该行写出的 `server_uri`，并与契约表、对应 `conf/server` 一致；目录 id 上的同名序列不计入。九份 `conf/server` 的 `server_uri` 与契约表一致，`worker_nums` 已是 1；九个产品 toml 已是 `type=tensorrt`。`yolov8s` 的分母 400.677 与 `docs/benchmark-h3/yolov8s.txt` 一致，其余八个 id 的 Throughput 文件也都在。错误率高于 1% 的行不勾销。`libface` 允许空白，其余写明字段的行仍勾销。标准 3 在同一行读到 gateway 五段（`gw_in`、`route`、`auth`、`fwd`、`upstream`）和模型 `server_time_ms` p50，阶段 p50 大于该值一半时写明主导段，且不创建 `/tmp/mortred_stage_timing`。标准 4 把这张清单当作第一次合格 HTTP RPS 记录。标准 5 不要求九个 id 同时常驻，也不要求 `yolov8s` 与 `libface` 同时 ready。calibrator v4 的 `--raw` 与默认 30 秒写 pack、`http_infer_rps.py --raw`、gateway 五段、响应里的 `server_time_ms`、以及按 `server_uri` 渲染的 worker gauge 与直方图都是复用；扩展的是 `run_one_point` 把已有 `server_uri` 交给直方图读取，压测输出补上 `server_time_ms` 的 p50，并新增这一张清单。引擎保持 static shape、batch=1。共享 context 不作为勾销条件。本方案留在 `docs/unsupported-boundaries.md` 的支持范围内，不把训练或标注写进本仓库。
