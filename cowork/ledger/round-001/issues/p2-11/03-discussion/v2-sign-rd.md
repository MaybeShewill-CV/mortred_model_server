# 第3步确认

- 问题: P2-11
- 方案版本: v2
- 结论: 不同意
- 条件: `run_one_point` 在 `run_load` 之前抓 `/metrics`。`mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` 在第一次 `observe_queue_wait_ms` / `observe_inference_duration_ms` 之前没有 bucket 和 count，`render_histogram_locked` 只遍历已经有样本的 `hist.buckets`。这次 before 上 `_metric_series` 返回空，`_hist_delta` 返回空，成功请求的 `metrics_missing` 仍为真。第三个参数改成 `server_listen` 的 `uri` 只能让 after 对上 `{model="<server_uri>",le="..."}` 和 `{model="<server_uri>"}`。方案只允许改这一个参数，不准改差值、`assess_point`、`pick_w_star` 和写 pack 规则。默认 30 秒的 `--write-pack` 因此仍不写 `worker_nums` 与 `decode_auto`，解码投票不跑。
- 理由: 九份 `conf/server` 的 `server_uri` 与方案表一致。`run_one_point` 已经取出这个 `uri`，样本出现后现有正则对得上。gateway `:8080`、supervisor `:8787` 和九个模型端口（9002、9056、9051、9072、9060、9111、9020、9012、9041）彼此不同，开始前连不上就退出做得到。两只 worker gauge 在 `/metrics` 进入 `set_model(_m_server_uri)` 之后带的是 `server_uri`；数值要到 `reply_sync_request` 才离开 0，压测后的那一次刮取可以把二者之和读成 `worker_nums`。写回 pack 仍然做不到：before 没有直方图样本行，标签对齐清不掉 `metrics_missing`，清单第 1、2 条的勾销字段出不来。
