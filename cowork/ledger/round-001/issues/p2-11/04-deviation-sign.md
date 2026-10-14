# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底）
- 结论: 不同意
- 理由: 这一处偏差不能接受。方案第 4 条的通过结果是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 存在，九个 id 完成官方压测（`libface` 可以空白）。当前两次都停在 mobilenetv2 的 calibrate，清单不存在，退出码是 1，没有提交。w=1 的 `metrics_missing` 为 false、ok=7332，解码投票为 `decode_auto=gpu`，只说明两处允许改动在该测点生效。`pick_w_star` 给出 eligible=true、reason=start_failed 后，已提交的写 pack 规则没有写入 `[pack.mobilenetv2]` 的 `worker_nums` 与 `decode_auto`，部署和官方压测没有发生。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，都要能打开这张按本次约束完成的报告。方案规定的非 0 退出是未写出这两项时的失败处理，不能代替第 4 条的通过。要求 RD 改回本方案版本的交付。方案版本不改。`wait_port_free` 的 15 秒、`pick_w_star`、`occupancy_fields_from_report` 的跳过和手填仍然禁止。
