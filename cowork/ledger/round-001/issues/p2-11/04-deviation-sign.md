# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底）
- 结论: 不同意
- 理由: 这一处偏差不能接受。方案第 4 条的通过结果是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 存在，九个 id 完成官方压测（`libface` 可以空白）。当前两次都停在 mobilenetv2 的 calibrate，清单不存在，退出码是 1，没有提交。w=1 的 `metrics_missing` 为 false、ok=7332，解码投票为 `decode_auto=gpu`，只说明两处允许改动在该测点生效。`pick_w_star` 给出 eligible=true、reason=start_failed 后，已提交的写 pack 规则没有写入 `[pack.mobilenetv2]` 的 `worker_nums` 与 `decode_auto`，部署和官方压测没有发生。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，都要能打开这张按本次约束完成的报告。方案规定的非 0 退出是未写出这两项时的失败处理，不能代替第 4 条的通过。要求 RD 改回本方案版本的交付。方案版本不改。`wait_port_free` 的 15 秒、`pick_w_star`、`occupancy_fields_from_report` 的跳过和手填仍然禁止。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底）
- 结论: 不同意
- 理由: 这一处偏差不能接受。方案第 4 条的通过结果是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 存在；除允许空白的 `libface` 外，各行满足验收标准 1 至 3。`pphuman_mobile` 的 calibrate 已写出 `worker_nums=8`、`decode_auto=gpu`，官方压测 30.032 秒、ok=10474、错误 0。契约要求此时 `backend="jpeggpu"` 的增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和。官方图 `demo_data/model_test_input/scene_segmentation/human_image.jpg` 是 SOF2 渐进 JPEG，现有 GPU 路径遇到渐进扫描不走 jpeggpu；同一次 `MORTRED_DECODE_AUTO=gpu` 的单次请求只有 `mortred_jpeg_decode_total{backend="cpu-reduced"} 1`，jpeggpu 槽已打开。投票网格仍是基线 JPEG，pack 因此写成 gpu，官方图落点仍是 cpu-reduced。该行不能勾销。脚本因此以退出码 1 结束并丢掉清单；dbnet、dinov2_vits14、superpoint、realesrgan、metric3d_512 没有官方压测，没有提交。打不开这一张文件，就不能判定。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，都要能打开这张按本次约束完成的报告。mobilenetv2、yolov8s、libface 在日志里已经满足勾销条件，这三行日志不能代替清单。要求 RD 改回本方案版本的交付：第 4 条退出码 0，清单存在，并且 `pphuman_mobile` 的落点满足上述契约。方案版本不改。`wait_port_free` 的 15 秒、`pick_w_star`、`occupancy_fields_from_report` 的跳过和手填仍然禁止。不得改这张官方图，不得改解码路径，也不得改投票网格。worktree 的 `_bin`、`_lib` 重新链接覆盖了基线里未被 git 跟踪的二进制。这件事不作为接受这一处偏差的理由，方案版本保持本版。
