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

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑）
- 结论: 不同意
- 理由: 这一处偏差不能接受。方案第 4 条的通过结果是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 存在；除允许空白的 `libface` 外，各行满足验收标准 1 至 3。dbnet 的 calibrate 已写出 `worker_nums=2`、`decode_auto=gpu`。官方压测 `wall_s=60.104`、warmup 5 秒、并发 4（已部署 worker 数的两倍）、HTTP 200、成功 RPS 88.93，测量参数与本版一致。契约要求此时 `backend="jpeggpu"` 的增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和。官方图是现有 demo `demo_data/model_test_input/ocr/railway_ticket.png`。现有 GPU 路径只接受 JPEG SOF，这张 PNG 落到 CPU `imdecode`，`cpu-full` 增加而 `jpeggpu` 不增加。评分标准写明实际路径和 `decode_auto` 不一致的报告不算数。该行不能勾销。失败行不写入、已写入行保留、脚本以非 0 退出，是后面的 id 失败时的处理；`--check` 仍要求只有 `libface` 空白且其余八行齐全才通过。当前四行清单和退出码 1 不能代替第 4 条的通过。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，都要能打开这张按本次约束完成的报告。`pphuman_mobile` 使用 `face_w_mask.jpg` 且落点为 gpu，解码路径和 16 张投票网格未改，这三项符合本版。要求研发改回本方案版本的交付：第 4 条退出码 0，清单存在，并且 dbnet 的落点满足上述契约。方案版本不改。不得改 `railway_ticket.png`，不得改解码路径，不得改 16 张投票网格，不得手填该行，不得放宽落点检查。`superpoint` 与 `metric3d_512` 尚未运行，不在这一处偏差里，不据此改方案版本。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑）
- 结论: 同意
- 理由: 接受这一处停住。停住不是第 4 条通过。第 4 条的通过结果仍是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外八行齐全、满足验收标准 1 至 3。当前清单只有 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile` 四行，自测未执行，没有提交。这四行不能代替第 4 条。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，仍要能打开那张按本次约束完成的报告。`dbnet` 的现有 demo 是 `demo_data/model_test_input/ocr/railway_ticket.png`。`jpeg_full_dimensions` 只在 `FF D8` 时返回真；`use_gpu_path` 还要求这次 SOF 非渐进且方向不超过 1。`jpeggpu` 只在 GPU JPEG 解码成功后增加。这张 PNG 进不了该门，`jpeg_reduce_factor` 对非 JPEG 返回 1，请求落到 `cv::imdecode`，增加的是 `cpu-full`。上次确认已经记下该行 `worker_nums=2`、`decode_auto=gpu`。gpu 契约要求 `jpeggpu` 增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和。评分标准写明实际路径和 `decode_auto` 不一致的报告不算数。本版允许的改动不改变这张 PNG 的官方 `--raw` 字节。上次禁止改 `railway_ticket.png`、改解码路径、改 16 张投票网格、手填该行、放宽落点检查，仍然有效。投票网格是另外 16 张 JPEG，时长仍是 10 秒。在这些限制里，官方压测产生不了 `jpeggpu` 增量，第 4 条到不了退出码 0。研发没有改代码，没有续跑，没有清空已有四行，没有提交。方案版本不改。`superpoint` 与 `metric3d_512` 尚未运行，不在这一处偏差里。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑，含继续决定）
- 结论: 同意
- 理由: 接受这一处停住。停住不是第 4 条通过。第 4 条的通过结果仍是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外各行齐全、满足验收标准 1 至 3。当前清单是 6 行：前四行保留，并写入了 `dbnet` 与 `dinov2_vits14`。`superpoint`、`realesrgan`、`metric3d_512` 不在清单里。`python3 scripts/coverage_http_report.py --from dbnet` 退出码 1，没有提交。这 6 行不能代替第 4 条。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，仍要能打开那张按本次约束完成的报告。`superpoint` 使用现有 demo `demo_data/model_test_input/feature_point/test.png`。w=1、并发 2、60 秒：ok=7523、errors=0、`metrics_missing=false`、p99=32.053ms。已提交的 `P99_CAP_MS["superpoint"]` 是 25.0。`assess_point` 在 p99 高于该上限时返回 `latency_budget`。第一测点失败且没有已通过测点时，`pick_w_star` 返回 `eligible=false`、`passed_worker_nums` 为空。`eligible` 不为真时解码投票不跑。pack 保持初值 `worker_nums=1`，没有 `decode_auto`。官方 `--raw` 压测没有开始。失败行不写入、已写入行保留、脚本以非 0 退出，是本版对后面的 id 因已提交门槛写不出 `worker_nums` 与 `decode_auto` 的处理。本版禁止改 `pick_w_star` 和 25ms 上限，禁止手填，禁止为 `superpoint` 改 demo。在这些限制里，这一行写不出来，第 4 条到不了退出码 0。研发没有改这两处门槛，没有把失败行写入，没有跑 `realesrgan` 与 `metric3d_512`，没有提交。方案版本不改。`dbnet` 使用 `coco_text_test.jpg`，`worker_nums=2`，`decode_auto=gpu`，`-c 4`，`wall_s=60.0819`，warmup 5 秒，成功 RPS 168.67，错误率 0，`jpeggpu=11214`。`dinov2_vits14` 使用现有 demo，`worker_nums=2`，`decode_auto=cpu`，成功 RPS 179.725。这两行是继续决定里已经写入的行，不把这一处停住写成第 4 条已经通过。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑，含继续决定与再跑 superpoint）
- 结论: 同意
- 理由: `superpoint` 在 demo 仍为 `demo_data/model_test_input/feature_point/test.png`、w=1、并发 2、ok=7377、errors=0、`metrics_missing=false`、p99=32.085ms 时，被已提交的 `P99_CAP_MS["superpoint"]=25.0` 判为 `latency_budget`。`pick_w_star` 返回 `eligible=false`、`passed_worker_nums` 为空，解码投票没跑，`[pack.superpoint]` 只有 `worker_nums=1`、没有 `decode_auto`，官方 `--raw` 压测没有开始。脚本退出码 1，失败行未写入，清单仍是 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14` 这 6 行，`realesrgan` 与 `metric3d_512` 没有跑，没有提交。这就是「用户决定（再跑 superpoint）」写明的停住：不改 25ms 上限，不改 `pick_w_star`、`assess_point`、demo、解码路径和 16 张投票网格，不手填。研发没有改这些。同意的是这次停住。自测第 4 条仍要求退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外各行齐全；这次没有达到。同意停住不是第 4 条通过，不授权合入。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑，含继续决定与 superpoint 上限 40）
- 结论: 不同意
- 理由: 这一处偏差不能接受。自测第 4 条的通过结果是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外各行齐全、满足验收标准 1 至 3。`P99_CAP_MS["superpoint"]` 已改为 40.0。w=1、并发 2：ok=7525、errors=0、p99=31.869ms。w=2、并发 4：ok=13390、errors=0、p99=32.561ms。w=4、并发 8：p99=63.922ms，reason=`latency_budget`，扫描停住。`eligible=true`，`passed_worker_nums` 为 1 和 2。解码投票 16 张，cpu=6、gpu=10，选出 `decode_auto=gpu`。pack 写出 `worker_nums=2`、`gpu_mem_mib=140`、`gpu_mem_at_workers=2`、`gpu_mem_source=device_delta`、`decode_auto=gpu`。官方压测 `-c 4`、`-d 60s`、`--warmup 5s`：ok=12875、errors=0、p99=32.537ms、wall_s=60.0788、成功 RPS=214.302。契约要求 `decode_auto=gpu` 时 `backend="jpeggpu"` 的增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和。demo 仍是 `demo_data/model_test_input/feature_point/test.png`，PNG，444778 字节，文件头 `89 50 4E 47`。GPU 路径只接受可解析的 JPEG，这张 PNG 落到 CPU 全解码并计入 `cpu-full`，落点对不上。该行不能勾销。脚本 reason=`jpeg landing`，退出码 1，失败行未写入。清单仍是 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14` 这 6 行。`realesrgan` 与 `metric3d_512` 没有跑。没有提交。用户决定要求 `superpoint` 写出 `worker_nums` 与 `decode_auto` 后继续这两个 id。这次停在 jpeg 落点，后面的 id 尚未因各自已有种子上限走到 `latency_budget`。评分标准写明实际路径和 `decode_auto` 不一致的报告不算数。PF1、TP1，以及尚无已记录值时的 PF5，都要能打开这张按本次约束完成的报告。要求研发改回本方案版本的交付：第 4 条退出码 0，清单里除允许空白的 `libface` 外各行齐全，并且 `superpoint` 的落点满足上述契约。方案版本不改。不得改这张 demo，不得改解码路径，不得改 16 张投票网格，不得手填，不得放宽落点检查，不得放宽 `realesrgan` 的 200.0 和 `metric3d_512` 的 150.0。这次停住不是第 4 条通过，不授权合入。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑，含继续决定与 superpoint 上限 40）
- 结论: 同意
- 理由: 接受这一处停住。停住不是第 4 条通过。第 4 条的通过结果仍是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外各行齐全、满足验收标准 1 至 3。当前清单仍是 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14` 这 6 行。`superpoint`、`realesrgan`、`metric3d_512` 不在清单里。官方压测和 `--from superpoint` 都未执行，没有提交。这 6 行不能代替第 4 条。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，仍要能打开那张按本次约束完成的报告。`superpoint` 的官方 `--raw` 字节是 `demo_data/model_test_input/feature_point/test.png` 的文件字节。444778 字节，前 8 字节 `89 50 4E 47 0D 0A 1A 0A`，随后 IHDR，不是 `FF D8`。coverage 脚本只给 `pphuman_mobile`、`dbnet`、`realesrgan` 设置替代图；`superpoint` 的校准和官方压测都用这张 PNG。`run_load` 用 `read_bytes()` 原样发送。`jpeg_full_dimensions` 只在 `FF D8` 时为真；`use_gpu_path` 还要求 SOF 非渐进且方向不超过 1。PNG 使 `sof_ok` 为假，不走 jpeggpu。`jpeg_reduce_factor` 对非 JPEG 返回 1，请求落到 `cv::imdecode`，增加 `cpu-full`。`jpeggpu` 只在 GPU JPEG 解码成功后增加。上次实跑已写出 `decode_auto=gpu`（投票 16 张 JPEG，cpu=6、gpu=10；上限 40 下 w=1、w=2 通过）。gpu 契约要求 `jpeggpu` 增量大于 0，并且严格大于 `cpu-reduced` 与 `cpu-full` 之和。这张 PNG 使 `jpeggpu` 保持 0。评分标准写明实际路径和 `decode_auto` 不一致的报告不算数。上次禁止改这张 PNG、另存替代图、改解码路径、改 16 张投票网格和 10 秒、手填 `decode_auto`、放宽落点、改 `pick_w_star`、`assess_point`、`_hist_delta` 和写 pack 规则，仍然有效。用户决定（superpoint 上限 40）也不改这张 demo。投票网格是另外 16 张 JPEG。在这些限制里，官方压测产生不了 `jpeggpu` 增量，失败行不写入，脚本非 0，`realesrgan` 与 `metric3d_512` 不会开始，第 4 条到不了退出码 0。研发没有改代码，没有续跑，没有提交，没有做这些被禁止的改法。同意的是这次停住。同意停住不是第 4 条通过，不授权合入。方案版本不改。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑，含继续决定、superpoint 上限 40 与 superpoint JPEG）
- 结论: 同意
- 理由: 接受这一处停住。停住不是第 4 条通过。第 4 条的通过结果仍是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外各行齐全、满足验收标准 1 至 3。当前清单是 7 行：原 6 行保留，并写入了 `superpoint`（`worker_nums=1`，`decode_auto=gpu`，图 `demo_data/model_test_input/feature_point/test.jpg`，SOF0，640×480，55892 字节；calibrate w=1、并发 2、p99=15.986ms，上限仍是 40.0；解码投票 16 张、10 秒，结果 gpu；官方 `--raw` `-c 2 -d 60s --warmup 5s`，`wall_s=60.0691`，成功 RPS 206.978，错误率 0，429 为 0，`jpeggpu` 增量 14440，`cpu-reduced` 与 `cpu-full` 增量为 0，落点 gpu）。`test.png` 未改。`realesrgan` 与 `metric3d_512` 不在清单里。`python3 scripts/coverage_http_report.py --from superpoint` 退出码 1，没有提交。这 7 行不能代替第 4 条。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，仍要能打开那张按本次约束完成的报告。`realesrgan` 使用已有 `logs/p2-11-realesrgan-256.jpg`。w=1、并发 2、60 秒：ok=426、errors=0、p50=282.618ms、p99=306.621ms、queue_p50=174.838、infer_p50=175.0、`metrics_missing=false`。已有 `P99_CAP_MS["realesrgan"]=200.0`。`assess_point` 在 p99 高于该上限时返回 `latency_budget`。第一测点失败且没有已通过测点时，`pick_w_star` 返回 `eligible=false`、`passed_worker_nums` 为空。pack 只有初值 `worker_nums=1`，没有 `decode_auto`。官方压测没有开始。失败行未写入。`metric3d_512` 没有跑。这就是「用户决定（superpoint JPEG）」写明的停住：后面的 id 因各自已有种子上限停在 `latency_budget` 时停住，不提前放宽 200.0 与 150.0，不手填，不改 `pick_w_star`、`assess_point`、解码路径和 16 张投票网格。研发没有做这些。同意的是这次停住。同意停住不是第 4 条通过，不授权合入。方案版本不改。

# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户选定（以 v2 为底，2026-10-14 重跑，含继续决定、superpoint 上限 40、superpoint JPEG 与 realesrgan 上限 400）
- 结论: 同意
- 理由: 接受这一处停住。停住不是第 4 条通过。第 4 条的通过结果仍是退出码 0，并且 `docs/perf/coverage_set_http_rps.md` 里除允许空白的 `libface` 外各行齐全、满足验收标准 1 至 3。当前清单是 8 行：原 7 行保留，并写入了 `realesrgan`（`conf_server_worker_nums=1`，`worker_nums=4`，`decode_auto=cpu`；官方压测 `scripts/server/http_infer_rps.py --raw --model realesrgan --gateway http://127.0.0.1:8080 -c 8 -d 60s --warmup 5s --image logs/p2-11-realesrgan-256.jpg`；`wall_s=60.3619`，成功 RPS 24.568，错误率 0，`p99_ms=383.273`，`p50_ms=322.613`，`jpeg_landing=cpu`，`jpeggpu` 增量 0，`cpu-full` 增量 1620）。`P99_CAP_MS["realesrgan"]` 已是 400.0，没有改回去。`metric3d_512` 不在清单里。`python3 scripts/coverage_http_report.py --from realesrgan` 退出码 1。脚本提交是 `44b95e6f9ffb8ad7fac81ccb924f4bb6ad28b382`，清单和图片没有进这次提交。这 8 行不能代替第 4 条。评分标准里 PF1、TP1，以及尚无已记录值时的 PF5，仍要能打开那张按本次约束完成的报告。`metric3d_512` 的 calibrate 使用 `demo_data/model_test_input/mono_depth_estimation/0000000005.jpg`。w=1、并发 2、60 秒：`started=true`，rps=14.223，p99=165.429ms，`queue_p50=75.0`，`infer_p50=75.027`，errors 未单列，reason=`latency_budget`，`oom=false`，metrics 有读数。已有 `P99_CAP_MS["metric3d_512"]=150.0`。`assess_point` 在 p99 高于该上限时返回 `latency_budget`。第一测点失败且没有已通过测点时，`pick_w_star` 返回 `eligible=false`、`passed_worker_nums` 为空。`eligible` 不为真时解码投票不跑。pack 只有初值 `worker_nums=1`，没有 `decode_auto`。官方压测没有开始。失败行未写入。这就是「用户决定（realesrgan 上限 400 与 metric3d JPEG）」写明的停住：`metric3d_512` 因已有 150.0 停在 `latency_budget` 时停住，不提前放宽这个上限，不手填，不改 `0000000005.png`，不改 `pick_w_star`、`assess_point`、解码路径和 16 张投票网格。研发没有做这些。同意的是这次停住。同意停住不是第 4 条通过，不授权合入。方案版本不改。


# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户决定（官方压测并发等于 worker_nums）
- 结论: 不同意
- 理由: 这一处偏差不能接受。本版要求 `python3 scripts/coverage_http_report.py --reload` 读取现有九行的 `worker_nums` 与 `decode_auto`，按该 pack 拉起后跑 `http_infer_rps.py --raw`，`-c` 等于该行 `worker_nums`，`-d 60s`，`--warmup 5s`。九行全部成功后才覆盖 `docs/perf/coverage_set_http_rps.md`，保留原来的 calibrate 命令，并更新这次官方压测的命令、成功 RPS、p50、p99、错误率、429、两只直方图、JPEG 落点、gateway 五段和 `server_time_ms`。通过结果是 `--reload` 退出码 0，清单被这次测量覆盖。实际临时 pack 只有 `[pack.mobilenetv2]`、`worker_nums = 4`、`decode_auto = "cpu"`，没有 `gpu_mem_mib`，没有调用 calibrate。TensorRT 占用门闩在 `gpu_mem_mib` 缺失或小于等于 0 时拒绝拉起。supervisor 把该模型标成 failed：`occupancy gate: stamp missing for mobilenetv2 (gpu_mem_mib>0 required for TensorRT)`。官方压测没有开始，其余八个 id 没有跑。清单没有被覆盖，仍是上一轮 `-c` 等于 `worker_nums` 两倍的九行，其中 mobilenetv2 为 `-c 8`、`worker_nums=4`。没有提交。`--reload` 退出码是 1。三条自测退出码 0 只覆盖脚本样例，不是这次官方压测。评分标准 PF1、TP1 要求按结论部署后再用 `--raw` 压测，并发数等于已部署的 `worker_nums`；GPU 作业被跳过时不得写成已满足。PF2、PF4 要同一次压测的排队直方图、推理直方图、gateway 五段和 `server_time_ms`。现有清单的并发仍是两倍，不满足本版「`-c` 不等于该行 `worker_nums` 则不通过」。任一行失败时不覆盖清单，是失败处理，不能代替本版的通过。要求研发改回本方案版本的交付：`--reload` 退出码 0，模型按该 pack 拉起并完成官方压测，九行成功后清单才被覆盖，压测字段来自这次 `-c` 等于 `worker_nums` 的测量。方案版本不改。不得调用 calibrate，不得重新扫描 worker，不得手填 `worker_nums`、`decode_auto` 或压测字段，不得执行无参数的 `coverage_http_report.py`，不得改 `scripts/calibrate_pack.py` 的两倍扫描。这次退出码 1 不是通过，不授权合入。


# 实现偏差确认

- 问题: P2-11
- 方案版本: 用户决定（官方压测并发等于 worker_nums）
- 结论: 同意
- 理由: 接受这一处停住。停住不是本版通过。本版通过结果仍是 `python3 scripts/coverage_http_report.py --reload` 退出码 0：读取现有九行的 `worker_nums` 与 `decode_auto`，按该 pack 拉起后跑 `http_infer_rps.py --raw`，`-c` 等于该行 `worker_nums`，`-d 60s`，`--warmup 5s`；九行全部成功后才覆盖 `docs/perf/coverage_set_http_rps.md`，保留原来的 calibrate 命令，并更新这次官方压测的命令、成功 RPS、p50、p99、错误率、429、两只直方图、JPEG 落点、gateway 五段和 `server_time_ms`。当前清单仍是上一轮 `-c` 等于 `worker_nums` 两倍的九行，没有被覆盖，没有提交。这九行不满足本版「`-c` 不等于该行 `worker_nums` 则不通过」。评分标准 PF1、TP1 要求按结论部署后再用 `--raw` 压测，并发数等于已部署的 `worker_nums`；GPU 作业被跳过时不得写成已满足。PF2、PF4 要同一次压测的排队直方图、推理直方图、gateway 五段和 `server_time_ms`。这次没有这些读数。上次确认要求 `--reload` 退出码 0，并禁止调用 calibrate、重新扫描 worker、手填 `worker_nums`、`decode_auto` 或压测字段。本版做法写明临时 pack 只含这两项，不调用 calibrate。`MORTRED_PACK` 使 `pack_active` 为真。九个 id 都是 TensorRT。`occupancy_ready_for_spawn` 在 `gpu_mem_mib` 缺失或小于等于 0 时拒绝拉起，错误是 `stamp missing for <id> (gpu_mem_mib>0 required for TensorRT)`。只含这两项的 pack 过不了这道门闩。补写 `gpu_mem_mib` 就不再是本版写明的两项 pack。清单点名的九份 pack 里，只有 realesrgan（`gpu_mem_mib=1346`，`worker_nums=4`，`decode_auto=cpu`，`gpu_mem_at_workers=4`）和 metric3d_512（`gpu_mem_mib=1349`，`worker_nums=1`，`decode_auto=gpu`，`gpu_mem_at_workers=1`）还在，戳是 calibrate 写下的。其余七份目录不存在：`mobilenetv2-b3yy_35j`、`yolov8s-r7zbzezu`、`libface-5y7pumc3`、`pphuman_mobile-q2mtq1su`、`dbnet-zb8nyfcy`、`dinov2_vits14-j6juuu5q`、`superpoint-7pokmrlc`。`/tmp` 里其余 pack 要么没有 `gpu_mem_mib`，要么不是这九行的部署结论，不能靠已有戳拉起九行。关掉占用门闩或设置 `MORTRED_OCCUPANCY_ENFORCE=0` 不在本版做法里。这次没有改代码。`--reload` 未执行，60 秒官方压测没有开始。未补写 `gpu_mem_mib`，未手填占用戳，未调用 calibrate，未重新扫描 worker，未改 `scripts/calibrate_pack.py` 的两倍扫描，未执行无参数的 `coverage_http_report.py`，未关占用门闩。在这些限制里，九行按本版拉起不了，`--reload` 到不了退出码 0，清单不能被覆盖。同意的是这次停住。同意停住不是本版通过，不授权合入。方案版本不改。
