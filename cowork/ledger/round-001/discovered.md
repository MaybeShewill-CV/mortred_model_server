# 本轮新增问题

| 来源 | 问题 | 依据 |
| --- | --- | --- |
| 无 | 无 | 无 |

## 附录

`python3 scripts/product_static_bs1.py --check` 退出码为 1。stderr 是 `mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 必须保持 MNN on cpu，这两份文件的 `type` 已是 `onnx`。P0-1 的验收标准 7 写明此时仍然通过，不插入当前队列。

P1-1 外部验收的五段 `_count` 都是 1，没有覆盖累计超过 4096 条后不再输出主导段。这次五个 `_sum` 没有并列，获胜段是 `upstream`。方案已规定样本上限和并列取靠后名字，外部验收只要求 count 为 1。一次外部验收确认不了这两句，不插入当前队列。

P1-2 指南示例的 display name 仍是 `densenet classification`，源码注册行是 `Densenet classification`。方案写明 display name 不是完成条件。中文 README 第一小时没有 `mortredctl next` 已是 P2-10，不重复入列。

P2-1 外部验收只打了 `yolov8s`、N=2。方案写明其余八个 id 共用这段同步路径，不各做一次外部验收。一次外部验收确认不了每个 id，不插入当前队列。

P2-2 的 shadow 外部序列没有单独量剩余配额，只复跑了原断言：8 次无 Authorization 为 401、0 次 429。admit 成功到归还完成之间的暂扣，验收在收到上一次响应之后才发下一次，序列看不到。方案已规定 shadow 与 enforce 都归还这种 401，并且验收序列是串行的。一次外部验收确认不了这两句，不插入当前队列。

P2-3 的跟踪只挂钩 libc 的 prctl、execvp、setenv、execve。syscall(SYS_prctl) 不出现在日志里，这次没有另测父进程退出后子进程是否收到 SIGINT。方案规定日志里没有 syscall 不影响通过。这条库函数跟踪不在 CI 里。每个直接子进程在终点行之后还有 1 条跟踪，按方案不计入窗口。一次外部验收确认不了这三句，不插入当前队列。

P2-4 没有执行 `ctest -R resident_pair_record_rules`。worktree 没有已配置的 CTest 构建目录，也没有重新编译 C++。第 1 条已经执行同一命令 `python3 scripts/resident_pair_record.py --check`，退出码 0，没有写 json，也没有拉起模型。方案写明无 GPU 的 CI 跑过这条规则不能写成 TP2 已满足。一次外部验收确认不了 CTest 注册是否被默认 check 调度，不插入当前队列。

P2-5 外部脚本按方案不启动 supervisor。`__gateway` 永久失败和管理面继续监听只在 `supervisor_unittest` 里看到。一次外部验收确认不了管理面进程，不插入当前队列。

P2-6 没有执行 `ctest -R stop_order_record_rules`。worktree 没有已配置的 CTest 构建目录，也没有重新编译 C++。第 1 条已经执行同一命令 `python3 scripts/stop_order_record.py --check`，退出码 0，没有写 json，也没有拉起模型。方案写明无 GPU 的 CI 跑过这条规则不能写成 OP5 已满足。一次外部验收确认不了 CTest 注册是否被默认 check 调度，不插入当前队列。

P2-7 没有发送正好 256×256 的成功请求。方案规定成功增强图不作为通过。TensorRT 把已有引擎 `realesr-general-x4v3.engine.static_bs1.fp32` 记成 H/W 动态，profile 为 min 64、opt 256、max 640。验收标准 5 只核对 toml 与基准文件中的字符串，这两处已通过，该 toml 不在本次 diff 里。一次外部验收确认不了引擎剖面是否仍是静态 256，不插入当前队列。

P2-9 没有重新跑推理。方案规定 golden 数值与当前权重是否仍一致不在通过条件内。标准 4 没有改文件去制造不一致。一次外部验收确认不了权重是否仍与 golden 数值一致，不插入当前队列。

P2-10 没有执行 `mortredctl`。方案规定不以执行该命令作为通过。`scripts/mortredctl_next.sh` 与改动前相同。一次外部验收确认不了脚本打印的下一条命令，不插入当前队列。

P2-11 的清单已合入 `9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32`。四张指定 JPEG 仍未跟踪：`face_w_mask.jpg`、`coco_text_test.jpg`、`feature_point/test.jpg`、`0000000005.jpg`。realesrgan 的 256×256 JPEG 留在已忽略的 `logs/p2-11-realesrgan-256.jpg`。文件头日期是 2026-10-14，superpoint、realesrgan、metric3d_512 三行的日期是 2026-10-15。`http_infer_rps.py` 在测量中复用连接，并由客户端先关闭。这些句子一次外部验收确认不了新的产品行为，不插入当前队列。

