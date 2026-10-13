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
