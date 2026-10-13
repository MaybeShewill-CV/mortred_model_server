# 第3步确认

- 问题: P2-6
- 方案版本: v3
- 结论: 不同意
- 条件: 实跑把相邻两次查看这四个 pid 的间隔固定为 1 毫秒，并要求 `models_gone` 一定是这一拍：两个模型已不在，gateway 与 supervisor 仍在。现有 `perform_ordered_shutdown()` 在不持连接时给不出一段已知长于这个间隔的停留。方案不许改这段退出，也不许对 `127.0.0.1:8080` 保持连接把 `stop()` 拖长；没有这一拍必须判失败，不能改成两个时间戳小于等于。
- 理由: 有序退出本身成立。`perform_ordered_shutdown()` 先对两个模型发 SIGINT 并 `wait_child_exit`，全部返回后才对 gateway 发 SIGINT；主进程要等 gateway 这段结束才置 `_shutdown_done`。采样看的是 `/proc/<pid>`。`waitpid` 回收时该目录消失，这发生在通知 `wait_child_exit` 和对 gateway 发 SIGINT 之前，所以「两个模型 pid 已不在、gateway 仍在」在进程顺序上会出现一次。0.1314 秒和 0.1336 秒来自一次 `time.sleep(0.002)` 探针：两个模型第一次被看成已消失的时刻，和此后 gateway 第一次被看成已消失的时刻。差是 0.0022 秒，就是这一次 sleep 的间隔。同一次循环里若 gateway 已经不在，两个时刻相同，差为 0。差为 2.2 毫秒只说明有一次采样落在两个模型已不在、gateway 仍在，下一次约 2.2 毫秒后的采样里 gateway 已经不在。停留可以从这次采样之前刚开始，到这次采样之后很快结束，短于 1 毫秒。gateway 侧 `ProcessStop::signal_loop()` 的 `sigtimedwait` 在 SIGINT 已挂起时立刻返回，接着 `server_->stop()`；没有连接时这次 `stop()` 不会等客户端。本机 WSL 上 `time.sleep(0.001)` 的实际间隔约 1.04 到 1.33 毫秒。把名义间隔写成 1 毫秒，不能让验收标准 6 的 `models_gone` 一定出现。同一拍里 gateway 与两个模型都已不在，按方案记为失败。
