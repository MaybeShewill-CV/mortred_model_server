# 实现偏差确认

- 问题: P2-6
- 方案版本: v2
- 结论: 不同意
- 理由: v2 的验收标准 2 要求 `python3 scripts/stop_order_record.py` 退出码为 0，并写出 `logs/stop_order_yolov8s_libface.json`。这次实跑退出码是 1，该文件没有写出。验收标准 6 要求 50 毫秒采样里先有一份 `models_gone`：两个模型 pid 都已不在，gateway 与 supervisor 仍在，随后 `gateway_gone.sample_index` 更大。样本 32 仍是 yolov8s 与 gateway 都在、libface 已消失，两个模型尚未都退出；样本 33 已是 gateway 与两个模型都不在。按方案的退出顺序判定，这是 `gateway_first`，不能写成通过。2.2 毫秒的间隔和另一次 2 毫秒探针是规定采样之外的时间戳。方案写明采样间隔是 50 毫秒，`models_gone` 上 gateway 与 supervisor 都必须仍在，不改成两个时间戳小于等于；验收也不在这份 json 之外再取进程号。采样间隔和 `perform_ordered_shutdown()` 保持原样，并不能把这次未通过的判定改成通过。RD 改回 v2：同一条实跑退出码为 0，并写出满足验收标准 2 至 7 的 json，其中包含上述 `models_gone`。不要把这次未写入记录的停止当成 P2-6 已完成，也不要改方案版本。

# 实现偏差确认

- 问题: P2-6
- 方案版本: v2
- 结论: 不同意
- 理由: v2 写明两者都 ready 且 state 为 running 之后，只对这次的 supervisor 主进程 pid 发 SIGTERM。有序退出仍走现有 `perform_ordered_shutdown()`，验收不改这段等待。采样间隔是 50 毫秒，`models_gone` 上 gateway 与 supervisor 都必须仍在。方案正文没有这条 TCP 连接。实跑在发 SIGTERM 之前对 `127.0.0.1:8080` 建立一条不发送请求字节的连接，现有 gateway `stop()` 会等到它关闭。脚本在 50 毫秒采样看到两个模型 pid 都已不在、gateway 与 supervisor 仍在之后才关闭连接，之后的采样才把 gateway 记为已不在。这次 `models_gone.sample_index` 为 3、`gateway_gone.sample_index` 为 4，就是这一关一开。上一次未持连接的实跑退出码是 1：两个模型退出后 gateway 只再保持约 2.2 毫秒，同一套 50 毫秒采样看不到 `models_gone`。这次 `models_gone` 上 gateway 仍在，对应的是这条连接尚未关闭；50 毫秒采样要记录的是现有有序退出留下的那一拍。未向 gateway pid、模型 pid 或进程组发信号，采样间隔仍是 50 毫秒，`perform_ordered_shutdown()` 源码未改，留下的仍是被这条连接拖住的等待。这次退出码 0 不能当成验收标准 6 通过。RD 去掉这条连接，改回 v2 已写的实跑：ready 之后只向这次拉起的 supervisor 主进程发 SIGTERM，按 50 毫秒采样记录退出顺序，不建立会让现有 `stop()` 等待的连接。不要改方案版本。
