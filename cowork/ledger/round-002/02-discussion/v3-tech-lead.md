# 第2步确认

- 轮次: 2
- 针对版本: v3
- 结论: 同意
- 理由: P2-16 已回到 P2。一次打开 `docs/how_to_add_new_server.zh-cn.md` 就能核对示例的 `model_section`、`--model`、`model=` 和 family 都是 `densenet`。它有出处和现行行为，不够 P0 或 P1，留在 P2。中文添加入口是 `README.zh-cn.md` 指向的扩展步骤，验收不把 `densenet` 放进覆盖集。已关闭的 P1-2 仍只含两份英文文件，留在附录。P2-12、P2-14、P2-17 各是一次打开两份文档。`yolov5l`、`bisenetv2`、`attentive_gan` 在覆盖集外，留在附录。P2-19 只验收第 4097 次成功后五段直方图和一行 dominant 仍在。P2-20 与 P2-21 是两次运行。已关闭 id 没有回到队列。官方压测并发是已部署 `worker_nums` 的两倍，附录里的 P2-11 按这个写，不据此否定这一版。
