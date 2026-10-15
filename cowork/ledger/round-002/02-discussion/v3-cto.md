# 第2步确认

- 轮次: 2
- 针对版本: v3
- 结论: 同意
- 理由: P2-16 留在 P2。`docs/how_to_add_new_server.zh-cn.md` 的示例仍是 `Entry{"DENSENET", ...}`，参数停在 `classification_param_specs()`，没有 family，下文是 `--model DENSENET` 和 `model = "DENSENET"`。`classification_task.h` 与 `densenet_server_config.toml` 的 id 和 family 都是 `densenet`。一次打开这一份就能核对 `model_section`、`--model`、`model=` 和 family。英文 `docs/how_to_add_new_server.md` 已是 `densenet`，留在本条之外。已关闭的 P1-2 只含两份英文文件，仍在附录。覆盖集外的 id 不进入完成标准；这份中文添加入口有出处、现行行为和一次外部验收，不够 P0 或 P1 也留在队列。P2-12、P2-14、P2-17 的中英并条，附录里的 `yolov5l`、`bisenetv2`、`attentive_gan`，P2-19 只验收第 4097 次成功后的五段直方图和 dominant 行，以及 P2-20、P2-21 拆成两次运行，都符合入列和一次验收。已关闭 id 没有回到队列。
