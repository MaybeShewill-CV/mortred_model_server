# 第2步确认

- 轮次: 2
- 针对版本: v3
- 结论: 同意
- 理由: P2-12、P2-14、P2-16、P2-17、P2-19、P2-20、P2-21 都有场景、现在的行为，以及一次打开文档或一次运行就能判定的变化。P2-16 已回到 P2：只打开 `docs/how_to_add_new_server.zh-cn.md`，示例的 `model_section`、`--model`、`model=` 和 family 都是 `densenet`。它不够 P0 或 P1，但满足入列；`densenet` 不在覆盖集，只说明不为它另做完成线。P2-13、P2-15、P2-18 是同一句的重复描述。`yolov5l`、`bisenetv2`、`attentive_gan` 在覆盖集外。`stage` 是否为精确最大 p50，一次读 `/metrics` 对不齐，已移出。P2-20 与 P2-21 各是一次运行。已关闭的 P0-1、P1-1、P1-2、P2-1 至 P2-11，以及范围外的 P2-8，没有回到队列。
