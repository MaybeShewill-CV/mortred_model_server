# 第2步确认

- 轮次: 2
- 针对版本: v2
- 结论: 不同意
- 理由: P2-16 要留在队列里。`docs/how_to_add_new_server.zh-cn.md` 的示例仍是 `Entry{"DENSENET", ...}`，参数在 `classification_param_specs()` 结束、没有 family，下文是 `--model DENSENET` 和 `model = "DENSENET"`。`classification_task.h` 与 `densenet_server_config.toml` 的 id 和 family 都是 `densenet`。一次打开这一份就能核对这四处。已关闭的 P1-2 只含两份英文文件，这份中文不在该关闭范围里。范围外排除的是覆盖集以外的 id 进入完成标准，以及逐 id 的 H1 至 H4；这份中文添加入口是在教怎么往已有家族加模型，验收不要求把 `densenet` 放进覆盖集。它有出处、现在的行为和一次外部验收，不够 P0 或 P1 也不能删。P2-12、P2-14、P2-17 的中英并条，附录里的 `yolov5l`、`bisenetv2`、`attentive_gan`，P2-19 去掉一次 `/metrics` 对不齐的 stage 名，以及 P2-20、P2-21 拆成两次运行，都符合入列和一次验收。已关闭 id 没有回到队列。
