# 第2步确认

- 轮次: 2
- 针对版本: v2
- 结论: 不同意
- 理由: P2-16 有出处和现行行为，一次打开 `docs/how_to_add_new_server.zh-cn.md` 就能核对 `model_section`、`--model`、`model=` 和 family。按这份指南启动会得到 `unknown --model`。它不够 P0 或 P1，应留在 P2。移出理由写成 densenet 不在覆盖集，但这不是范围外：范围外是不把覆盖集外的 id 写成完成标准，也不再要求逐 id 做齐旧基准。这份中文指南是中文 README 指向的扩展入口。它也不是已关闭 P1-2 的重复描述，P1-2 只含两份英文文件，不能回到队列。其余可以留下：`yolov5l`、`bisenetv2`、`attentive_gan` 留在附录；中英文同句已并成一次打开两份文档的验收；`stage` 是否为最大 p50 一次读 `/metrics` 对不齐，已移出；没有 `TRT_PROOF=yes` 和出现该行时核对 gtest XML，已拆成两次运行。官方压测并发是已部署 `worker_nums` 的两倍，不据此否定这一版。
