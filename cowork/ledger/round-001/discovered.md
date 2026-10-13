# 本轮新增问题

| 来源 | 问题 | 依据 |
| --- | --- | --- |
| 无 | 无 | 无 |

## 附录

`python3 scripts/product_static_bs1.py --check` 退出码为 1。stderr 是 `mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 必须保持 MNN on cpu，这两份文件的 `type` 已是 `onnx`。P0-1 的验收标准 7 写明此时仍然通过，不插入当前队列。

P1-1 外部验收的五段 `_count` 都是 1，没有覆盖累计超过 4096 条后不再输出主导段。这次五个 `_sum` 没有并列，获胜段是 `upstream`。方案已规定样本上限和并列取靠后名字，外部验收只要求 count 为 1。一次外部验收确认不了这两句，不插入当前队列。

P1-2 指南示例的 display name 仍是 `densenet classification`，源码注册行是 `Densenet classification`。方案写明 display name 不是完成条件。中文 README 第一小时没有 `mortredctl next` 已是 P2-10，不重复入列。

P2-1 外部验收只打了 `yolov8s`、N=2。方案写明其余八个 id 共用这段同步路径，不各做一次外部验收。一次外部验收确认不了每个 id，不插入当前队列。
