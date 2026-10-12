# 本轮新增问题

| 来源 | 问题 | 依据 |
| --- | --- | --- |
| 无 | 无 | 无 |

## 附录

`python3 scripts/product_static_bs1.py --check` 退出码为 1。stderr 是 `mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 必须保持 MNN on cpu，这两份文件的 `type` 已是 `onnx`。P0-1 的验收标准 7 写明此时仍然通过，不插入当前队列。
