# 第3步确认

- 问题: P2-12
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 现有查找已经按大小写敏感的 catalog `model_section` 工作。`classification_task.h`、`feature_point_task.h` 的 id 与 family 是 `mobilenetv2`、`superpoint`，两份 server toml 的 `model=` 和表名也是这两个小写 id，`model_server_main.cpp` 对不上时打印 `unknown --model`。两份教程里对应命令现在是 `--model MOBILENETV2`、`--model SUPERPOINT`，续行和 toml 路径与方案自测的两行字符串一致，文件是 LF。只把这四个实参改成小写即可对上现有注册，不需要改进程、后端、catalog 或 toml。验收和自测只读文件，不要求 GPU、calibrate 或压测。官方 HTTP 压测并发仍是已部署 `worker_nums` 的两倍，方案不改这条规则。节标题、`YOLOV5`、`BISENETV2`、`ATTENTIVE_GAN_DERAIN`、`worker_nums` 原句和 `densenet` 都不进入这次改动。
