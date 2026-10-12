# 第3步确认

- 问题: P1-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 5 已编号，打开文件即可判定。指南示例要执行的 id、`classification_param_specs()` 之后的 family，以及 `model = "densenet"`，复用 `classification_task.h` 里已有的 `densenet` 行和 densenet server toml 的 `model="densenet"`；`server_section` 仍是 `DENSENET_CLASSIFICATION_SERVER`。中文 README 基准测试与 `mortred-model-server.out` 两处 `--model` 为 `mobilenetv2`，全文不再出现 `MOBILENETV2`。两份 README 的 cpu 精选为 `mobilenetv2` 与 `resnet`，全文不再出现 `resnet50`。`docs/deployment.md` 的 cpu 精选仍是 `resnet`。注册行与 `model="densenet"` 保持改前内容，diff 只落在这三份文档。指南仍写明不为每个模型新增服务器二进制，统一入口仍是 `mortred-model-server.out`；中文 README 第一小时命令块仍没有 `mortredctl next`。同一套核对在本机 WSL Ubuntu 跑给出的自测命令，期望退出码为 0。
