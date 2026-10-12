# 第3步确认

- 问题: P1-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 方案 v1 只覆盖列表 v3 里 P1-2 的可观察变化。指南示例的第一项 id、`classification_param_specs()` 之后的 family、以及 `model = "densenet"`，与 `classification_task.h` 里 `densenet` 那一行和 densenet server toml 的 `model="densenet"` 一致；`server_section` 保持 `DENSENET_CLASSIFICATION_SERVER`。中文 README 两处 `--model` 为 `mobilenetv2`，两份 README 的 cpu 精选为 `mobilenetv2` 与 `resnet`。验收标准 1 至 5 打开文件即可判定。自测命令在本机 WSL Ubuntu 的 `/mnt/g/Codex/mortred_model_server` 执行，期望退出码为 0；保留的 `DENSENET_CLASSIFICATION_SERVER` 不会被 `\bDENSENET\b` 判失败。改动面停在这三份文档，catalog、产品 toml、server toml、worker、TensorRT context 与 `docs/deployment.md` 保持原样。
