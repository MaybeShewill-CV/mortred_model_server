# 第3步确认

- 问题: P1-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `classification_task.h` 里 densenet 那一行已经是 id `densenet`、family `densenet`，`CvModelEntry` 的最后一项就是 family；`--model` 区分大小写并对齐 `model_section`。densenet server toml 已是 `model="densenet"`，`docs/deployment.md` 的 cpu 精选已是 `curated: mobilenetv2, resnet`。指南里的 `DENSENET`、中文 README 的两处 `--model MOBILENETV2`、两份 README 各一处 `resnet50` 都在方案点名的位置。按方案改完后，自测里的 Entry、family、保留的 `DENSENET_CLASSIFICATION_SERVER`、`model = "densenet"` 和精选表都能对上，源码、toml 和 deployment 不用动。
