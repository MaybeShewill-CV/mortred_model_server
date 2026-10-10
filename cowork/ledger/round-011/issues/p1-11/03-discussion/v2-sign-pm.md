# 第3步签字

- 问题: P1-11
- 方案版本: v2
- 结论: 不同意
- 条件: 把验收 9 改成下面三组，都在仓库根目录用 `<bin>`。每次只改这一条的 JSON，失败输出看完后恢复原字节，再跑同一条过滤，退出码 0。三组的失败退出码都是 1，失败输出都不含 `mask overlap`。尺寸不同：`model_golden.bisenetv2_segmentation`、`model_golden.pphuman_segmentation`、`model_golden.modnet_matting`、`model_golden.ppmatting_matting` 各做一次；把 `height` 改成另一个正整数，并把 `values` 改成新的 `height * width` 长度；分割的 `values` 仍至少两个不同类别值，抠图的值仍是 0–255 的整数且前景像素数至少为 2。分割类别不足：两条分割各做一次；不改 `height`、`width` 和 `values` 长度，把 `values` 全部写成同一个整数。抠图前景不足：两条抠图各做一次；不改 `height`、`width` 和 `values` 长度，把每个值改成 0–127，使前景像素数小于 2。`类型不符` 从验收、边界、契约和测试义务里去掉：它看的是这次模型输出的类型，改黄金 JSON 和改命令都触发不了，方案又不许改输出类型，验收者看不到这次失败。测试义务改为：验收 9 的三组都要做，组内每条都要做，做完恢复原字节。
- 理由: 验收 6、7 已经能让 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 各自失败于掩码重合低于 `0.90`，和冻结句一致。验收 9 仍只写了失败文案规则，没有命令、改哪一份黄金、退出码和恢复。尺寸、类别数、前景像素数可以改 JSON 后看到输出；类型不符改不了，验收者无法判定这条有没有做。四个 id、现有 CPU 用例和产品 toml 仍是 `tensorrt` 的边界与冻结句一致。
