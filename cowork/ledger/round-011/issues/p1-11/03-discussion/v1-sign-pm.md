# 第3步签字

- 问题: P1-11
- 方案版本: v1
- 结论: 不同意
- 条件: 把验收 6 改为：`model_golden.bisenetv2_segmentation` 与 `model_golden.pphuman_segmentation` 各做一次。不改 `height`、`width` 和 `values` 长度，把每个整数加 1，加 1 后仍至少两个不同类别值。两条都退出码 1，失败输出含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`。各自恢复原字节后退出码 0。把验收 7 改为：`model_golden.modnet_matting` 与 `model_golden.ppmatting_matting` 各做一次。不改 `height`、`width` 和 `values` 长度。改完后的值仍是 0–255 的整数，前景（值 `>= 128`）像素数至少为 2，且与改前前景没有交集。两条都退出码 1，失败输出含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`。各自恢复原字节后退出码 0。前景像素数小于 2 的改法不能用来做这一条。补上：尺寸不同、类型不符、分割黄金不同类别值少于两个、抠图黄金前景像素数小于 2，这四种失败的输出都不得含 `mask overlap`。只有平均重合或前景重合 `< 0.90` 时，失败输出才含 `mask overlap` 和 `0.90`。测试义务改为：验收 6 的两条分割和验收 7 的两条抠图都要做，不能任选一条。
- 理由: 冻结句要求交付前确认 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 四个 id 都失败于掩码重合低于测试写明的阈值。验收 6、7 各只要求两条里做一条，有两个 id 不会被看到这次失败。验收 7 把前景改成 1 个像素，正文又规定前景像素数小于 2 时直接失败，这次失败落在黄金不合格，落不到重合低于 `0.90`。尺寸、类型不符也没有单独的失败文案，「也不是尺寸或类型不符」无法从输出判定。四个 id、现有 CPU 用例和产品 toml 的边界与冻结句一致，不同意只因为这几条验收判不了冻结句。
