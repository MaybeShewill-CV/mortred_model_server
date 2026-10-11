# 第3步签字

- 问题: P1-11
- 方案版本: v2
- 结论: 不同意
- 条件: 把验收 9 拆成下面三步，退出码都是 1，失败输出不含 `mask overlap`，恢复原字节后再跑，退出码 0。尺寸不同：`model_golden.bisenetv2_segmentation` 与 `model_golden.modnet_matting` 各做一次。改 `height` 或 `width`，同时把 `values` 长度改成新的 `height * width`，新尺寸与该条输出不同。分割黄金不同类别值少于两个：`model_golden.bisenetv2_segmentation` 与 `model_golden.pphuman_segmentation` 各做一次。不改 `height`、`width` 和 `values` 长度，把 `values` 全部写成同一个整数。抠图黄金前景像素数小于 2：`model_golden.modnet_matting` 与 `model_golden.ppmatting_matting` 各做一次。不改 `height`、`width` 和 `values` 长度，把 `values` 全部写成小于 128 的整数。类型不符只留在白盒：核对该失败路径的输出不含 `mask overlap`。边界、契约、测试义务和「RD 不得自行决定」与这三步一致。生成的抠图黄金里，值小于 128 的像素至少为 2。
- 理由: v1 的条件已经写进验收 6、验收 7 和测试义务。四个 id 都要在掩码重合低于 `0.90` 时失败，现有 CPU 用例、阈值、产品 toml 和未合入的 P1-8 符号也与冻结句一致。验收 9 仍不能一次黑盒判定。类型不符查的是结果 Mat 的类型；`bisenetv2` 与 `pphuman_mobile` 的 `segmentation_result` 已是 `CV_32SC1`，`modnet` 的 `matting_result` 已是 `CV_8UC1`，`ppmatting_512` 已是 `CV_32SC1`。改黄金 JSON 走不进这条失败，方案又禁止改模型输出类型。尺寸不同、类别少于两个、前景少于 2 没有写改哪一份文件、怎么改、跑哪条、退出码是多少。只改 `height` 或 `width` 而让 `values` 长度对不上，会落到正文没有文案的失败。验收 7 要求改完后的前景与改前前景没有交集且至少 2 个像素，这要求原黄金里至少有 2 个值小于 128 的像素；正文只要求前景至少 2 个。
