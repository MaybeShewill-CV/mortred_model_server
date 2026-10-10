# 第3步签字

- 问题: P1-8
- 方案版本: v2
- 结论: 不同意
- 条件: 验收 6 的第二次只改该框的 `x`、`y`、`w`、`h`，不改框数、`class_id` 和 `score`。新位置与原框的 IoU 要大于 0 且小于 0.5，并且大于该原框与同一份黄金里其余每个框的 IoU。失败输出含 `low IoU`。不要移到与所有框 IoU 都为 0 的位置，也不要移到与另一个框 IoU ≥ 0.5 的位置。验收 8 写明黄金文件是 `test/golden/<id>.json`，编码是 UTF-8 JSON，并写明向量数组的键名。只把该数组换成等长、每个分量取相反数的数组，不改键名和长度。退出码 1，失败输出含 `embedding cosine similarity low`，而不是缺黄金或长度不符；改回后退出码 0。验收 10 写明被破坏的文件是 `test/golden/bisenetv2_segmentation.png`、`test/golden/pphuman_segmentation.png`、`test/golden/modnet_matting.png`、`test/golden/ppmatting_matting.png` 之一。文件仍能按原尺寸打开，整图涂成另一种颜色；不要删除文件，不要换成空文件或另一尺寸。该条退出码 1，失败输出含 `fingerprint drift`。
- 理由: 上一版要求的验收 6 前半、验收 7、验收 9 和验收 11 已经能对上固定输出。验收 6 的后半仍不能一次判定：按「移到与原框 IoU < 0.5」去做，框可以和所有框都不相交，也可以撞上另一个框，失败输出就不是 `low IoU`。验收 8 没有写黄金路径、编码和数组键，换错结构会先变成缺黄金或长度不符，看不到余弦门。验收 10 只写「破坏 png」，删文件、空文件或改尺寸的失败输出不是 `fingerprint drift`。这三处补进正文后才能签字。
