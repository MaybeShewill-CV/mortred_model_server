# 第3步签字

- 问题: P0-3
- 方案版本: v1
- 结论: 不同意
- 条件: 验收 3、4、5 的头个数等于符合下列形状的张量个数：5 维，`[1,3,H,W,C]`，batch 为 1，第二维为 3，H、W、C 都大于 0。dtype、是否有限、通道数是否等于 `class_nums + 5` 不计入这个点数。这三条的样本里，该形状的张量个数分别为 1、2、大于 3，`output`、`518`、`532` 没有同时存在，向量长度大于 0；响应与验收 1 的 HTTP 状态、顶层 `status`、`status_str`、`results` 项数、该项 `status`、该项 `data` 相同，`model.name` 为该 id。形状个数恰好为 3 的样本不算验收 3、4、5 通过。验收 7 写明 `results` 有 1 项，该项 `status` 为 0，该项 `data` 不是 JSON null，`model.name` 为该 id。验收 1 至 7 每条的通过只认写明上述张量的那份响应 JSON。验收 8 的命令输出里，验收 1 至 7 各有一行：验收 1 含 `yolov7`，验收 2 含 `yolov7x`，验收 3 至 7 同时含 `yolov7` 与 `yolov7x`，并且该行含该条期望的 HTTP 状态、顶层 `status` 和 `status_str`。缺少任一行则验收 8 失败。
- 理由: 验收 6 能判定：向量长度为 0 时仍是 HTTP 500、顶层 `status` 5、`status_str` 为 `model output empty`、该项 `data` 为 JSON null。验收 1、2 对「一个该形状的头都没有」的响应字段也能判定。验收 3 至 5 只写「合格头」，验收 7 又把 f32、有限值和 `class_nums + 5` 写进合格。形状已经凑满 3 个、其后通道或 dtype 不符时，响应同样是 HTTP 500、顶层 `status` 6、`status_str` 为 `model output contract failed`。只看这份 JSON，判定不了头个数为 1、2 或大于 3 的样本已经跑过。验收 8 以通过条数不少于 4 和退出码 0 为通过，输出里没有验收 1 至 7 各自的 id、HTTP 状态、顶层 `status` 和 `status_str`。验收开头把这 7 条写成对两个产品路径各发一张图；发图指定不了输出张量里的名字和形状。
