# 第3步签字

- 问题: P1-12
- 方案版本: v2
- 结论: 不同意
- 条件: 验收 17 和产品经理条件原文里的同一句，要求把 `superpoint` 每个点移到与原点距离大于 `3.0`，再看退出码 1 且输出含 `keypoint match ratio low for superpoint_feature_point`。现有比较用的是黄金点到预测点的最近距离，不超过 `3.0` 算匹配，比例低于 `0.9` 才打出这句。`test/golden/superpoint_feature_point.json` 里的点已经全部离坐标原点超过 `3.0`。只挪开不到 `3.0` 的一小段，字面上仍满足这句，正确实现的退出码是 0，输出里没有这句失败文案。这一条不能一次用退出码和输出判定。测试义务和 RD 不得自行决定都把这句原文和编号 17 当成改法依据，所以这三处都停在这句上。
- 理由: v1 产品经理条件的原文已经放进验收标准，并按编号 9 至 20 展开。测试义务和 RD 不得自行决定都规定改法、黄金文件、退出码和输出文字以这段原文和同一组编号为准。编号 9 至 16、18 至 20 各自对应一条失败文案：分类是 `score mismatch at` 与 `class_id`，检测和 OCR 是 `low IoU for` 与 `score`，人脸另有 `landmarks`，分割和抠图是 `mask overlap` 与 `0.90`，增强是 `fingerprint drift for`，embedding 是 `embedding cosine similarity low`，深度是 `depth error` 与 `0.02`。每条都是一个进程、一个过滤名，看一次退出码和输出就能判定，按原字节恢复后再跑，退出码 0 且输出含 `NUMERICAL_PASSED`。`msocrnet` 不执行过滤；验收 1 和验收 3 只看列表和二进制里的产品 toml 路径。验收 17 是例外。
