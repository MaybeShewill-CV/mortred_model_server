# 第3步签字

- 问题: P1-12
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收 5 至 13 不改现有比较文案也能一次打出。`expect_scores` 的 `EXPECT_EQ` 打出 `230` 与 `170`，分数失败句是 `score mismatch at`，且 `EXPECT_EQ` 不中断，实测分数差超过 `1e-3` 时两段同时出现。`expect_boxes` 的 `ASSERT_EQ` 打出框数 `5`/`7` 和 `detection count changed for` 加茎名。`yolov7` 的分数比较是 `EXPECT_NEAR(..., k_det_score_tol)`，本机 gtest 1.11 把断言流精度设为 17，`1e-2` 印成 `0.01`，差也印在同一句；框数一致时不会出现 `detection count changed`。`libface` 的 `EXPECT_LE(cv::norm(get_landmarks(...)))` 表达式含 `get_landmarks`，距离由该断言打出。`expect_fingerprint` 的失败句含 `fingerprint drift for `、`mean abs diff`，并流式打出均值。`expect_embeddings` 的失败句含 `embedding cosine similarity low for ` 加茎名；`EXPECT_GE` 用默认 6 位有效数字打出两个操作数，阈值 `0.999` 印成 `0.999`，实测余弦约 `0.126` 的这串数字也在这行里。
