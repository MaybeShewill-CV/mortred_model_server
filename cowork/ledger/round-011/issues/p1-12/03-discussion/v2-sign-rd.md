# 第3步签字

- 问题: P1-12
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 现有失败文案能对上验收 9 至 20 里交给这些函数的输出。`expect_scores` 的 `score mismatch at` 对应验收 9 的分数破坏；`EXPECT_EQ(output.class_id, …)` 的表达式含 `class_id`，分数未改时不会打出 `score mismatch at`。`expect_boxes` 的 `low IoU for ` 加茎名对应验收 10、12 的 IoU 句；只改 `class_id` 时表达式含 `class_id`；只改 `score` 时 `box.score` / `g["score"]` 含 `score`，IoU 断言不失败，输出不含 `low IoU`。人脸第四次的 `EXPECT_LE(cv::norm(get_landmarks(…)))` 含 `landmarks`，黄金里已有该字段。`expect_text_regions` 同样给出 `low IoU for dbnet_text_detection`，以及只改分数时的 `score`、不含 `low IoU`。`expect_fingerprint`、`expect_keypoints`、`expect_embeddings` 的自定义句分别就是 `fingerprint drift for `、`keypoint match ratio low for `、`embedding cosine similarity low for ` 加茎名；点数不变时不会打出 `keypoint count changed`；`dinov2_vitb14` 不要求句内再写 `0.999`。验收 11、19、20 不走这六个函数的原文案：方案允许另写会打印 `0.5`、`0.01`、`0.999`、`depth error` 与 `0.02` 的比较，且不改现有文案。分割与抠图现有 `mask overlap … 0.90` 已满足验收 13、14。`InferenceSession::create` 在 `type=tensorrt` 且 `device=gpu` 时建 `TrtSession`；名单上的 creator 已在 `src/factory`；`depth_map` 已是拉回原图尺寸的 `CV_32FC1`。`golden_process_exit` 与 `MORTRED_SKIP_OR_FAIL_WEIGHTS` 已能给出退出码 0、1、77 以及 `NUMERICAL_PASSED` / `NUMERICAL_NOT_RUN`。不改进程模型、HTTP 和产品 toml。
