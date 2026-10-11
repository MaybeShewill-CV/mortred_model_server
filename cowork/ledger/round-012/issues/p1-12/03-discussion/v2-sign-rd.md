# 第3步签字

- 问题: P1-12
- 方案版本: v2
- 结论: 不同意
- 条件: `_bin/model_golden_test --gtest_filter=TrtProductGolden.realesrgan` 退出码 0，标准输出含 `product toml conf/model/enhancement/real_esrgan/realesrgan.toml type=tensorrt device=gpu model_file_path=weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32` 和 `NUMERICAL_PASSED`。
- 理由: 契约表把 `realesrgan` 的 engine 写成 `weights/enhancement/real_esrgan/real-esr-general-x4v3.engine.static_bs1.fp32`。`conf/model/enhancement/real_esrgan/realesrgan.toml` 的 `model_file_path` 剥掉 `../` 之后是 `weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32`。验收 2 要求 toml 文本含契约表里的 engine，产品行又定义为契约表那一行全文；方案规定不改 `conf/model/**`。按剥路径打印出来的是 `realesr-general`，按契约表打印出来的是 `real-esr-general`，验收 2 和这条产品行不能同时通过。其余 37 个 id 的 toml、engine、现有比较函数、`golden_process_exit` 和深度 `depth_map`（`CV_32FC1`、拉回原图尺寸）可以对上。
