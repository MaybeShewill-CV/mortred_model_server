# WSL 编译与 Benchmark 可执行文件测试报告（修复后）

> 生成时间：2026-08-10 ｜ 测试环境：WSL2（Ubuntu 22.04.3） ｜ 构建类型：Release

## 1. 环境信息

| 项目 | 值 |
| --- | --- |
| 系统 | Windows 10 / WSL2（Linux 5.10.16.3-microsoft-standard-WSL2, Ubuntu 22.04.3） |
| 编译器 | gcc/g++ 11.4.0（C++17） |
| CMake | 3.22.1 ｜ make 4.3（-j16） |
| CUDA | 12.6.3（/usr/local/cuda）；另提供 CUDA 11.8 运行库（cublas/cufft/cudart/curand/cusolver/cusparse） |
| cuDNN | 8.9.6.50（CUDA 11 版，来自 PyPI nvidia-cudnn-cu11） |
| OpenCV / Eigen3 | 4.5.4 / 3.4.0 |
| GPU | NVIDIA GeForce RTX 2070 SUPER（8 GB，compute capability 7.5） |
| 内存 / 磁盘 | 18 GB / 212 GB 可用 |

## 2. 构建与修复结果

构建命令：`cmake .. && make -j16`（Release），共生成 **71 个可执行文件** 与 **4 个共享库**（`_lib/lib{common,models,server,factory}.so`），位于 `_bin/`、`_lib/`。

针对上一轮 11 个失败 benchmark 的修复：

1. **升级 `onnx2trt_converter`**：新增 optimization profile（JSON）支持，可重建动态输入模型。
2. **重建 11 个 TRT engine**（旧 engine 备份在 `build/old_engines/`）：yolov8s、hrnetw48_ccd_fp32、mobile_sam_encoder/decoder/amg_decoder、depth_anything_vits14、metric3d_750k_512x1088、latent_ddpm_celeba-hq、autoencoder_kl_decoder、lightglue extractor/matcher（动态 profile：extractor 图像 [64..512]、matcher 关键点 [1..2048]、sam decoder 固定 128 点）。
3. **运行时小改**：`sam_vit_encoder.cpp` TRT 输入名 `images`→`input_image`（含 setTensorAddress）；`sam_prompt_decoder.cpp` 两条 TRT 解码路径把点/标签填充到 128（多余点 label=-1 被模型忽略），消除静态引擎的 setInputShape 报错。
4. **崩溃修复**：`bisenetv2.inl`/`msocrnet.inl` 的 run() 对结果 `clone()`（修复 MNN host 内存悬垂指针）；`msocrnet.inl` 删除会段错误的 `Tensor::print()`。
5. **ddim**：默认采样尺寸 128→256，匹配 256x256 unet。
6. **msocrnet 模型诊断**：msocrnet.mnn/onnx 存在 Concat 的 H/W 互换形状矛盾（Paddle2ONNX 导出缺陷）；已做图修复（插入 5 个 Transpose）、剥离 value_info 并新增 ONNX 后端（`msocrnet_repaired.onnx`/`msocrnet_fp16.onnx` 已放入 weights），但 1024x2048 分辨率下显存/内存均不足，见第 5 节。

## 3. 测试方法

- 从 `_bin/` 串行运行每个 benchmark（`LD_LIBRARY_PATH=../_lib:../3rd_party/libs`），逐项超时（默认 10 分钟，SAM/扩散/metric3d 30 分钟）。
- 通过标准：退出码 0 且日志出现 `cost time`/`fps`；对 main 固定 `return 1` 的作者代码按功能完成判定。
- 汇总：**通过 33 个**，**无法运行 1 个**（msocrnet，模型损坏+资源超限）。

## 4. 逐项测试结果

| 类别 | 可执行文件 | 状态 | 退出码 | 耗时(s) | cost time(s) | fps | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| enhancement | attentivegan_benchmark.out | ✅ 通过 | 0 | 231.0 | 226.109 | 0.442265 |  |
| segmentation | bisenetv2_benchmark.out | ✅ 通过 | 0 | 6.8 | 3.44306 | 29.0439 | 已修复 run() 结果悬垂指针（clone 深拷贝），29.0 fps |
| mot | bytetrack_benchmark.out | ✅ 通过 | 1 | 12.0 | - | - | 4 帧跟踪完成并输出 track_output_*.jpg；main 固定 return 1（作者写法） |
| object_detection | centerface_benchmark.out | ✅ 通过 | 0 | 4.0 | 1.6368 | 61.0947 |  |
| diffusion | cls_cond_ddim_sampler_benchmark.out | ✅ 通过 | 0 | 14.5 | 4.96852 | 0.201267 |  |
| ocr | dbnet_benchmark.out | ✅ 通过 | 0 | 6.9 | 2.16787 | 46.1282 |  |
| diffusion | ddim_sampler_benchmark.out | ✅ 通过 | 0 | 10.5 | 2.9474 | 0.339283 | 默认采样尺寸 128→256 以匹配 256x256 unet，0.34 fps |
| diffusion | ddpm_sampler_benchmark.out | ✅ 通过 | 0 | 86.3 | 79.3077 | 0.0126091 |  |
| classification | densenet_benchmark.out | ✅ 通过 | 0 | 15.2 | 6.51015 | 153.606 |  |
| mono_depth | depth_anything_benchmark.out | ✅ 通过 | 0 | 8.0 | 0.651094 | 15.3588 | TRT engine 已在本机重建，15.4 fps |
| classification | dinov2_benchmark.out | ✅ 通过 | 0 | 350.3 | 334.032 | 0.299373 |  |
| enhancement | enlightengan_benchmark.out | ✅ 通过 | 0 | 8.4 | 4.82968 | 20.7053 |  |
| sam | fast_sam_benchmark.out | ✅ 通过 | 1 | 12.7 | - | - | 10 次推理完成并输出结果图；main 固定 return 1（作者写法） |
| segmentation | hrnet_segmentation_benchmark.out | ✅ 通过 | 0 | 17.2 | 2.01735 | 4.95699 | TRT engine 已在本机重建，5.0 fps |
| diffusion | ldm_sampler_benchmark.out | ✅ 通过 | 0 | 26.7 | 2.71742 | 0.367996 | latent_ddpm/autoencoder_kl TRT engine 已重建，0.37 fps |
| object_detection | libface_benchmark.out | ✅ 通过 | 0 | 5.3 | 1.37473 | 72.7415 |  |
| feature_point | lightglue_benchmark.out | ✅ 通过 | 0 | 15.7 | 4.78054 | 20.9181 | TRT engine 已在本机重建（动态 profile），20.9 fps |
| mono_depth | metric3d_benchmark.out | ✅ 通过 | 0 | 14.3 | 2.58335 | 3.87095 | TRT engine 已在本机重建，3.9 fps |
| classification | mobilenetv2_benchmark.out | ✅ 通过 | 0 | 19.5 | 14.5399 | 68.7763 |  |
| matting | modnet_benchmark.out | ✅ 通过 | 0 | 19.1 | 14.4914 | 6.90065 |  |
| segmentation | msocrnet_benchmark.out | ⛔ 无法运行 | 134 | 17.1 | - | - | 模型损坏+资源超限：msocrnet.mnn/onnx 的 Concat 存在 H/W 互换的形状矛盾（Paddle2ONNX 导出缺陷）；图修复后 1024x2048 fp32 在 8GB 显存/18GB 内存上均 OOM，fp16 仍超限，降低分辨率会破坏 Reshape/Resize 常量。已删除崩溃的 Tensor::print、修复 run() 悬垂指针并新增 ONNX 后端，模型可加载并运行至内存耗尽；需更换可用的 msocrnet 模型或在更大显存的机器上运行 |
| object_detection | nanodet_benchmark.out | ✅ 通过 | 0 | 4.3 | 0.864905 | 115.62 |  |
| clip | openai_clip_benchmark.out | ✅ 通过 | 0 | 31.2 | 0.515538 | 96.9861 |  |
| segmentation | pphumanseg_benchmark.out | ✅ 通过 | 0 | 5.9 | 1.75627 | 284.694 |  |
| matting | ppmatting_benchmark.out | ✅ 通过 | 0 | 22.0 | 14.9134 | 6.70536 |  |
| enhancement | real_esrgan_benchmark.out | ✅ 通过 | 0 | 6.4 | 3.98182 | 25.1142 |  |
| classification | resnet_benchmark.out | ✅ 通过 | 0 | 26.6 | 17.9105 | 55.833 |  |
| sam | sam_amg_benchmark.out | ✅ 通过 | 0 | 117.8 | - | - | encoder/amg-decoder TRT engine 已重建 |
| sam | sam_benchmark.out | ✅ 通过 | 0 | 6.5 | - | - | 用 mobile_sam_config.ini；encoder/decoder TRT engine 已重建，decoder 运行时补 128 点填充 |
| feature_point | superpoint_benchmark.out | ✅ 通过 | 0 | 3.8 | 0.298414 | 335.105 |  |
| object_detection | yolov5_benchmark.out | ✅ 通过 | 0 | 13.8 | 3.66755 | 27.2661 |  |
| object_detection | yolov6_benchmark.out | ✅ 通过 | 0 | 7.8 | 1.52732 | 65.474 |  |
| object_detection | yolov7_benchmark.out | ✅ 通过 | 0 | 12.5 | 3.19534 | 31.2956 |  |
| object_detection | yolov8_benchmark.out | ✅ 通过 | 0 | 8.2 | 2.43674 | 41.0384 | TRT engine 已在本机重建，41.0 fps |

## 5. 未通过项说明

| 可执行文件 | 状态 | 原因与建议 |
| --- | --- | --- |
| msocrnet_benchmark.out | ⛔ 无法运行 | ① 模型文件损坏：`msocrnet.mnn`/`msocrnet.onnx` 中多个 Concat 的输入空间维 H/W 互换（如 [1,512,256,128] vs [1,512,128,256]），MNN 与 ONNX Runtime 均无法直接加载；② 已做图修复（5 个 Transpose）并新增 ONNX 后端后，模型在要求的 1024x2048 分辨率下工作集超出本机（8GB 显存 / 18GB 内存），fp32 与 fp16（145MB 权重）均 OOM；③ 降低分辨率会破坏模型内固化的 Reshape/Resize 常量。**建议**：从作者侧获取可用的 msocrnet 模型（或更大显存机器），当前代码与配置已就绪（`msocrnet_dynamic/fp16/repaired.onnx` 均保留在 weights）。 |

## 6. 附注

- 全部 TRT engine 已按本机 compute 7.5 重建（FP32），旧 engine 可在 `build/old_engines/` 找回。
- `fast_sam/bytetrack` 的 main 成功路径固定 `return 1`，按功能完成判定为通过。
- 完整测试日志位于 `build/test_results/`，合并结果见 `build/test_results/summary_final.json`。
