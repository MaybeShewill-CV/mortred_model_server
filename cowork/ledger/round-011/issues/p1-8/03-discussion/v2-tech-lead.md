# 技术方案

- 问题: P1-8
- 版本: v2
- 对应列表: v3

## 边界

只给下面 19 个 catalog id 各加一条 CPU 数值用例：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。

用例挂在现有 `model_golden_test` 里，套件名 `model_golden`，测试名就是 catalog id。比较在进程内完成：读 CI 用的 CPU 配置，建对应家族已有的模型，对一张固定输入图 `run`，再和 `test/golden/` 里该 id 自己的黄金比较。

上一版审查条件已写入验收 6、7、9、11。

`yolov8s` 已关闭的同类 IoU 门仍只覆盖 `yolov8s`。`model_golden.yolov8_detection` 与 `model_golden.yolov8_onnx_detection` 继续加载 `yolov8s.onnx`。`yolov8n`、`yolov8l`、`yolov8x` 各自一条用例、各自一份黄金、各自一个 onnx。

缺权重继续走现有门闩。单条缺权重仍跳过，跳过文案含 `weights not available`。选中用例全部因此跳过时，进程退出 77，标准输出一行 `NUMERICAL_NOT_RUN`。设置 `MORTRED_CI_REQUIRE_WEIGHTS=1` 时，同一处改为失败，进程退出 1，并且不打印 `NUMERICAL_NOT_RUN`。缺黄金、比较失败，都是失败，退出码 1。静默通过不满足本条。

不覆盖 P1-11、P1-12、P0-2、P1-9、P1-10、D-8 至 D-12。不重开已关闭原句。

## 做法

### 推荐

在现有 `model_golden_test` 里登记 19 条 CPU 用例。检测和 embedding 沿用已经满足 H2 的比较。分割、抠图、深度为这 19 条里的对应 id 另加比较，不替换现有 fingerprint 比较。

每条用例走现有七步：`weights_available` 失败则 `MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available")`；通过后 `load_model_cfg`（去掉 `../` 前缀并强制 cpu），用该家族现有的 creator `init`、读图、`run`、比较。不新写第二套退出码。

配置不读 `conf/model` 下的产品 toml。产品文件的 `type` 是 `tensorrt`，路径是 `.engine`。新用例使用 `conf/ci/` 里 `type` 为 `onnx`、`device` 为 `cpu` 的 toml，`model_file_path` 指向 `conf/product_static_bs1.json` 里该 id 的 `onnx`（文件内写成 `../weights/...`，与现有 CI toml 相同）。`params` 抄该 id 产品 toml 的 `params`，包括类别名、阈值、输入尺寸、`msocrnet` 的 `model_input_image_size`、两个 Metric3D 各自的 `intrinsic` 与 `canonical_size`、两份 PP-HumanSeg 的 `input_names` 与 `output_names`。段名用该 creator 已经在读的段：`NanoDet`、`YOLOV7`、`YOLOV8`、`PP_HUMANSEG`、`HRNET_SEGMENTATION`、`MSOCRNET`、`DINOV2`、`PP_MATTING`、`DEPTH_ANYTHING`、`METRIC3D`。

`dinov2_vits14` 用已有的 `conf/ci/dinov2_vits14_onnx_hosted.toml`。`dinov2_vitl14` 用已有的 `conf/ci/dinov2_vitl14_onnx_hosted.toml`。其余 17 个 id 各新建一份 `conf/ci/<id>_onnx_hosted.toml`。

| id | creator | 输入图 | onnx |
| --- | --- | --- | --- |
| `nanodet_416` | `create_nanodet_detector` | `demo_data/model_test_input/object_detection/bus.jpg` | `weights/object_detection/nanodet/nanodet_plus_m_416.static_bs1.onnx` |
| `yolov7x` | `create_yolov7_detector` | 同上 `bus.jpg` | `weights/object_detection/yolov7/yolov7x.static_bs1.onnx` |
| `yolov8n` | `create_yolov8_detector` | 同上 | `weights/object_detection/yolov8/yolov8n.static_bs1.onnx` |
| `yolov8l` | 同上 | 同上 | `weights/object_detection/yolov8/yolov8l.static_bs1.onnx` |
| `yolov8x` | 同上 | 同上 | `weights/object_detection/yolov8/yolov8x.static_bs1.onnx` |
| `pphuman_lite` | `create_pphuman_segmentor` | `demo_data/model_test_input/scene_segmentation/human_image.jpg` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.static_bs1.onnx` |
| `pphuman_server` | 同上 | 同上 | `weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.static_bs1.onnx` |
| `hrnet` | `create_hrnet_segmentor` | `demo_data/model_test_input/scene_segmentation/cityscapes_test.png` | `weights/scene_segmentation/hrnet/hrnetw48_ccd.static_bs1.onnx` |
| `msocrnet` | `create_msocrnet_segmentor` | `demo_data/model_test_input/ocr/railway_ticket.png` | `weights/scene_segmentation/msocrnet/msocrnet_fp16.static_bs1.onnx` |
| `dinov2_vits14` | `create_dinov2_feature_extractor` | `demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG` | `weights/classification/dinov2/dinov2_vits14_pretrain.static_bs1.onnx` |
| `dinov2_vitl14` | 同上 | 同上 | `weights/classification/dinov2/dinov2_vitl14_pretrain.static_bs1.onnx` |
| `ppmatting_1024` | `create_ppmatting_segmentor` | `demo_data/model_test_input/matting/matting_test.jpg` | `weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.static_bs1.onnx` |
| `ppmatting_resnet34` | 同上 | 同上 | `weights/matting/ppmatting/pp_humanmatting-resnet34_vd.static_bs1.onnx` |
| `ppmatting_v2` | 同上 | 同上 | `weights/matting/ppmatting/ppmattingv2_stdc1_human_512.static_bs1.onnx` |
| `depth_vits14` | `create_depth_anything_estimator` | `demo_data/model_test_input/mono_depth_estimation/0000000005.png` | `weights/mono_depth_estimation/depth_anything/depth_anything_vits14.static_bs1.onnx` |
| `depth_vitb14` | 同上 | 同上 | `weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.static_bs1.onnx` |
| `depth_vitl14` | 同上 | 同上 | `weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.static_bs1.onnx` |
| `metric3d_512` | `create_metric3d_estimator` | 同上 | `weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.static_bs1.onnx` |
| `metric3d_1088` | 同上 | 同上 | `weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.static_bs1.onnx` |

比较规则：

- 五个检测 id 沿用现有框比较：框数一致，配对后 `class_id` 相同，IoU ≥ 0.5，分数差不超过现有 0.01。IoU 低于 0.5 时失败输出含 `low IoU`。类别不同则该断言失败。黄金是该用例名下的框 json，互不共用，也不共用 `test/golden/yolov8_detection.json`。
- 四个分割 id 比较 `segmentation_result` 整型类别图，全分辨率。对黄金里出现过的每个类别算 IoU（交集 / 并集），再取平均。平均 IoU < 0.90 失败，失败输出含 `mask overlap` 和 `0.90`。尺寸不同、不是整型单通道、或黄金里少于两个不同类别值，直接失败。
- 三个抠图 id 比较 `matting_result`。该输出已是 0–255 的整型 alpha。前景是值 ≥ 128 的像素，前景 IoU < 0.90 失败，失败输出同样含 `mask overlap` 和 `0.90`。黄金前景像素数为 0，或当前与黄金尺寸不同，直接失败。
- 两个 embedding id 沿用现有余弦比较，余弦 < 0.999 失败，失败输出含 `embedding cosine similarity low`。各自一份向量 json，不共用 `dinov2_feature_embedding` 的黄金。
- 五个深度 id 只比较 `depth_map`（float 单通道，全分辨率）。相对误差是逐像素 `|当前 − 黄金| / max(|黄金|, 1e-6)` 的平均。平均值 > 0.02 失败，失败输出含 `depth error` 和 `0.02`。尺寸不同、出现非有限值、没有有限像素、或黄金里没有绝对值大于 `1e-6` 的像素，直接失败。不比较 `colorized_depth_map`。

黄金由该 id 的 CPU onnx 在上表这张输入图上生成：设 `MORTRED_UPDATE_GOLDEN=1` 跑对应过滤，写入 `test/golden/<用例名>` 后提交。这一次可以退出 0，但跳过文案是黄金已更新，不是数值通过。数值通过是去掉该环境变量再跑，退出码 0，且标准输出含 `NUMERICAL_PASSED`。

代价：新增 17 份短 toml 和 19 份黄金。`yolov8x`、`dinov2_vitl14`、`hrnet`、`ppmatting_1024`、`metric3d_1088` 在 CPU 上生成黄金和复跑都慢，只占该次用例的主机内存。不新增进程，不生成 engine，不占 GPU。

### 否决

改现有分割和抠图比较，让 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 也改成掩码重合。这四条的失败条件会从 fingerprint 变成掩码重合，那是 P1-11 的冻结句。本条的新用例单独使用掩码重合比较，现有 fingerprint 比较保持原样。

让这 19 条直接加载产品 toml，再靠现有强制 cpu 改设备。产品 toml 的 `type` 仍是 `tensorrt`，路径仍是 engine；现有强制 cpu 只改 `device` / `compute_backend`，不改 `type`，也不把 engine 换成 onnx。cpu profile 不链 TensorRT。这会把 CPU 用例绑到产品 TensorRT 配置上，冻结句不允许。

## 改动面

- `test/model_golden_test.cc`：登记 19 条 `model_golden.<id>`。
- `test/model_golden_registry.h`：增加只给这 19 条里分割、抠图、深度使用的比较。检测和 embedding 继续调用现有比较。
- `conf/ci/<id>_onnx_hosted.toml`：上表里除两个已有 DINOv2 文件以外的 17 份。
- `test/golden/`：每个新用例一份自己的黄金。

## 明确不改

- `test/CMakeLists.txt`。`model_golden_test` 仍在 `MORTRED_BUILD_FULL` 下登记，自带 `main`，不链接 `gtest_main`，`SKIP_RETURN_CODE` 仍是 77。不新增 ctest 别名。
- `golden_process_exit` 与 `MORTRED_SKIP_OR_FAIL_WEIGHTS`。
- 现有 fingerprint 比较，以及 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting` 的配置、输入图和黄金。
- `Yolov8TrtDetection`，以及任何产品 toml 为 `type=tensorrt` 的加载路径。
- `conf/model/**`、`conf/server/**`、`src/models`、`src/factory`、`src/server`、`src/apps`、`src/control`、HTTP 契约、OpenAPI、pack、`mortredctl`、engine 生成。
- 网关 `:8080` 与 supervisor `:8787` 的进程模型。

## 契约

无 HTTP 变化，无模型输入输出张量契约变化，无产品 toml 变化。

测试配置契约：每条新用例的 toml 中 `backend.type` 为 `onnx`，`backend.device` 为 `cpu`，`backend.model_file_path` 为上表该 id 的 `static_bs1.onnx`。加载时仍会强制 cpu。用例不打开 `conf/model` 里该 id 的产品文件。

数值契约：

- 检测：同类且 IoU ≥ 0.5；另保持现有框数一致和分数差 ≤ 0.01。
- 分割：类别平均 IoU ≥ 0.90；黄金至少两个不同类别值。
- 抠图：alpha ≥ 128 的前景 IoU ≥ 0.90；黄金前景非空。
- embedding：余弦 ≥ 0.999。
- 深度：`depth_map` 平均相对误差 ≤ 0.02。

失败文案要能看出是哪一种门：检测含 `low IoU` 或类别断言失败；分割和抠图含 `mask overlap` 与 `0.90`；embedding 含 `embedding cosine similarity low`；深度含 `depth error` 与 `0.02`。缺权重的跳过文案含 `weights not available`。

## 性能与资源

不新增常驻进程，不加载 engine，不占用显存。gpu profile 上这 19 条仍强制 cpu。cpu profile 不链 TensorRT，这 19 条仍可编译、可列出、可跑。

每个用例在现有测试进程里 `init`、跑一张图、结束即释放。缺权重在 `init` 之前返回。权重齐全时，`yolov8x`、`dinov2_vitl14`、`hrnet`、`ppmatting_1024`、`metric3d_1088` 的单次 CPU 推理会拉高主机内存和耗时；它们留在同一个 `model_golden_test` 里，不为此另起进程，也不从该二进制里摘掉。

## 验收标准

下列命令都在仓库根目录执行，二进制是 `cmake --preset full-cpu` 建成的 `./build/full-cpu/test/model_golden_test`。下面用 `<bin>` 表示它。19 条过滤记为 `<filter>`，即 `model_golden.nanodet_416:model_golden.yolov7x:model_golden.yolov8n:model_golden.yolov8l:model_golden.yolov8x:model_golden.pphuman_lite:model_golden.pphuman_server:model_golden.hrnet:model_golden.msocrnet:model_golden.dinov2_vits14:model_golden.dinov2_vitl14:model_golden.ppmatting_1024:model_golden.ppmatting_resnet34:model_golden.ppmatting_v2:model_golden.depth_vits14:model_golden.depth_vitb14:model_golden.depth_vitl14:model_golden.metric3d_512:model_golden.metric3d_1088`。

1. `<bin> --gtest_list_tests` 退出码 0。`model_golden.` 下能看到上表 19 个测试名，每个名字一条。同时仍能看到 `yolov8_detection`、`yolov8_onnx_detection`、`nanodet_detection`、`pphuman_segmentation`、`ppmatting_matting`、`dinov2_feature_embedding`。这 19 个名字不并进上述旧名字。
2. 打开这 19 条所用的 `conf/ci` toml：`type` 为 `onnx`，`device` 为 `cpu`，`model_file_path` 指向该 id 的 `static_bs1.onnx`。`yolov8n`、`yolov8l`、`yolov8x` 三条分别指向 `yolov8n.static_bs1.onnx`、`yolov8l.static_bs1.onnx`、`yolov8x.static_bs1.onnx`。`conf/ci/yolov8_onnx_hosted.toml` 仍指向 `yolov8s.onnx`。这 19 个 id 的 `conf/model` 产品文件仍是 `type = "tensorrt"`，路径仍是 `.engine`。
3. 权重和黄金都在、不设 `MORTRED_UPDATE_GOLDEN`、不设 `MORTRED_CI_REQUIRE_WEIGHTS` 时，`<bin> --gtest_filter=<filter>` 退出码 0，标准输出含 `NUMERICAL_PASSED`。
4. 只选中这 19 条，且它们的 onnx 都不存在，不设 `MORTRED_CI_REQUIRE_WEIGHTS`：同一过滤退出码 77，标准输出有一行 `NUMERICAL_NOT_RUN`。没有一条被记成通过。
5. 同一前提下设置 `MORTRED_CI_REQUIRE_WEIGHTS=1`：退出码 1，标准输出不含 `NUMERICAL_NOT_RUN`，失败信息含 `weights not available`。
6. 从上表五个检测 id 里任选一个。下面两次都要做，不能二选一。改该检测 id 的 `test/golden/<id>.json` 里一个框的 `class_id` 后，`model_golden.<id>` 退出码 1，失败输出含 `class_id`。把这次改动改回后，退出码 0。再把该框移到与原框 IoU < 0.5 的位置后，退出码 1，失败输出含 `low IoU`。这次也改回后，退出码 0。
7. 从上表四个分割 id 里任选一个，从上表三个抠图 id 里任选一个，两条都做。分割黄金是 `test/golden/<id>.json`，抠图黄金是 `test/golden/<id>.json`。编码都是 UTF-8 JSON：对象含整数 `height`、整数 `width`，以及按行展开的单通道整数数组 `values`，长度等于 `height * width`。分割的 `values` 是类别 id。抠图的 `values` 是 0–255 的 alpha。把 `values` 换成重合低于 0.90 的数据，替换后仍同尺寸：不改 `height` 和 `width`，`values` 长度不变。分割仍是整型单通道且至少两个类别值，抠图前景（值 ≥ 128）非空。该条退出码 1，失败输出含 `mask overlap` 和 `0.90`，而不是尺寸或类型不符。失败输出不含 `fingerprint drift`。改回后退出码 0。
8. 任选 `dinov2_vits14` 或 `dinov2_vitl14`。把黄金向量换成与原向量余弦 < 0.999 的等长向量。退出码 1，失败输出含 `embedding cosine similarity low`。改回后退出码 0。
9. 从上表五个深度 id 里任选一个。黄金是 `test/golden/<id>.json`。编码是 UTF-8 JSON：对象含整数 `height`、整数 `width`，以及按行展开的单通道有限浮点数组 `values`，长度等于 `height * width`，表示 `depth_map`。把 `values` 里每个数乘 2。替换后仍同尺寸：不改 `height` 和 `width`，`values` 长度不变，深度仍是有限 float 单通道且有绝对值大于 `1e-6` 的像素。退出码 1，失败输出含 `depth error` 和 `0.02`，而不是尺寸或类型不符。改回后退出码 0。
10. 破坏 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting` 之一的黄金 png 后，该条失败输出仍含 `fingerprint drift`。测完恢复该 png。
11. `<bin> --gtest_list_tests` 退出码 0，输出里没有新的 TensorRT 套件名。`./build/full/test/model_golden_test --gtest_list_tests` 退出码 0，`Yolov8TrtDetection` 仍在。新增数值用例只有 `model_golden.` 下这 19 个名字。

## 测试义务

QA 按验收 1 至 11 做黑盒。抽查覆盖每一类，不必 19 条都做破坏实验，但列表、配置路径和缺权重两条必须覆盖全部 19 个 id。

白盒核对：新分割、抠图用例没有调用 fingerprint 比较；新深度用例没有比较 `colorized_depth_map`；检测和 embedding 调用的是现有框比较和余弦比较；跳过文案仍是 `weights not available`；`golden_process_exit` 的 77 分支没有被绕开。黄金由对应 CPU onnx 生成，分割黄金至少两个类别值，抠图黄金前景非空，深度黄金有绝对值大于 `1e-6` 的有限像素。三个 YOLOv8 用例的 onnx 与黄金文件名互不相同，也不同于 `yolov8s`。

验收 4、5、6、7、8、9、10 会改黄金或挪走权重，结束后恢复。`MORTRED_UPDATE_GOLDEN=1` 那一次退出 0 不算验收 3 通过。

## 自测命令

在仓库根目录执行。期望退出码如下。

| 命令 | 前提 | 期望退出码 |
| --- | --- | --- |
| `<bin> --gtest_list_tests` | 已用 `full-cpu` 建成 | 0 |
| `<bin> --gtest_filter=<filter>` | 19 个 onnx 与黄金都在；未设置 `MORTRED_UPDATE_GOLDEN` 与 `MORTRED_CI_REQUIRE_WEIGHTS` | 0，且标准输出含 `NUMERICAL_PASSED` |
| `<bin> --gtest_filter=<filter>` | 这 19 个 onnx 都不存在；未设置 `MORTRED_CI_REQUIRE_WEIGHTS` | 77，且标准输出含 `NUMERICAL_NOT_RUN` |
| `MORTRED_CI_REQUIRE_WEIGHTS=1 <bin> --gtest_filter=model_golden.nanodet_416` | `nanodet_plus_m_416.static_bs1.onnx` 不存在 | 1，且标准输出不含 `NUMERICAL_NOT_RUN` |

生成黄金的命令是 `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=<filter>`。它不是上表第二行，不能当作数值通过。

## RD 不得自行决定

- 19 个 id 的名单、测试名、creator、输入图、onnx 路径、toml 段名。
- 阈值：检测 IoU 0.5 与分数差 0.01，分割和抠图 0.90，抠图前景 128，embedding 余弦 0.999，深度相对误差 0.02。
- 用 `yolov8s` 的 onnx、配置或黄金代替 `yolov8n`、`yolov8l`、`yolov8x` 中的任何一个。
- 手写空黄金、全 0 掩码、全 0 深度，或把另一个 id 的黄金复制过来。
- 把缺权重或然缺黄金写成通过；改掉 `weights not available`、退出码 77 或 `NUMERICAL_NOT_RUN`。
- 加载产品 TensorRT toml，或给这 19 个 id 加 TensorRT 比较。
- 修改 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 fingerprint 比较。
- 改 HTTP、产品 toml、模型张量契约、进程和 engine。
