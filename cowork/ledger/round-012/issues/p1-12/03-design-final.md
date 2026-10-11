# 技术方案

- 问题: P1-12
- 版本: v2
- 对应列表: `cowork/ledger/round-012/02-p0-p1-final.md` 的 P1-12

## 边界

只做已经冻结的 P1-12。冻结句不改写：除 yolov8s 以外、产品 toml 为 type=tensorrt 的 id 没有同规则 TRT 比较。改完后能观察到的变化：每个 id 有一条加载该产品 toml、且不把 backend 改成 cpu 的用例，失败条件与该 id 的 CPU 规则相同。yolov8s 的已关闭 IoU 门不写入这条的完成。

名单按冻结句顺序，38 个，一个不能少：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。

第 11 轮已测量，本版按这些事实验收，阈值保持现有值。

已有 CPU 黄金、加载产品 toml、不改成 cpu 时退出码是 1、标准输出没有 `NUMERICAL_PASSED` 的 9 条：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`yolov7`、`libface`、`enlightengan`、`dinov2_vitb14`。它们的期望就是这个失败，本条不把它们改成通过。实测原文：

- 分类三条：`class_id` 是 230，黄金是 170，失败文本含 `score mismatch at`。黄金茎名是 `mobilenetv2_classification`、`resnet50_classification`、`densenet121_classification`。
- `yolov5l`：检测数 5，黄金 7，失败文本含 `detection count changed for yolov5_detection`。
- `yolov6s`：检测数 7，黄金 5，失败文本含 `detection count changed for yolov6_detection`。
- `yolov7`：分数差约 0.065–0.38，门是现有 `0.01`。茎名 `yolov7_detection`。
- `libface`：关键点距离约 3.4–25.8，门是现有 `3.0`。产品 toml `conf/model/object_detection/libfacedetection/libface.toml` 的 `model_input_image_size` 是 `[320, 240]`。现有 CPU 用例读 `conf/model/object_detection/libfacedetection/640x480_config.toml`，尺寸是 `[640, 480]`。两条尺寸都保持原文件。茎名 `libface_detection`。
- `enlightengan`：平均绝对差约 7，门是现有 `1.0`，失败文本含 `fingerprint drift for enlightengan_enhancement`。
- `dinov2_vitb14`：余弦约 0.126，门是现有 `0.999`，失败文本含 `embedding cosine similarity low for dinov2_feature_embedding`。

同一规则下退出码 0 且有 `NUMERICAL_PASSED` 的已有 CPU 黄金 id：`nanodet_1x5`、`centerface`、`bisenetv2`、`pphuman_mobile`、`dbnet`、`attentive_gan`、`realesrgan`、`superpoint`、`modnet`、`ppmatting_512`。这 10 条仍要求退出码 0 且有 `NUMERICAL_PASSED`。

无 CPU 用例、黄金文件名 `trt_<id>`、比较已经退出码 0 的 id：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。基线仓库没有这些 `test/golden/trt_<id>.json`。它们只能由测试进程在 `MORTRED_UPDATE_GOLDEN=1` 下写出，然后 `GTEST_SKIP`，退出码 0，不打印 `NUMERICAL_PASSED`。去掉该变量再跑，退出码 0 且有 `NUMERICAL_PASSED`。

`msocrnet` 仍在 38 个里。二进制必须含 `conf/model/scene_segmentation/msocrnet/msocrnet.toml`。本机不跑 `TrtProductGolden.msocrnet`。若该过滤被执行且进程被杀掉，输出里没有 `[  PASSED  ]` 也没有 `[  FAILED  ]`，整条 P1-12 挂起，不从名单删除 `msocrnet`，该次不算通过。

一个测试进程只跑这 38 条中的一条。不使用 `--gtest_filter=TrtProductGolden.*`，也不把多个过滤名用冒号连在一起。

抽样破坏黄金只覆盖未加破坏时已经退出码 0 的 id。上面这 9 条不再做破坏黄金。破坏测完按原字节恢复。

不覆盖 P0-2、P1-8、D-8 至 D-12。不重开已关闭原句。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`centerface` 的 engine 输入名 `input.1` 不重开。H2 整行不因本条写成已满足。

## 做法

### 推荐

在现有 `model_golden_test` 里、`MORTRED_HAS_TRT` 条件下新增套件 `TrtProductGolden`。38 个测试名就是上面的 catalog id。每条：产品 toml 里的 `model_file_path` 不存在时，走现有 `MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available")`，发生在读图和 `init` 之前。文件在，就解析该产品 toml，只做现有的 `../` 路径剥除，不调用 `force_cpu_backend`，也不调用 `load_model_cfg`。解析后的 `type` 必须是 `tensorrt`，`device` 必须是 `gpu`。然后向标准输出打一行 `product toml <产品toml相对路径> type=tensorrt device=gpu model_file_path=<剥掉../之后的engine路径>`，用该家族现有 creator `init`、读该 id 现有 CPU 用例的那张图（没有 CPU 用例的 id 用下面写明的同家族图）、`run`，再按该 id 的现有 CPU 规则比较。

已有 CPU 用例的 19 个 id，比较函数、阈值和黄金茎名与现有 CPU 用例相同。`MORTRED_UPDATE_GOLDEN=1` 时这 19 条在读图和 `init` 之前失败，失败输出含 `refusing to overwrite cpu golden`，不写 `test/golden/`。其中 9 条的数值结果按上面的实测失败验收。另外 10 条按通过验收。

没有 CPU 用例的 id，黄金茎名是 `trt_<id>`。`msocrnet` 的比较写在用例里，本机不执行，也不写出 `trt_msocrnet.json`。

分组和规则，阈值都不放宽：

- 分类三条：现有 `expect_scores`。`class_id` 必须相等；每个分数与黄金差不超过现有 `1e-3`。分数失败文案含 `score mismatch at`。输入图是 `demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG`。
- 已有 CPU 用例的检测 `yolov5l`、`yolov6s`、`nanodet_1x5`、`yolov7`：现有 `expect_boxes`。框数一致，配对后 `class_id` 相同，IoU ≥ `0.5`，分数差不超过现有 `0.01`。框数失败文案含 `detection count changed for` 加该茎名。IoU 失败文案含 `low IoU for` 加该茎名。输入图是 `demo_data/model_test_input/object_detection/bus.jpg`。
- `libface`、`centerface`：现有人脸比较，框规则同上，并比较 landmarks，距离不超过现有 `3.0`。输入图分别是 `demo_data/model_test_input/object_detection/face_wo_mask.jpg`、`demo_data/model_test_input/object_detection/face_w_mask.jpg`。`libface` 仍加载产品 toml 的 320×240，不改去跟 CPU 用例的 640×480 对齐。
- 分割 `bisenetv2`、`pphuman_mobile` 和抠图 `modnet`、`ppmatting_512`：现有掩码重合。平均重合低于 `0.90` 时失败文案含 `mask overlap` 和 `0.90`。抠图前景是值 ≥ `128` 的像素。茎名是 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting`。分割输入图是 `demo_data/model_test_input/scene_segmentation/cityscapes_test.png`，`pphuman_mobile` 是 `demo_data/model_test_input/scene_segmentation/human_image.jpg`，两份抠图都是 `demo_data/model_test_input/matting/matting_test.jpg`。
- `dbnet`：现有 `expect_text_regions`。区域数一致，IoU ≥ `0.5`，分数差不超过 `0.01`。IoU 失败文案含 `low IoU for dbnet_text_detection`。输入图是 `demo_data/model_test_input/ocr/railway_ticket.png`。
- `attentive_gan`、`enlightengan`、`realesrgan`：现有 `expect_fingerprint`。均值绝对差超过现有 `1.0` 时失败文案含 `fingerprint drift for` 加该茎名。茎名是 `attentivegan_enhancement`、`enlightengan_enhancement`、`realesrgan_enhancement`。输入图分别是 `demo_data/model_test_input/enhancement/derain/test_1.png`、`demo_data/model_test_input/enhancement/low_light/lol_test_1.png`、`demo_data/model_test_input/enhancement/real_esr/test.JPG`。
- `superpoint`：现有 `expect_keypoints`。匹配距离不超过 `3.0`，匹配比例低于 `0.9` 时失败文案含 `keypoint match ratio low for superpoint_feature_point`。点数不同时失败输出含 `keypoint count changed`。输入图是 `demo_data/model_test_input/feature_point/test.png`。
- `dinov2_vitb14`：现有 `expect_embeddings`。余弦低于 `0.999` 时失败文案含 `embedding cosine similarity low for dinov2_feature_embedding`。输入图与分类相同。
- 没有 CPU 用例的检测：数值规则与现有 `expect_boxes` 相同。IoU 失败输出同时含 `low IoU` 和 `0.5`。分数失败输出含 `0.01`。类别失败输出含 `class_id`。不用 `Yolov8TrtDetection` 那条不比分数的规则，不读 `test/golden/yolov8_detection.json`。黄金是 `test/golden/trt_<id>.json`。输入图是 `bus.jpg`。`yolov8n`、`yolov8l`、`yolov8x` 各自一份黄金、各自的产品 engine。
- 没有 CPU 用例的分割：现有分割掩码重合，失败文案含 `mask overlap` 和 `0.90`。黄金茎名 `trt_<id>`。`pphuman_lite` 与 `pphuman_server` 的输入图是 `human_image.jpg`，`hrnet` 与 `msocrnet` 是 `cityscapes_test.png`。
- 没有 CPU 用例的抠图：现有抠图掩码重合，前景阈值 `128`，失败文案含 `mask overlap` 和 `0.90`。输入图是 `matting_test.jpg`。黄金茎名 `trt_<id>`。
- `dinov2_vits14`、`dinov2_vitl14`：余弦规则与现有 embedding 相同，阈值 `0.999`。失败文案同时含 `embedding cosine similarity low` 和 `0.999`。不读 `test/golden/dinov2_feature_embedding.json`。黄金是 `test/golden/trt_<id>.json`。输入图与分类相同。
- 五个深度 id：只比较 `depth_map`。`depth_map` 为 float 单通道、全分辨率。相对误差是逐像素 `|当前 − 黄金| / max(|黄金|, 1e-6)` 的平均。平均值大于 `0.02` 时失败，失败文案同时含 `depth error` 和 `0.02`。尺寸不同、出现非有限值、没有有限像素，或黄金里没有绝对值大于 `1e-6` 的像素，直接失败。不比较 `colorized_depth_map`。`depth_vits14`、`depth_vitb14`、`depth_vitl14` 用 `create_depth_anything_estimator`，`metric3d_512`、`metric3d_1088` 用 `create_metric3d_estimator`。输入图都是 `demo_data/model_test_input/mono_depth_estimation/0000000005.png`。黄金是 `test/golden/trt_<id>.json`。

creator 用该 id 在 `src/factory` 里已经绑定的那个。缺 engine 与缺权重走同一句 `weights not available`。只选中这一条且因此跳过时，现有 `golden_process_exit` 使进程退出码 77，标准输出含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。`MORTRED_CI_REQUIRE_WEIGHTS=1` 时同一处改为失败，退出码 1，标准输出不含 `NUMERICAL_NOT_RUN`。缺黄金、规则失败，都是退出码 1。不因为看不到 `/dev/nvidia*` 而跳过。

代价：full 构建才能链上 TensorRT，这 38 条才会出现在 `_bin/model_golden_test`。这 9 条产品 engine 与 CPU 黄金不一致，本条不把它们改成通过。`libface` 的 320×240 与 CPU 用例的 640×480 保持两边原文件，关键点距离仍按现有 `3.0` 失败。没有 CPU 用例的 18 个 id 要逐个进程生成自己的 `trt_<id>` 黄金，`msocrnet` 除外，所以本条交不出 `msocrnet` 的数值通过。37 条数值要分 37 个进程跑，显存按一次一个 engine 占用。

### 否决

把这 9 条做成退出码 0。做法包括放宽 `1e-3`、`0.5`、`0.01`、`3.0`、`0.90`、`1.0`、`0.999`、`0.02` 里的任何一个，改产品 toml，改 `src/`，或把 `libface` 的输入改成 640×480。第 11 轮偏离签字要求这 9 条各自退出码 0 且含 `NUMERICAL_PASSED`。那一条不写入本版：同一阈值下已经测得退出码 1，没有 `NUMERICAL_PASSED`。代价是把产品 engine 与 CPU 黄金的不一致写成通过。否决。

一个进程跑完全部 38 条，或用 `TrtProductGolden.*`。代价是把 `msocrnet` 卷进同一次地址空间，也和一条进程一条 id 冲突。给这 9 条破坏黄金，或按第 11 轮审查把除 `msocrnet` 外每个 id 的每种失败都破坏一遍。这 9 条未加破坏已经退出码 1，再改黄金仍是退出码 1，一次黑盒分不出是规则失败还是黄金被改。其余 id 的失败原文由下面的抽样覆盖，未抽样的 id 以未加破坏时的退出码 0 和 `NUMERICAL_PASSED` 验收。`superpoint` 不采用「每个点移到与原点距离大于 `3.0`」：`test/golden/superpoint_feature_point.json` 里的点已经全部离原点超过 `3.0`，只挪开不到 `3.0` 的一小段时，正确实现的退出码仍是 0，输出里没有 `keypoint match ratio low for superpoint_feature_point`。增强抽样不删 png，也不整图涂成另一种纯色。三条都否决。

把这 38 条做成 `Yolov8TrtDetection` 那样只比同类 IoU、不比分数，或调用 `load_model_cfg` / `force_cpu_backend`，或给那 19 个无 CPU 用例的 id 补 CPU 用例，或从 `G:\Codex\mortred-cowork-r11-p1-8`、`G:\Codex\mortred-cowork-r11-p1-12` 拷贝黄金。第一条会改掉已有 CPU 规则，并把已关闭的 `yolov8s` 门写进本条。第二条会把 backend 改成 cpu。第三条等于补 P1-8。第四条不是测试进程在 `MORTRED_UPDATE_GOLDEN=1` 下写出。四条都否决。

## 改动面

- `test/model_golden_test.cc`：在 `MORTRED_HAS_TRT` 内登记 38 条 `TrtProductGolden.<id>`，并包含深度工厂头。`Yolov8TrtDetection` 的函数体不动。
- `test/model_golden_registry.h`：增加只剥路径、不改 backend 的产品配置加载；为没有 CPU 用例的检测、embedding、深度准备会在失败时打印 `0.5`、`0.01`、`class_id`、`0.999`、`depth error` 与 `0.02` 的比较。已有的 `expect_scores`、`expect_boxes`、`expect_text_regions`、`expect_fingerprint`、`expect_keypoints`、`expect_embeddings`、分割和抠图重合比较的文案与阈值不改。
- `test/golden/trt_<id>.json`：仅上面 18 个无 CPU 用例且本机允许跑数值的 id，由 `MORTRED_UPDATE_GOLDEN=1` 的测试进程写出。不含 `trt_msocrnet`，不含任何 CPU 茎名。

## 明确不改

- 任何数值阈值，以及这 9 条的期望退出码。退出码 1 和对应失败原文就是期望。
- `Yolov8TrtDetection`，以及 `test/golden/yolov8_detection.json` 的写入路径。
- `force_cpu_backend`、`load_model_cfg`、`golden_process_exit`、`MORTRED_SKIP_OR_FAIL_WEIGHTS`。
- 现有 `model_golden.*` 用例的配置、输入图、黄金茎名和失败文案。
- `test/CMakeLists.txt`。不新增会一次跑多条 `TrtProductGolden` 的 ctest。
- `conf/model/**`、`conf/ci/**`、`conf/server/**`、`conf/packs/**`。`libface.toml` 的 `[320, 240]` 与 `640x480_config.toml` 的 `[640, 480]` 都保持。
- `src/models`、`src/factory`、`src/server`、`src/apps`、`src/control`。
- HTTP、OpenAPI、进程模型。gateway 仍是 `:8080`，supervisor 仍是 `:8787`。不拉起常驻服务。
- `centerface` 产品 toml 不补 `input_names`，不改 engine 输入名 `input.1`。
- 不给那 19 个无 CPU 用例的 id 补 CPU 用例。不跑 `golden_drift_check.py --record`。不手写黄金，不从第 11 轮未提交工作树拷贝 `trt_<id>.json`。

## 契约

无 HTTP 变化，无模型输入输出张量契约变化，无产品 toml 变化。产品 toml 的 `type` 仍是 `tensorrt`，`model_file_path` 仍是下表 engine。测试加载时不把 `device` 改成 `cpu`。标准输出的产品行是 `product toml <下表toml> type=tensorrt device=gpu model_file_path=<下表engine>`。

| id | 产品 toml | engine |
| --- | --- | --- |
| `mobilenetv2` | `conf/model/classification/mobilenetv2/mobilenetv2.toml` | `weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32` |
| `resnet` | `conf/model/classification/resnet/resnet.toml` | `weights/classification/resnet/resnet.engine.static_bs1.fp32` |
| `densenet` | `conf/model/classification/densenet/densenet.toml` | `weights/classification/densenet/densenet.engine.static_bs1.fp32` |
| `yolov5l` | `conf/model/object_detection/yolov5/yolov5l.toml` | `weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16` |
| `yolov6s` | `conf/model/object_detection/yolov6/yolov6s.toml` | `weights/object_detection/yolov6/yolov6s.engine.static_bs1.fp16` |
| `nanodet_1x5` | `conf/model/object_detection/nano_det/nanodet_1x5.toml` | `weights/object_detection/nanodet/nanodet_plus_m_1x5.engine.static_bs1.fp16` |
| `nanodet_416` | `conf/model/object_detection/nano_det/nanodet_416.toml` | `weights/object_detection/nanodet/nanodet_plus_m_416.engine.static_bs1.fp16` |
| `yolov7` | `conf/model/object_detection/yolov7/yolov7.toml` | `weights/object_detection/yolov7/yolov7.engine.static_bs1.fp16` |
| `yolov7x` | `conf/model/object_detection/yolov7/yolov7x.toml` | `weights/object_detection/yolov7/yolov7x.engine.static_bs1.fp16` |
| `yolov8n` | `conf/model/object_detection/yolov8/yolov8n.toml` | `weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16` |
| `yolov8l` | `conf/model/object_detection/yolov8/yolov8l.toml` | `weights/object_detection/yolov8/yolov8l.engine.static_bs1.fp16` |
| `yolov8x` | `conf/model/object_detection/yolov8/yolov8x.toml` | `weights/object_detection/yolov8/yolov8x.engine.static_bs1.fp16` |
| `libface` | `conf/model/object_detection/libfacedetection/libface.toml` | `weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16` |
| `centerface` | `conf/model/object_detection/centerface/centerface.toml` | `weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16` |
| `bisenetv2` | `conf/model/scene_segmentation/bisenetv2/bisenetv2.toml` | `weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.engine.static_bs1.fp32` |
| `pphuman_mobile` | `conf/model/scene_segmentation/pphuman/pphuman_mobile.toml` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.engine.static_bs1.fp16` |
| `pphuman_lite` | `conf/model/scene_segmentation/pphuman/pphuman_lite.toml` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.engine.static_bs1.fp16` |
| `pphuman_server` | `conf/model/scene_segmentation/pphuman/pphuman_server.toml` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.engine.static_bs1.fp16` |
| `hrnet` | `conf/model/scene_segmentation/hrnet/hrnet.toml` | `weights/scene_segmentation/hrnet/hrnetw48_ccd.engine.static_bs1.fp16` |
| `msocrnet` | `conf/model/scene_segmentation/msocrnet/msocrnet.toml` | `weights/scene_segmentation/msocrnet/msocrnet_fp16.engine.static_bs1.fp16` |
| `dbnet` | `conf/model/ocr/db_text_detector/dbnet.toml` | `weights/ocr/db_text_detector/db_model_large.engine.static_bs1.fp16` |
| `attentive_gan` | `conf/model/enhancement/attentive_gan_derain/attentive_gan.toml` | `weights/enhancement/attentive_gan_derain/attentive_gan_derain.engine.static_bs1.fp32` |
| `enlightengan` | `conf/model/enhancement/enlighten_gan/enlightengan.toml` | `weights/enhancement/enlighten_gan/enlightengan.engine.static_bs1.fp16` |
| `realesrgan` | `conf/model/enhancement/real_esrgan/realesrgan.toml` | `weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32` |
| `superpoint` | `conf/model/feature_point/superpoint/superpoint.toml` | `weights/feature_point/superpoint/superpoint_120x160.engine.static_bs1.fp16` |
| `dinov2_vits14` | `conf/model/feature_embedding/dinov2/dinov2_vits14.toml` | `weights/classification/dinov2/dinov2_vits14_pretrain.engine.static_bs1.fp16` |
| `dinov2_vitb14` | `conf/model/feature_embedding/dinov2/dinov2_vitb14.toml` | `weights/classification/dinov2/dinov2_vitb14_pretrain.engine.static_bs1.fp16` |
| `dinov2_vitl14` | `conf/model/feature_embedding/dinov2/dinov2_vitl14.toml` | `weights/classification/dinov2/dinov2_vitl14_pretrain.engine.static_bs1.fp16` |
| `modnet` | `conf/model/matting/modnet/modnet.toml` | `weights/matting/modnet/modnet_hrnet_w18.engine.static_bs1.fp16` |
| `ppmatting_512` | `conf/model/matting/ppmatting/ppmatting_512.toml` | `weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.engine.static_bs1.fp16` |
| `ppmatting_1024` | `conf/model/matting/ppmatting/ppmatting_1024.toml` | `weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.engine.static_bs1.fp16` |
| `ppmatting_resnet34` | `conf/model/matting/ppmatting/ppmatting_resnet34.toml` | `weights/matting/ppmatting/pp_humanmatting-resnet34_vd.engine.static_bs1.fp16` |
| `ppmatting_v2` | `conf/model/matting/ppmatting/ppmatting_v2.toml` | `weights/matting/ppmatting/ppmattingv2_stdc1_human_512.engine.static_bs1.fp16` |
| `depth_vits14` | `conf/model/mono_depth_estimation/depth_anything/depth_vits14.toml` | `weights/mono_depth_estimation/depth_anything/depth_anything_vits14.engine.static_bs1.fp16` |
| `depth_vitb14` | `conf/model/mono_depth_estimation/depth_anything/depth_vitb14.toml` | `weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.engine.static_bs1.fp16` |
| `depth_vitl14` | `conf/model/mono_depth_estimation/depth_anything/depth_vitl14.toml` | `weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.engine.static_bs1.fp16` |
| `metric3d_512` | `conf/model/mono_depth_estimation/metric3d/metric3d_512.toml` | `weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16` |
| `metric3d_1088` | `conf/model/mono_depth_estimation/metric3d/metric3d_1088.toml` | `weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.engine.static_bs1.fp16` |

## 性能与资源

不新增常驻进程，不占用 gateway 或 supervisor。每条用例在自己的进程里加载一个 engine，进程结束释放显存。本机 GPU 是 RTX 2070 SUPER，8GiB，TensorRT 10.3，CUDA 12.6。一次只留一个 engine。`yolov8x`、`dinov2_vitl14`、`hrnet`、`ppmatting_1024`、`metric3d_1088` 仍是每个 id 单独一个进程。`msocrnet` 的产品输入尺寸是 1024×2048，本机约 19GiB 内存上不跑它的推理。这 9 条也会各加载一次产品 engine，比较失败后退出码 1，不因为失败而改成跳过。cpu profile 不链 TensorRT。验收二进制是 full 构建的 `_bin/model_golden_test`，不是 `build/full-cpu/test/model_golden_test`。

## 验收标准

下列命令都在仓库根目录的 WSL 里执行。`<bin>` 是 `_bin/model_golden_test`。先用 `cmake --preset full && cmake --build --preset full --target model_golden_test` 得到这个二进制。`PATH` 要能找到本机 CUDA 12.x 的 `nvcc`。未写明的环境变量都不要设置。每条数值命令的 `--gtest_filter` 只有一个测试名。产品行指契约表里该 id 的那一行全文。

1. `<bin> --gtest_list_tests` 退出码 0。输出含原文行 `TrtProductGolden.`，其下 38 个测试名行分别是 `mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`，没有测试名行 `yolov8s`。输出含原文行 `Yolov8TrtDetection.` 和测试名行 `MatchesCpuGoldenBoxes`。输出含原文行 `model_golden.`，其下测试名行含 `yolov5_detection`、`nanodet_detection`、`yolov8_detection`，不含 `nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。不要求 `TrtProductGolden.yolov8s` 或 `model_golden.nanodet_416` 这种点号整串出现在同一行。
2. 用一段脚本读契约表里 38 份产品 toml。`type = "tensorrt"` 且文件文本含该行 engine 时，每行打印 `ok <id> type=tensorrt <engine>`，脚本退出码 0。输出含 38 行 `ok`，不含 `MISMATCH`，不含 `ok yolov8s`。
3. `grep -a -F "conf/model/scene_segmentation/msocrnet/msocrnet.toml" <bin>` 退出码 0，输出含这条路径。不执行 `--gtest_filter=TrtProductGolden.msocrnet`。验收 1 和本条通过时，输出里没有 `TrtProductGolden.msocrnet` 的 `NUMERICAL_PASSED`。若该过滤被执行且输出里没有 `[  PASSED  ]` 也没有 `[  FAILED  ]`，该进程退出码不是 0，输出不含 `NUMERICAL_PASSED`，整条 P1-12 挂起，不从名单删除 `msocrnet`。
4. `grep -F "model_input_image_size = [320, 240]" conf/model/object_detection/libfacedetection/libface.toml` 退出码 0。`grep -F "model_input_image_size = [640, 480]" conf/model/object_detection/libfacedetection/640x480_config.toml` 退出码 0。
5. `<bin> --gtest_filter=TrtProductGolden.mobilenetv2` 退出码 1。标准输出含该 id 的产品行，含 `score mismatch at`，含 `230`，含 `170`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
6. `<bin> --gtest_filter=TrtProductGolden.resnet` 退出码 1。标准输出含该 id 的产品行，含 `score mismatch at`，含 `230`，含 `170`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
7. `<bin> --gtest_filter=TrtProductGolden.densenet` 退出码 1。标准输出含该 id 的产品行，含 `score mismatch at`，含 `230`，含 `170`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
8. `<bin> --gtest_filter=TrtProductGolden.yolov5l` 退出码 1。标准输出含该 id 的产品行，含 `detection count changed for yolov5_detection`，含 `5`，含 `7`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
9. `<bin> --gtest_filter=TrtProductGolden.yolov6s` 退出码 1。标准输出含该 id 的产品行，含 `detection count changed for yolov6_detection`，含 `7`，含 `5`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
10. `<bin> --gtest_filter=TrtProductGolden.yolov7` 退出码 1。标准输出含该 id 的产品行，含 `0.01`，打印出的分数差落在 0.065 到 0.38。不含 `detection count changed`，不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
11. `<bin> --gtest_filter=TrtProductGolden.libface` 退出码 1。标准输出含该 id 的产品行，含 `get_landmarks`，打印出的关键点距离落在 3.4 到 25.8，且这些距离大于 3.0。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
12. `<bin> --gtest_filter=TrtProductGolden.enlightengan` 退出码 1。标准输出含 `product toml conf/model/enhancement/enlighten_gan/enlightengan.toml type=tensorrt device=gpu model_file_path=weights/enhancement/enlighten_gan/enlightengan.engine.static_bs1.fp16`，含 `fingerprint drift for enlightengan_enhancement, mean abs diff = `，等号后的数大于 `1.0`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
13. `<bin> --gtest_filter=TrtProductGolden.dinov2_vitb14` 退出码 1。标准输出含该 id 的产品行，含 `embedding cosine similarity low for dinov2_feature_embedding`，含 `0.126`，含 `0.999`。不含 `device=cpu`，不含 `NUMERICAL_PASSED`，不含 `NUMERICAL_NOT_RUN`。
14. 下面 10 条各自一个进程，过滤名是 `TrtProductGolden.` 加 id：`nanodet_1x5`、`centerface`、`bisenetv2`、`pphuman_mobile`、`dbnet`、`attentive_gan`、`realesrgan`、`superpoint`、`modnet`、`ppmatting_512`。每条退出码 0。标准输出含 `NUMERICAL_PASSED` 和该 id 的产品行。不含 `device=cpu`，不含 `NUMERICAL_NOT_RUN`。
15. 下面 18 个 id 先各自一个进程执行 `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.<id>`：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。每条退出码 0，标准输出含 `golden updated`，不含 `NUMERICAL_PASSED`。然后去掉该环境变量，再各自一个进程跑同一过滤。每条退出码 0，标准输出含 `NUMERICAL_PASSED` 和该 id 的产品行。不含 `device=cpu`，不含 `NUMERICAL_NOT_RUN`。写出那一次退出码 0 不算数值通过。
16. `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.mobilenetv2` 退出码 1。输出含 `refusing to overwrite cpu golden`，不含 `NUMERICAL_PASSED`。`test/golden/mobilenetv2_classification.json` 字节不变。
17. 临时让 `weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16` 不存在，不设 `MORTRED_CI_REQUIRE_WEIGHTS`，执行 `<bin> --gtest_filter=TrtProductGolden.yolov5l`。退出码 77，标准输出含 `NUMERICAL_NOT_RUN` 和 `weights not available`，不含 `NUMERICAL_PASSED`，不含 `detection count changed`。恢复该 engine。
18. 同一 engine 仍不存在，执行 `MORTRED_CI_REQUIRE_WEIGHTS=1 <bin> --gtest_filter=TrtProductGolden.yolov5l`。退出码 1，输出含 `weights not available`，不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。恢复该 engine。

验收 19 至 30 只改未加破坏时已经退出码 0 的 id。每次只改这一份黄金，一个进程一个过滤名。看完退出码和输出后按原字节恢复，再跑同一条过滤。恢复后退出码 0，输出含 `NUMERICAL_PASSED`。不改验收 5 至 13 那 9 条的黄金。

19. 黄金是 `test/golden/nanodet_detection.json`。只改一个框的 `x`、`y`、`w`、`h`，不改框数、`class_id` 和 `score`。新位置与原框的 IoU 大于 0 且小于 0.5，并且大于该原框与同一份黄金里其余每个框的 IoU。`<bin> --gtest_filter=TrtProductGolden.nanodet_1x5` 退出码 1，输出含 `low IoU for nanodet_detection`，不含 `NUMERICAL_PASSED`。
20. 黄金是 `test/golden/trt_nanodet_416.json`。第一次只把一个框的 `class_id` 改成另一个整数，不改几何和 `score`。`<bin> --gtest_filter=TrtProductGolden.nanodet_416` 退出码 1，输出含 `class_id`，不含 `NUMERICAL_PASSED`。按原字节恢复后，再按验收 19 的几何限制移这个框。退出码 1，输出同时含 `low IoU` 和 `0.5`，不含 `NUMERICAL_PASSED`。
21. 黄金是 `test/golden/trt_yolov8n.json`。只把一个框的 `score` 改到与原值差大于 `0.01`，不改几何、框数和 `class_id`。`<bin> --gtest_filter=TrtProductGolden.yolov8n` 退出码 1，输出含 `0.01`，不含 `low IoU`，不含 `NUMERICAL_PASSED`。不读 `test/golden/yolov8_detection.json`。
22. 黄金是 `test/golden/centerface_detection.json`。框上已有 `landmarks`。只把一个点的一个坐标移动，使该点与原点位的距离大于 `3.0`。不改该框的 `x`、`y`、`w`、`h`、`score`、`class_id`，不改点的个数。`<bin> --gtest_filter=TrtProductGolden.centerface` 退出码 1，输出含 `get_landmarks`，不含 `NUMERICAL_PASSED`。
23. 黄金是 `test/golden/bisenetv2_segmentation.json`。`values` 每个整数加 1，不改 `height`、`width` 和长度，且仍至少两个类别值。`<bin> --gtest_filter=TrtProductGolden.bisenetv2` 退出码 1，输出同时含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`，不含 `NUMERICAL_PASSED`。
24. 对 `test/golden/trt_pphuman_lite.json` 做与验收 23 相同的加 1。`<bin> --gtest_filter=TrtProductGolden.pphuman_lite` 退出码 1，输出同时含 `mask overlap` 和 `0.90`，不含 `NUMERICAL_PASSED`。
25. 黄金是 `test/golden/modnet_matting.json`。值仍是 0–255 的整数。前景（值 ≥ `128`）至少 2 个像素，与改前前景没有交集。不改 `height`、`width` 和长度。`<bin> --gtest_filter=TrtProductGolden.modnet` 退出码 1，输出同时含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`，不含 `NUMERICAL_PASSED`。
26. 黄金是 `test/golden/dbnet_text_detection.json`。只改一个区域的 `x`、`y`、`w`、`h`，不改区域数和 `score`。位置限制与验收 19 的框相同。`<bin> --gtest_filter=TrtProductGolden.dbnet` 退出码 1，输出含 `low IoU for dbnet_text_detection`，不含 `NUMERICAL_PASSED`。
27. 黄金是 `test/golden/realesrgan_enhancement.png`。不删除文件。png 仍能按颜色图打开。宽高不变。每个像素的每个通道都加上 `2`，加完大于 `255` 的通道改为 `0`。`<bin> --gtest_filter=TrtProductGolden.realesrgan` 退出码 1，输出含 `fingerprint drift for realesrgan_enhancement`，不含 `NUMERICAL_PASSED`。
28. 黄金是 `test/golden/superpoint_feature_point.json`。不改 `points` 的个数。把每个点的 `x` 和 `y` 都加上 `1000`。`<bin> --gtest_filter=TrtProductGolden.superpoint` 退出码 1，输出含 `keypoint match ratio low for superpoint_feature_point`，不含 `keypoint count changed`，不含 `NUMERICAL_PASSED`。
29. 黄金是 `test/golden/trt_dinov2_vits14.json`。UTF-8 JSON，键名保持 `embeddings`，不改长度，每个分量取相反数。`<bin> --gtest_filter=TrtProductGolden.dinov2_vits14` 退出码 1，输出同时含 `embedding cosine similarity low` 和 `0.999`，不含 `golden missing`，不含 `NUMERICAL_PASSED`。
30. 黄金是 `test/golden/trt_depth_vits14.json`。不改 `height`、`width` 和 `values` 长度。每个数乘 2，仍是有限 float，且有绝对值大于 `1e-6` 的像素。`<bin> --gtest_filter=TrtProductGolden.depth_vits14` 退出码 1，输出同时含 `depth error` 和 `0.02`，不含 `colorized_depth_map`，不含 `NUMERICAL_PASSED`。

## 测试义务

QA 按验收 1 至 30 做黑盒。验收 5 至 13 的 9 条、验收 14 的 10 条、验收 15 的 18 条必须逐条单独进程跑完。不跑 `TrtProductGolden.msocrnet`。不跑 `TrtProductGolden.*`。验收 19 至 30 覆盖已经退出码 0 的检测、`yolov8n` 的分数门、人脸关键点、分割、抠图、OCR、增强、特征点、embedding 和深度各一条。分类、`yolov5l`、`yolov6s`、`yolov7`、`libface`、`enlightengan`、`dinov2_vitb14` 的失败原文以验收 5 至 13 的未加破坏输出为准，不再破坏它们的黄金。验收 19 至 30 改过的黄金，测完按原字节恢复。

白盒核对：38 条都不调用 `force_cpu_backend` 和 `load_model_cfg`。已有 CPU 用例的 19 条调用的比较函数与该 CPU 用例相同，黄金茎名与「做法」一致，阈值未改。`yolov8n`、`yolov8l`、`yolov8x` 不调用 `Yolov8TrtDetection` 的比较，也不读 `yolov8_detection` 黄金。深度比较只读 `depth_map`。`msocrnet` 的配置字符串是 `conf/model/scene_segmentation/msocrnet/msocrnet.toml`。跳过文案仍含 `weights not available`。没有从 `G:\Codex\mortred-cowork-r11-p1-8` 或 `G:\Codex\mortred-cowork-r11-p1-12` 引入黄金。

## 自测命令

在仓库根目录的 WSL 里执行。`<bin>` 同上。数值命令都是一个进程一个过滤名。未写明的环境变量不要设置。RD 不改验收 5 至 13 的黄金，也不做验收 19 至 30。

| 命令 | 前提 | 期望退出码 |
| --- | --- | --- |
| `cmake --preset full && cmake --build --preset full --target model_golden_test` | `PATH` 能找到本机 CUDA 12.x 的 `nvcc`，能链 TensorRT 10.x | 0，得到 `_bin/model_golden_test` |
| `<bin> --gtest_list_tests` | 上一步的二进制 | 0。按套件行加测试名行核对，与验收 1 相同：输出含原文行 `TrtProductGolden.`，其下 38 个测试名行分别是 `mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`，没有测试名行 `yolov8s`。输出含原文行 `Yolov8TrtDetection.` 和测试名行 `MatchesCpuGoldenBoxes`。输出含原文行 `model_golden.`，其下测试名行含 `yolov5_detection`、`nanodet_detection`、`yolov8_detection`，不含 `nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。不要求 `TrtProductGolden.yolov8s` 或 `model_golden.nanodet_416` 这种点号整串出现在同一行。 |
| 验收 2 的 toml 检查 | 产品 toml 未改 | 0，输出含 38 行 `ok`，不含 `MISMATCH`，不含 `ok yolov8s` |
| `grep -a -F "conf/model/scene_segmentation/msocrnet/msocrnet.toml" <bin>` | 同上二进制 | 0，输出含该路径 |
| 验收 4 的两条 `grep -F` | 两份 toml 未改 | 每条 0 |
| 验收 15 的 18 条 `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.<id>` | 该 id 的 engine 在；不含 `msocrnet` | 每条 0，输出含 `golden updated`，不含 `NUMERICAL_PASSED` |
| 验收 14 的 10 条 `<bin> --gtest_filter=TrtProductGolden.<id>` | engine 在；未设置 `MORTRED_UPDATE_GOLDEN` 与 `MORTRED_CI_REQUIRE_WEIGHTS` | 每条 0，输出含 `NUMERICAL_PASSED` 与该 id 的产品行 |
| 验收 15 去掉更新变量后的 18 条 `<bin> --gtest_filter=TrtProductGolden.<id>` | 上一步已经写出 `test/golden/trt_<id>.json` | 每条 0，输出含 `NUMERICAL_PASSED` 与该 id 的产品行 |
| 验收 5 至 13 的 9 条 `<bin> --gtest_filter=TrtProductGolden.<id>` | 现有 CPU 黄金与产品 engine 都在；阈值未改 | 每条 1，输出含该条写明的失败原文和产品行，不含 `NUMERICAL_PASSED` |
| `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.mobilenetv2` | 无 | 1，输出含 `refusing to overwrite cpu golden`，不含 `NUMERICAL_PASSED`；`test/golden/mobilenetv2_classification.json` 字节不变 |
| `<bin> --gtest_filter=TrtProductGolden.yolov5l` | `yolov5l.engine.static_bs1.fp16` 不存在；未设置 `MORTRED_CI_REQUIRE_WEIGHTS` | 77，输出含 `NUMERICAL_NOT_RUN` 和 `weights not available`，不含 `NUMERICAL_PASSED` |
| `MORTRED_CI_REQUIRE_WEIGHTS=1 <bin> --gtest_filter=TrtProductGolden.yolov5l` | 同一 engine 不存在 | 1，输出含 `weights not available`，不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED` |

不执行 `<bin> --gtest_filter=TrtProductGolden.msocrnet`。不执行 `<bin> --gtest_filter=TrtProductGolden.*`。

## RD 不得自行决定

- 38 个 id、测试名 `TrtProductGolden.<id>`、creator、输入图、产品 toml 和 engine 路径。
- 这 9 条的期望退出码。它们必须保持退出码 1 和验收 5 至 13 的失败原文。不得放宽阈值、改黄金、改产品 toml 或改 `src/` 来换成退出码 0。这个退出码 1 不是偏离。
- 已有 CPU 用例的 19 个 id 的黄金茎名、阈值和失败文案。分类仍是 `class_id` 加分数差 `1e-3`。检测和 OCR 的 IoU 仍是 `0.5`、分数差仍是 `0.01`。人脸 landmarks 距离仍是 `3.0`。分割和抠图仍是 `0.90`，抠图前景仍是 `128`。增强均值绝对差仍是 `1.0`。特征点匹配距离仍是 `3.0`、比例仍是 `0.9`。`dinov2_vitb14` 余弦仍是 `0.999`。
- 没有 CPU 用例的检测失败输出必须能看到 `0.5`、`0.01`、`class_id`。embedding 必须能看到 `0.999`。分割和抠图必须能看到 `0.90`。深度必须能看到 `depth error` 和 `0.02`。深度平均相对误差的公式不得改。深度不比较 `colorized_depth_map`。
- 用 `yolov8s` 的用例、配置或黄金代替 `yolov8n`、`yolov8l`、`yolov8x` 中的任何一个，或修改 `Yolov8TrtDetection`。
- 给这 19 个没有 CPU 用例的 id 加 CPU 用例。手写黄金、复制别的 id 的黄金、从第 11 轮工作树拷贝 `trt_<id>.json`，或运行 `golden_drift_check.py --record`。`MORTRED_UPDATE_GOLDEN=1` 那一次退出码 0 不得记成数值通过。
- 对验收 5 至 13 的 9 条做破坏黄金。缺 engine 时不得打印 `NUMERICAL_PASSED`。不得把跳过文案改成不含 `weights not available` 的句子。
- 一个进程跑多于一个本条 id。本机不得跑 `TrtProductGolden.msocrnet`。不得因 OOM 从名单删除 `msocrnet`。
- 调用 `force_cpu_backend` 或 `load_model_cfg` 来跑这 38 条。因缺少 `/dev/nvidia*` 而跳过。
- 改产品 toml、HTTP、张量契约、gateway、supervisor、engine 生成，或给 `centerface` 改输入名，或把 `libface` 的 320×240 改成 640×480。
- 用 `build/full-cpu/test/model_golden_test` 或 full-cpu 构建作为本条二进制。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 产品 | 同意 | 无 | v2 | `03-discussion/v2-sign-pm.md` |
| 研发 | 不同意 | 见记录 | v2 | `03-discussion/v2-sign-rd.md` |
| 技术负责人 | 同意 | 无 | v2 | `03-discussion/v2-sign-tech-lead.md` |
| 用户 | 选用 | engine 名为 `realesr-general-x4v3` | v2 | 见用户决定 |

## 用户决定
- 日期: 2026-10-11
- 选用版本: v2
- 决定原文: engine应该是realesr-general-x4v3
- 未选用: 停止本轮。研发把契约表读成 `real-esr-general-x4v3` 的那一读法不采用。正文契约表与产品 toml 剥掉 `../` 之后都是 `weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32`。
- 状态: running

