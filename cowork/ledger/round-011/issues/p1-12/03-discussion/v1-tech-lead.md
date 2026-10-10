# 技术方案

- 问题: P1-12
- 版本: v1
- 对应列表: `cowork/ledger/round-011/02-p0-p1-final.md` 版本 v3 的 P1-12

## 边界

只做已经冻结的 P1-12。冻结句是：除 `yolov8s` 以外、产品 toml 为 `type=tensorrt` 的 id 没有同规则 TRT 比较。改完后能观察到的变化只是：下面 38 个 id 各有一条加载该 id 产品 toml、且不把 backend 改成 cpu 的用例，失败条件与该 id 的 CPU 规则相同。`yolov8s` 的已关闭同类 IoU 门不写入本条完成。

名单按冻结句顺序，一个不能少：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。

基线上已有 CPU 数值比较的 19 个 id，TRT 失败条件、阈值和失败文案与该 id 现有 CPU 用例相同，比较对象是该 CPU 用例已经在用的那份黄金。这 19 个是：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`yolov7`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vitb14`、`modnet`、`ppmatting_512`。`MORTRED_UPDATE_GOLDEN=1` 时这 19 条不得写黄金。

基线上没有 CPU 用例的另外 19 个仍在名单里。它们的 TRT 用例使用基线上同家族已经写明的失败规则和阈值，失败输出里打印该阈值。不新增这 19 个 id 的 CPU 用例，不读未合入工作树 `G:\Codex\mortred-cowork-r11-p1-8` 的符号或黄金。这 19 个是：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。

基线 `test/model_golden_registry.h` 没有深度比较函数，也没有深度阈值常量。五个深度 id 仍按本条写明规则：只比较 `depth_map`，平均相对误差大于 `0.02` 失败，失败输出同时含 `depth error` 和 `0.02`。不比较 `colorized_depth_map`。这个数由本条固定，不从那份未合入工作树引用函数或黄金。

`msocrnet` 留在 38 个里。本机约 19GiB 的 WSL 上一次推理被 OOM 杀掉，本方案的自测和验收都不跑 `TrtProductGolden.msocrnet`。用例必须能被列出，并且测试二进制里含有它的产品 toml 路径。若有人跑了这条过滤且进程被 OOM 杀掉（没有 gtest 的通过或失败收尾），整条 P1-12 挂起，不从名单删掉 `msocrnet`，也不把未跑完的数值记成通过。

一个测试进程只跑这 38 条中的一条。不使用 `--gtest_filter=TrtProductGolden.*`，也不使用把这 38 个过滤名用冒号连在一起的过滤。

不覆盖 P0-2、P1-8、D-8 至 D-12。不重开已关闭原句：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、D-1 至 D-6。P1-5 的关闭范围只有 `yolov8s`、同类 IoU ≥ 0.5。P1-11 的关闭范围只有 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 这四个 id 的 CPU 全分辨率掩码重合，阈值 0.90。这两条关闭都不写入 P1-12 的完成。`centerface` 的 engine 输入名 `input.1` 保持关闭，不重开。H2 整行不因本条方案写成已满足。

## 做法

### 推荐

在现有 `model_golden_test` 里、`MORTRED_HAS_TRT` 条件下新增套件 `TrtProductGolden`。38 个测试名就是上面的 catalog id。每条只做这些事：`weights_available` 发现产品 toml 里的 `model_file_path` 不存在，就走现有 `MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available")`；文件在，就解析该产品 toml，只做现有的 `../` 路径剥除，不调用 `force_cpu_backend`，也不调用 `load_model_cfg`。解析后的 `type` 必须是 `tensorrt`，`device` 必须是 `gpu`。然后向标准输出打一行 `product toml <产品toml相对路径> type=tensorrt device=gpu model_file_path=<剥掉../之后的engine路径>`，用该家族现有 creator `init`、读下表那张图、`run`、再按本组规则比较。

已有 CPU 用例的 19 个 id，比较函数和黄金茎名与现有 CPU 用例相同，因此失败文案相同。`MORTRED_UPDATE_GOLDEN=1` 时这 19 条在读图和 `init` 之前失败，失败输出含 `refusing to overwrite cpu golden`，不写 `test/golden/`。

没有 CPU 用例的 19 个 id，黄金茎名是 `trt_<id>`，只在 `MORTRED_UPDATE_GOLDEN=1` 时由该测试进程写出，然后 `GTEST_SKIP`，跳过文案含 `golden updated`。这一次退出码 0，标准输出不含 `NUMERICAL_PASSED`，不算数值通过。去掉该环境变量再跑，才是数值通过。不手写黄金，不跑 `golden_drift_check.py --record`。

分组和规则：

- 分类 `mobilenetv2`、`resnet`、`densenet`：调用现有 `expect_scores`。`class_id` 必须相等；每个分数与黄金差不超过现有 `1e-3`。分数失败文案含 `score mismatch at`。黄金茎名分别是 `mobilenetv2_classification`、`resnet50_classification`、`densenet121_classification`。输入图是 `demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG`。
- 已有 CPU 用例的检测 `yolov5l`、`yolov6s`、`nanodet_1x5`、`yolov7`：调用现有 `expect_boxes`。框数一致，配对后 `class_id` 相同，IoU ≥ `0.5`，分数差不超过现有 `0.01`。IoU 失败文案含 `low IoU for` 加该 CPU 茎名。茎名分别是 `yolov5_detection`、`yolov6_detection`、`nanodet_detection`、`yolov7_detection`。输入图是 `demo_data/model_test_input/object_detection/bus.jpg`。
- `libface`、`centerface`：调用现有人脸比较（框规则同上，并比较 landmarks，距离不超过现有 `3.0`）。茎名是 `libface_detection`、`centerface_detection`。输入图分别是 `demo_data/model_test_input/object_detection/face_wo_mask.jpg`、`demo_data/model_test_input/object_detection/face_w_mask.jpg`。
- 已有 CPU 用例的分割 `bisenetv2`、`pphuman_mobile` 和抠图 `modnet`、`ppmatting_512`：调用现有掩码重合比较。平均重合低于 `0.90` 时失败文案含 `mask overlap` 和 `0.90`。抠图前景是值 ≥ `128` 的像素。茎名是 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting`。分割输入图是 `demo_data/model_test_input/scene_segmentation/cityscapes_test.png`，`pphuman_mobile` 是 `demo_data/model_test_input/scene_segmentation/human_image.jpg`，两份抠图都是 `demo_data/model_test_input/matting/matting_test.jpg`。
- `dbnet`：调用现有 `expect_text_regions`。区域数一致，IoU ≥ `0.5`，分数差不超过 `0.01`。IoU 失败文案含 `low IoU for dbnet_text_detection`。输入图是 `demo_data/model_test_input/ocr/railway_ticket.png`。
- `attentive_gan`、`enlightengan`、`realesrgan`：调用现有 `expect_fingerprint`。均值绝对差超过现有 `1.0` 时失败文案含 `fingerprint drift for` 加该 CPU 茎名。茎名是 `attentivegan_enhancement`、`enlightengan_enhancement`、`realesrgan_enhancement`。输入图分别是 `demo_data/model_test_input/enhancement/derain/test_1.png`、`demo_data/model_test_input/enhancement/low_light/lol_test_1.png`、`demo_data/model_test_input/enhancement/real_esr/test.JPG`。
- `superpoint`：调用现有 `expect_keypoints`。匹配距离不超过 `3.0`，匹配比例低于 `0.9` 时失败文案含 `keypoint match ratio low for superpoint_feature_point`。输入图是 `demo_data/model_test_input/feature_point/test.png`。
- `dinov2_vitb14`：调用现有 `expect_embeddings`。余弦低于 `0.999` 时失败文案含 `embedding cosine similarity low for dinov2_feature_embedding`。输入图与分类相同。
- 没有 CPU 用例的检测 `nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`：数值规则与现有 `expect_boxes` 相同（框数、`class_id`、IoU ≥ `0.5`、分数差 ≤ `0.01`）。IoU 失败文案同时含 `low IoU` 和 `0.5`。分数失败文案含 `0.01`。类别失败文案含 `class_id`。不用 `Yolov8TrtDetection` 那条不比分数的规则，不读 `test/golden/yolov8_detection.json`。黄金是 `test/golden/trt_<id>.json`。输入图是 `bus.jpg`。三个 YOLOv8 id 各自一份黄金、各自的产品 engine。
- 没有 CPU 用例的分割 `pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`：调用现有分割掩码重合。失败文案含 `mask overlap` 和 `0.90`。黄金茎名 `trt_<id>`。`pphuman_lite` 与 `pphuman_server` 的输入图是 `human_image.jpg`，`hrnet` 与 `msocrnet` 是 `cityscapes_test.png`。`msocrnet` 的这条比较写在用例里，本机不执行它。
- 没有 CPU 用例的抠图 `ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`：调用现有抠图掩码重合，前景阈值 `128`，失败文案含 `mask overlap` 和 `0.90`。输入图是 `matting_test.jpg`。黄金茎名 `trt_<id>`。
- `dinov2_vits14`、`dinov2_vitl14`：余弦规则与现有 embedding 相同，阈值 `0.999`。失败文案同时含 `embedding cosine similarity low` 和 `0.999`。不读 `test/golden/dinov2_feature_embedding.json`。黄金是 `test/golden/trt_<id>.json`。输入图与分类相同。
- 五个深度 id：`depth_map` 为 float 单通道、全分辨率。相对误差是逐像素 `|当前 − 黄金| / max(|黄金|, 1e-6)` 的平均。平均值大于 `0.02` 时失败，失败文案同时含 `depth error` 和 `0.02`。尺寸不同、出现非有限值、没有有限像素，或黄金里没有绝对值大于 `1e-6` 的像素，直接失败，且该失败不写成重合通过。不比较 `colorized_depth_map`。`depth_vits14`、`depth_vitb14`、`depth_vitl14` 用 `create_depth_anything_estimator`，`metric3d_512`、`metric3d_1088` 用 `create_metric3d_estimator`。输入图都是 `demo_data/model_test_input/mono_depth_estimation/0000000005.png`。黄金是 `test/golden/trt_<id>.json`。

creator 用该 id 在 `src/factory` 里已经绑定的那个：分类三个、`create_yolov5_detector`、`create_yolov6_detector`、`create_nanodet_detector`、`create_yolov7_detector`、`create_yolov8_detector`、`create_libface_detector`、`create_centerface_detector`、`create_bisenetv2_segmentor`、`create_pphuman_segmentor`、`create_hrnet_segmentor`、`create_msocrnet_segmentor`、`create_dbtext_detector`、三个 enhancement creator、`create_superpoint_extractor`、`create_dinov2_feature_extractor`、`create_modnet_segmentor`、`create_ppmatting_segmentor`，以及上面两个深度 creator。

缺 engine 与缺权重走同一句 `weights not available`。只选中这一条且因此跳过时，现有 `golden_process_exit` 使进程退出码 77，标准输出含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。`MORTRED_CI_REQUIRE_WEIGHTS=1` 时同一处改为失败，退出码 1，标准输出不含 `NUMERICAL_NOT_RUN`。缺黄金、规则失败，都是退出码 1。不因为看不到 `/dev/nvidia*` 而跳过；本机 GPU 经 `/dev/dxg`。

代价：full 构建才能链上 TensorRT，这 38 条才会出现在 `_bin/model_golden_test`。没有 CPU 用例的 id 要逐个进程生成自己的 `trt_<id>` 黄金，`msocrnet` 除外，所以本条交不出 `msocrnet` 的数值通过。已有 CPU 黄金的 fp16 engine（以及 `libface` 产品 toml 的输入边是 320×240，而现有 CPU 用例读的是 640×480 配置）可能过不了现有分数差或框数门；过不了就是退出码 1，不能放宽阈值。37 条数值要分 37 个进程跑，显存按一次一个 engine 占用，主机内存不叠 38 个模型。

### 否决

一个进程里循环或用一条 `TrtProductGolden.*` 过滤跑完全部 38 个 id。代价看起来是少启动几次，但和「一个进程不要同时跑全部 38 条」冲突，而且会把 `msocrnet` 卷进同一次地址空间。否决。

把 38 条都做成 `Yolov8TrtDetection` 那样只比同类 IoU、不比分数，或者让这 38 条调用 `load_model_cfg` / `force_cpu_backend`，或者把未合入的 P1-8 CPU 用例和黄金接进来。第一条会改掉已有 CPU 用例的失败条件，并把已关闭的 `yolov8s` 规则扩到其他 id。第二条会把 backend 改成 cpu，产品 engine 不再按产品 toml 的 `device=gpu` 跑。第三条依赖未合入工作树，还等于在本条里补 P1-8。三条都否决。

## 改动面

- `test/model_golden_test.cc`：在 `MORTRED_HAS_TRT` 内登记 38 条 `TrtProductGolden.<id>`，并包含深度工厂头。`Yolov8TrtDetection` 的函数体不动。
- `test/model_golden_registry.h`：增加只剥路径、不改 backend 的产品配置加载；为没有 CPU 用例的检测、embedding、深度增加会打印阈值的比较。已有的 `expect_scores`、`expect_boxes`、`expect_text_regions`、`expect_fingerprint`、`expect_keypoints`、`expect_embeddings`、分割和抠图重合比较的文案与阈值不改。
- `test/golden/trt_<id>.json`：仅没有 CPU 用例、且本机允许跑数值的那些 id，由 `MORTRED_UPDATE_GOLDEN=1` 的测试进程写出。不含 `trt_msocrnet`，不含任何 CPU 茎名。

## 明确不改

- `Yolov8TrtDetection`，以及 `test/golden/yolov8_detection.json` 的写入路径。
- `force_cpu_backend`、`load_model_cfg`、`golden_process_exit`、`MORTRED_SKIP_OR_FAIL_WEIGHTS`。
- 现有 `model_golden.*` 用例的配置、输入图、黄金茎名和失败文案。
- `test/CMakeLists.txt`。不新增会一次跑多条 `TrtProductGolden` 的 ctest。现有 `Yolov8TrtDetection` 与 `model_golden.yolov8_detection` 两条 ctest 保持原过滤。
- `conf/model/**`、`conf/ci/**`、`conf/server/**`、`conf/packs/**`。
- `src/models`、`src/factory`、`src/server`、`src/apps`、`src/control`。
- HTTP、OpenAPI、进程模型。gateway 仍是 `:8080`，supervisor 仍是 `:8787`。不拉起常驻服务。
- `centerface` 产品 toml 不补 `input_names`，不改 engine 输入名 `input.1`。
- 不跑 `golden_drift_check.py --record`。不从 `G:\Codex\mortred-cowork-r11-p1-8` 拷贝黄金或头文件。

## 契约

无 HTTP 变化，无模型输入输出张量契约变化，无产品 toml 变化。产品 toml 的 `type` 仍是 `tensorrt`，`model_file_path` 仍是下表 engine。测试加载时不把 `device` 改成 `cpu`。

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

数值契约按「做法」里的分组。已有 CPU 用例的 id，失败文案与该 CPU 用例相同。没有 CPU 用例的 id，检测失败输出含 `0.5`，分割和抠图含 `0.90`，embedding 含 `0.999`，深度含 `0.02`。

## 性能与资源

不新增常驻进程，不占用 gateway 或 supervisor。每条用例在自己的进程里加载一个 engine，进程结束释放显存。本机 GPU 是 RTX 2070 SUPER，8GiB，TensorRT 10.3，CUDA 12.6；缺 `/dev/nvidia*` 不等于没有 GPU。一次只留一个 engine，避免 8GiB 上叠多个大 engine。

`yolov8x`、`dinov2_vitl14`、`hrnet`、`ppmatting_1024`、`metric3d_1088` 单次会拉高显存和主机内存，仍然每个 id 单独一个进程。`msocrnet` 的产品输入尺寸是 1024×2048；本机约 19GiB 内存上不跑它的推理。cpu profile 不链 TensorRT，full-cpu 构建的二进制里不会有 `TrtProductGolden`。验收二进制必须是 full 构建落到 `_bin/model_golden_test` 的那一个，不是 `build/full-cpu/test/model_golden_test`。

## 验收标准

下列命令都在仓库根目录的 WSL 里执行。`<bin>` 是 `_bin/model_golden_test`。先用 `cmake --preset full && cmake --build --preset full --target model_golden_test` 得到这个二进制。未写明的环境变量都不要设置。每条数值命令的 `--gtest_filter` 只有一个测试名。

1. `<bin> --gtest_list_tests` 退出码 0。输出含 `TrtProductGolden.` 下全部 38 个 id，每个一条；含 `Yolov8TrtDetection.` 和 `MatchesCpuGoldenBoxes`；含 `model_golden.yolov5_detection`、`model_golden.nanodet_detection`、`model_golden.yolov8_detection`。输出不含 `TrtProductGolden.yolov8s`。输出不含 `model_golden.nanodet_416`、`model_golden.yolov7x`、`model_golden.yolov8n`、`model_golden.yolov8l`、`model_golden.yolov8x`、`model_golden.pphuman_lite`、`model_golden.pphuman_server`、`model_golden.hrnet`、`model_golden.msocrnet`、`model_golden.dinov2_vits14`、`model_golden.dinov2_vitl14`、`model_golden.ppmatting_1024`、`model_golden.ppmatting_resnet34`、`model_golden.ppmatting_v2`、`model_golden.depth_vits14`、`model_golden.depth_vitb14`、`model_golden.depth_vitl14`、`model_golden.metric3d_512`、`model_golden.metric3d_1088`。
2. 用一段脚本读「契约」表里 38 份产品 toml。`type = "tensorrt"` 且文件文本含该行 engine 时，每行打印 `ok <id> type=tensorrt <engine>`，脚本退出码 0。输出含 38 行 `ok`，不含 `MISMATCH`，不含 `ok yolov8s`。
3. `grep -a -F "conf/model/scene_segmentation/msocrnet/msocrnet.toml" <bin>` 退出码 0，输出含这条路径。不执行 `--gtest_filter=TrtProductGolden.msocrnet`。
4. 已有 CPU 黄金、且对应 engine 都在时，下面 19 条各自一个进程，过滤名是 `TrtProductGolden.` 加 id：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`yolov7`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vitb14`、`modnet`、`ppmatting_512`。每条退出码 0，标准输出含 `NUMERICAL_PASSED`，并含该 id 契约表中的 `product toml` 行（`type=tensorrt`、`device=gpu`、该 engine 路径）。标准输出不含 `device=cpu`，不含 `NUMERICAL_NOT_RUN`。
5. 没有 CPU 用例、且不是 `msocrnet` 的 18 个 id，先各自一个进程执行 `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.<id>`。每条退出码 0，标准输出含 `golden updated`，不含 `NUMERICAL_PASSED`。然后去掉该环境变量，再各自一个进程跑同一过滤。每条退出码 0，标准输出含 `NUMERICAL_PASSED` 和该 id 的 `product toml` 行（`type=tensorrt`、`device=gpu`、该 engine）。不含 `device=cpu`，不含 `NUMERICAL_NOT_RUN`。这 18 个 id 是：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。
6. `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.mobilenetv2` 退出码 1。输出含 `refusing to overwrite cpu golden`，不含 `NUMERICAL_PASSED`。`test/golden/mobilenetv2_classification.json` 字节不变。
7. 临时让 `weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16` 不存在，不设 `MORTRED_CI_REQUIRE_WEIGHTS`，执行 `<bin> --gtest_filter=TrtProductGolden.yolov5l`。退出码 77，标准输出含 `NUMERICAL_NOT_RUN` 和 `weights not available`，不含 `NUMERICAL_PASSED`。恢复该 engine。
8. 同一 engine 仍不存在，执行 `MORTRED_CI_REQUIRE_WEIGHTS=1 <bin> --gtest_filter=TrtProductGolden.yolov5l`。退出码 1，输出含 `weights not available`，不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。恢复该 engine。
9. 把 `test/golden/mobilenetv2_classification.json` 里一个分数改到与原值差大于 `0.001`，尺寸和 `class_id` 不动。`<bin> --gtest_filter=TrtProductGolden.mobilenetv2` 退出码 1，输出含 `score mismatch at`，不含 `NUMERICAL_PASSED`。改回后同一命令退出码 0，输出含 `NUMERICAL_PASSED`。
10. 把 `test/golden/yolov5_detection.json` 里一个框移到与原框 IoU < `0.5` 的位置。`<bin> --gtest_filter=TrtProductGolden.yolov5l` 退出码 1，输出含 `low IoU for yolov5_detection`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
11. 把 `test/golden/trt_nanodet_416.json` 里一个框的 `class_id` 改成另一个整数。`<bin> --gtest_filter=TrtProductGolden.nanodet_416` 退出码 1，输出含 `class_id`，不含 `NUMERICAL_PASSED`。改回后，再把该框移到 IoU < `0.5`。退出码 1，输出同时含 `low IoU` 和 `0.5`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
12. 把 `test/golden/bisenetv2_segmentation.json` 的 `values` 每个整数加 1，不改 `height`、`width` 和长度，且仍至少两个类别值。`<bin> --gtest_filter=TrtProductGolden.bisenetv2` 退出码 1，输出同时含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
13. 对 `test/golden/trt_pphuman_lite.json` 做与验收 12 相同的加 1。`<bin> --gtest_filter=TrtProductGolden.pphuman_lite` 退出码 1，输出同时含 `mask overlap` 和 `0.90`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
14. 把 `test/golden/modnet_matting.json` 的前景（值 ≥ `128`）换成与改前前景交集为 0、且前景数 ≥ 2 的值，不改 `height`、`width` 和长度。`<bin> --gtest_filter=TrtProductGolden.modnet` 退出码 1，输出同时含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
15. 把 `test/golden/dbnet_text_detection.json` 里一个区域移到与原区域 IoU < `0.5`。`<bin> --gtest_filter=TrtProductGolden.dbnet` 退出码 1，输出含 `low IoU for dbnet_text_detection`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
16. 改 `test/golden/realesrgan_enhancement.png`，使均值绝对差大于 `1.0`。`<bin> --gtest_filter=TrtProductGolden.realesrgan` 退出码 1，输出含 `fingerprint drift for realesrgan_enhancement`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
17. 改 `test/golden/superpoint_feature_point.json` 的点坐标，使匹配比例低于 `0.9`。`<bin> --gtest_filter=TrtProductGolden.superpoint` 退出码 1，输出含 `keypoint match ratio low for superpoint_feature_point`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
18. 把 `test/golden/dinov2_feature_embedding.json` 换成等长、余弦 < `0.999` 的向量。`<bin> --gtest_filter=TrtProductGolden.dinov2_vitb14` 退出码 1，输出含 `embedding cosine similarity low for dinov2_feature_embedding`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
19. 把 `test/golden/trt_dinov2_vits14.json` 换成等长、余弦 < `0.999` 的向量。`<bin> --gtest_filter=TrtProductGolden.dinov2_vits14` 退出码 1，输出同时含 `embedding cosine similarity low` 和 `0.999`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。
20. `test/golden/trt_depth_vits14.json` 不改 `height`、`width` 和长度，把 `values` 每个数乘 2，且仍是有限 float、并有绝对值大于 `1e-6` 的像素。`<bin> --gtest_filter=TrtProductGolden.depth_vits14` 退出码 1，输出同时含 `depth error` 和 `0.02`，不含 `NUMERICAL_PASSED`。改回后退出码 0，输出含 `NUMERICAL_PASSED`。

验收 9 至 20 改过的黄金，测完按原字节恢复。验收 5 的更新那一次退出码 0 不算数值通过。

## 测试义务

QA 按验收 1 至 20 做黑盒。验收 4 的 19 条和验收 5 的 18 条必须逐条单独进程跑完，不能并成一个过滤。不跑 `TrtProductGolden.msocrnet`。验收 9 至 20 覆盖分类、两组检测、已有 CPU 用例的分割和抠图、没有 CPU 用例的分割、OCR、增强、特征点、两组 embedding、深度；不必对 37 个 id 都做破坏。

白盒核对：38 条都不调用 `force_cpu_backend` 和 `load_model_cfg`；已有 CPU 用例的 19 条调用的比较函数与该 CPU 用例相同，黄金茎名与「做法」一致；`libface` 与 `centerface` 的比较带 landmarks；`yolov8n`、`yolov8l`、`yolov8x` 不调用 `Yolov8TrtDetection` 的比较，也不读 `yolov8_detection` 黄金；深度比较只读 `depth_map`；`msocrnet` 的配置字符串是 `conf/model/scene_segmentation/msocrnet/msocrnet.toml`，比较是分割掩码重合 `0.90`。跳过文案仍含 `weights not available`。没有从 `G:\Codex\mortred-cowork-r11-p1-8` 引入符号或黄金。

## 自测命令

在仓库根目录的 WSL 里执行。`<bin>` 同上。期望退出码如下。数值命令都是一个进程一个过滤名。

| 命令 | 前提 | 期望退出码 |
| --- | --- | --- |
| `cmake --preset full && cmake --build --preset full --target model_golden_test` | 本机能链 TensorRT 10.x | 0 |
| `<bin> --gtest_list_tests` | 上一步的二进制 | 0，且满足验收 1 的出现与不出现 |
| 验收 2 的 toml 检查 | 产品 toml 未改 | 0，输出含 38 行 `ok`，不含 `MISMATCH` |
| `grep -a -F "conf/model/scene_segmentation/msocrnet/msocrnet.toml" <bin>` | 同上二进制 | 0，输出含该路径 |
| 验收 5 的 18 条 `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.<id>`，每条单独进程 | 该 id 的 engine 在；不包含 `msocrnet` | 每条 0，输出含 `golden updated`，不含 `NUMERICAL_PASSED` |
| 验收 4 的 19 条与验收 5 去掉更新变量后的 18 条，每条 `<bin> --gtest_filter=TrtProductGolden.<id>` | engine 与黄金都在；未设置 `MORTRED_UPDATE_GOLDEN` 与 `MORTRED_CI_REQUIRE_WEIGHTS` | 每条 0，输出含 `NUMERICAL_PASSED` 与该 id 的 `product toml` 行 |
| `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=TrtProductGolden.mobilenetv2` | 无 | 1，输出含 `refusing to overwrite cpu golden`，不含 `NUMERICAL_PASSED` |
| `<bin> --gtest_filter=TrtProductGolden.yolov5l` | `yolov5l.engine.static_bs1.fp16` 不存在；未设置 `MORTRED_CI_REQUIRE_WEIGHTS` | 77，输出含 `NUMERICAL_NOT_RUN` 和 `weights not available`，不含 `NUMERICAL_PASSED` |
| `MORTRED_CI_REQUIRE_WEIGHTS=1 <bin> --gtest_filter=TrtProductGolden.yolov5l` | 同一 engine 不存在 | 1，输出含 `weights not available`，不含 `NUMERICAL_NOT_RUN` |

不执行 `<bin> --gtest_filter=TrtProductGolden.msocrnet`。不执行 `<bin> --gtest_filter=TrtProductGolden.*`。

## RD 不得自行决定

- 38 个 id、测试名 `TrtProductGolden.<id>`、creator、输入图、产品 toml 和 engine 路径。
- 已有 CPU 用例的 19 个 id 的黄金茎名、阈值和失败文案。分类仍是 `class_id` 加分数差 `1e-3`，不是只比 top-1。检测和 OCR 的 IoU 仍是 `0.5`、分数差仍是 `0.01`。人脸 landmarks 距离仍是 `3.0`。分割和抠图仍是 `0.90`，抠图前景仍是 `128`。增强均值绝对差仍是 `1.0`。特征点匹配距离仍是 `3.0`、比例仍是 `0.9`。`dinov2_vitb14` 余弦仍是 `0.999`。
- 没有 CPU 用例的检测必须在失败输出打印 `0.5`，embedding 打印 `0.999`，分割和抠图打印 `0.90`，深度打印 `0.02`。深度平均相对误差的公式不得改。
- 用 `yolov8s` 的用例、配置或黄金代替 `yolov8n`、`yolov8l`、`yolov8x` 中的任何一个，或修改 `Yolov8TrtDetection`。
- 给这 19 个没有 CPU 用例的 id 加 CPU 用例，或引用未合入工作树的符号和黄金。
- 手写黄金、复制别的 id 的黄金、或运行 `golden_drift_check.py --record`。`MORTRED_UPDATE_GOLDEN=1` 那一次退出码 0 不得记成数值通过。
- 缺 engine 时不得打印 `NUMERICAL_PASSED`。不得把跳过文案改成不含 `weights not available` 的句子。
- 一个进程跑多于一个本条 id。本机不得跑 `TrtProductGolden.msocrnet`。不得因 OOM 从名单删除 `msocrnet`。
- 调用 `force_cpu_backend` 或 `load_model_cfg` 来跑这 38 条。因缺少 `/dev/nvidia*` 而跳过。
- 改产品 toml、HTTP、张量契约、gateway、supervisor、engine 生成，或给 `centerface` 改输入名。
- 用 `build/full-cpu/test/model_golden_test` 或 full-cpu 构建作为本条二进制。
