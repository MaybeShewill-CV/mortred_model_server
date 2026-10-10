# 技术方案

- 问题: P1-11
- 版本: v1
- 对应列表: v3

## 边界

只改这四条已经在基线 `806813aa` 上的 CPU 黄金。测试名、配置、输入图、onnx 和 catalog id 的对应是：

| 测试名 | catalog id | 配置 | onnx | 输入图 |
| --- | --- | --- | --- | --- |
| `model_golden.bisenetv2_segmentation` | `bisenetv2` | `conf/ci/bisenetv2_onnx_hosted.toml` | `weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.static_bs1.onnx` | `demo_data/model_test_input/scene_segmentation/cityscapes_test.png` |
| `model_golden.pphuman_segmentation` | `pphuman_mobile` | `conf/ci/pphuman_onnx_hosted.toml` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.static_bs1.onnx` | `demo_data/model_test_input/scene_segmentation/human_image.jpg` |
| `model_golden.modnet_matting` | `modnet` | `conf/ci/modnet_onnx_hosted.toml` | `weights/matting/modnet/modnet_hrnet_w18.static_bs1.onnx` | `demo_data/model_test_input/matting/matting_test.jpg` |
| `model_golden.ppmatting_matting` | `ppmatting_512` | `conf/ci/ppmatting_onnx_hosted.toml` | `weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.static_bs1.onnx` | `demo_data/model_test_input/matting/matting_test.jpg` |

现在 `compare_scene_segmentation` 与 `compare_matting` 只调用 `expect_fingerprint`：把输出缩成 64×64、做 min-max，再和 `test/golden/<用例名>.png` 比均值差，阈值是 `k_fingerprint_diff`（1.0）。基线 `test/golden_baseline.json` 里这四条的键是 `bisenetv2_segmentation.png`、`pphuman_segmentation.png`、`modnet_matting.png`、`ppmatting_matting.png`。

改完后，这四条在全分辨率掩码上算重合。平均或前景重合 `< 0.90` 时该条失败，失败输出含 `mask overlap` 和 `0.90`。`>= 0.90` 通过。

比较仍在现有 `model_golden_test` 进程里完成，仍走 `run_case`：缺权重走 `MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available")`，有权重则 `load_model_cfg`（去掉 `../` 并强制 cpu）、现有 creator、`init`、读上面那张图、`run`、比较。不新写第二套退出码。单条缺权重仍跳过。这四条都被选中且都因缺权重跳过时，进程退出 77，标准输出一行 `NUMERICAL_NOT_RUN`。设置 `MORTRED_CI_REQUIRE_WEIGHTS=1` 时，同一处改为失败，退出码 1，不打印 `NUMERICAL_NOT_RUN`。缺黄金、类型不符、重合低于 0.90，都是失败，退出码 1。静默通过不满足本条。

`MORTRED_UPDATE_GOLDEN=1` 只负责写出这四份 JSON 并跳过。该次可以退出 0，跳过文案表示黄金已更新，标准输出不含 `NUMERICAL_PASSED`。数值通过是去掉该环境变量再跑。

P1-8 挂起且未合入。本条在基线头文件里自己实现比较，不引用、不编译、不要求工作树里存在 `compare_scene_segmentation_overlap`、`compare_matting_overlap` 或任何未合入符号。

不覆盖 P0-2、P1-8、P1-12、D-8 至 D-12。不重开已关闭原句。`yolov8s` 已关闭的同类 IoU 门仍只覆盖 `yolov8s`。

## 做法

### 推荐

改 `compare_scene_segmentation` 与 `compare_matting` 的比较内容。基线上这两个函数只被上面四条用例经现有宏调用。改完后它们不再调用 `expect_fingerprint`。

两条分割比较 `segmentation_result`，全分辨率，类型必须是 `CV_32SC1`。黄金是 UTF-8 JSON：对象含整数 `height`、整数 `width`，以及按行展开的整数数组 `values`，长度等于 `height * width`，元素是类别 id。对黄金里出现过的每个类别算 IoU（该类别像素的交集 / 并集），再取平均。平均 `< 0.90` 失败。尺寸不同、不是 `CV_32SC1`、或黄金里少于两个不同类别值，直接失败，这种失败不写成重合通过。

两条抠图比较 `matting_result`。`modnet` 的现输出是 `CV_8UC1`，`ppmatting_512` 的现输出是 `CV_32SC1`，数值都是乘 255 之后的整数 alpha。比较同时接受这两种单通道整型，按像素整数值读，不改模型输出类型。前景是值 `>= 128` 的像素。前景 IoU `< 0.90` 失败。黄金前景像素数小于 2、当前类型不是这两种整型之一、或尺寸不同，直接失败。

失败输出在重合低于 0.90 时含字面量 `mask overlap` 和 `0.90`。缺权重的跳过文案仍含 `weights not available`。

黄金由该条自己的 CPU onnx 在上表现有输入图上生成，写入 `test/golden/<用例名>.json` 后提交。分割黄金至少两个不同类别值。抠图黄金前景像素数至少为 2。不手写、不复制另一条的黄金。

`test/golden/` 不再保留这四份 `.png`。`test/golden_baseline.json` 只把这四个 png 键换成同用例名的 json 键，哈希是新文件原始字节的 sha256，算法与 `scripts/golden_drift_check.py` 一致。`case_names` 的名单和顺序不变，`case_count` 仍是 30。其余 `golden_sha256` 条目保持原哈希。`golden_count` 仍是 26。不要执行会重写整份基线的 `--record`：当前工作树若缺其他黄金文件，`--record` 会把那些条目从基线删掉。

代价：四份全分辨率整数 JSON 比现在的 64×64 png 大，占主机磁盘；四条 CPU 推理仍是现有用例的一次 `run`。不新增进程，不生成 engine，不占显存。

### 否决

继续用 64×64 fingerprint，或在 fingerprint 图上再算一次重合。fingerprint 先做 min-max 再缩小，类别图或 alpha 错了仍可能均值差 ≤ 1.0。冻结句要求这四个 id 失败于掩码重合低于测试写明的阈值，64×64 图做不到这件事。

依赖 P1-8 未合入的 overlap 函数，或顺手把增强、SAM、`compare_raw_mat` 以及其他 id 的 fingerprint 改成掩码重合。P1-8 挂起且未合入，基线上没有那些符号。其他 id 的 fingerprint 不在本条冻结句里。

改后处理、HTTP 或产品 toml，让服务自己返回重合分数，或让这四条加载 `type=tensorrt` 的产品文件。冻结句只要求现有 CPU 用例在重合低于阈值时失败。产品文件的 `type` 仍是 `tensorrt`，路径仍是 engine；现有 `force_cpu_backend` 只改 `device` / `compute_backend`。TensorRT 同规则比较是 P1-12。

## 改动面

- `test/model_golden_registry.h`：`compare_scene_segmentation` 与 `compare_matting` 改为上面的全分辨率重合。阈值 `0.90` 和抠图前景 `128` 写在这个测试代码里。
- `test/golden/bisenetv2_segmentation.json`、`test/golden/pphuman_segmentation.json`、`test/golden/modnet_matting.json`、`test/golden/ppmatting_matting.json`：由对应 CPU onnx 生成。
- `test/golden_baseline.json`：只替换这四个黄金键和对应 sha256。

## 明确不改

- `test/model_golden_test.cc` 里这四条的测试名、`GOLDEN_SCENE_SEGMENTATION_CASE` / `GOLDEN_MATTING_CASE` 登记、配置路径和输入图。
- `expect_fingerprint`，以及 `compare_enhancement`、`compare_raw_mat`、`sam_prompt_prediction`、`sam_automask_generation` 对它的调用。`enlightengan_enhancement.png`、`attentivegan_enhancement.png`、`realesrgan_enhancement.png`、`fastsam_segmentation.png`、`sam_prompt_prediction.png`、`sam_automask_generation.png` 及其基线哈希。
- `yolov8_detection`、`yolov8_onnx_detection`、`Yolov8TrtDetection`，以及 `k_box_iou_thresh`。
- `golden_process_exit`、`MORTRED_SKIP_OR_FAIL_WEIGHTS`、`test/CMakeLists.txt`。`model_golden_test` 仍自带 `main`，不链接 `gtest_main`，`SKIP_RETURN_CODE` 仍是 77。
- 四份 `conf/ci/*_onnx_hosted.toml`、`conf/model/**`、`conf/server/**`。
- `src/models`、`src/factory`、`src/server`、`src/apps`、`src/control`、HTTP 契约、OpenAPI、pack、`mortredctl`、engine 生成。
- 网关 `:8080` 与 supervisor `:8787` 的进程模型。
- P1-8 的 19 个 id、`msocrnet` 黄金，以及那条 19 用例过滤。

## 契约

无 HTTP 变化，无模型输入输出张量契约变化，无产品 toml 变化。

这四条测试配置保持现状：`backend.type` 为 `onnx`，`backend.device` 为 `cpu`，`model_file_path` 为上表该 id 的 onnx。加载时仍强制 cpu。用例不打开 `conf/model` 里的产品文件。产品文件仍是 `type = "tensorrt"`，路径仍是 `.engine`。

数值契约：

- `bisenetv2`、`pphuman_mobile`：`segmentation_result` 为 `CV_32SC1` 类别图；黄金中每个出现过的类别的 IoU 取平均；平均 `>= 0.90` 通过，`< 0.90` 失败。黄金至少两个不同类别值。
- `modnet`、`ppmatting_512`：`matting_result` 为 `CV_8UC1` 或 `CV_32SC1` 的 0–255 整数 alpha；值 `>= 128` 的前景 IoU `>= 0.90` 通过，`< 0.90` 失败。黄金前景像素数至少为 2。

重合失败的输出含 `mask overlap` 与 `0.90`，且不含 `fingerprint drift`。缺权重的跳过文案含 `weights not available`。

## 性能与资源

不新增常驻进程，不加载 engine，不占用显存。gpu profile 上这四条仍强制 cpu。cpu profile 不链 TensorRT，这四条仍可编译、可列出、可跑。

每个用例在现有测试进程里 `init`、跑一张图、结束即释放。缺权重在 `init` 之前返回。全分辨率 JSON 只增加主机磁盘；不为此另起进程，也不把这四条从 `model_golden_test` 里摘出去。

## 验收标准

下列命令都在仓库根目录执行。二进制是已有 `cmake --preset full-cpu` 建成的 `./build/full-cpu/test/model_golden_test`。下面用 `<bin>` 表示它。不配置、不构建 `full`。四条过滤记为 `<filter>`，即 `model_golden.bisenetv2_segmentation:model_golden.pphuman_segmentation:model_golden.modnet_matting:model_golden.ppmatting_matting`。

1. `<bin> --gtest_list_tests` 退出码 0。`model_golden.` 下仍能看到 `bisenetv2_segmentation`、`pphuman_segmentation`、`modnet_matting`、`ppmatting_matting`，各一条。没有新的测试名，也没有新的 TensorRT 套件名。
2. 四份 `conf/ci` toml 仍是 `type` 为 `onnx`、`device` 为 `cpu`，`model_file_path` 仍指向上表那四个 onnx。这四个 id 的 `conf/model` 产品文件仍是 `type = "tensorrt"`，路径仍是 `.engine`。
3. 四个 onnx 与四份 JSON 黄金都在，不设 `MORTRED_UPDATE_GOLDEN`、不设 `MORTRED_CI_REQUIRE_WEIGHTS` 时，`<bin> --gtest_filter=<filter>` 退出码 0，标准输出含 `NUMERICAL_PASSED`。
4. 只选中这四条，且四个 onnx 都不存在，不设 `MORTRED_CI_REQUIRE_WEIGHTS`：同一过滤退出码 77，标准输出有一行 `NUMERICAL_NOT_RUN`。没有一条被记成通过。
5. 同一前提下设置 `MORTRED_CI_REQUIRE_WEIGHTS=1`，只跑 `model_golden.bisenetv2_segmentation`，且 `bisenetv2_cityscapes.static_bs1.onnx` 不存在：退出码 1，标准输出不含 `NUMERICAL_NOT_RUN`，失败信息含 `weights not available`。
6. 任选 `bisenetv2_segmentation` 或 `pphuman_segmentation`。黄金是 `test/golden/<用例名>.json`，编码是 UTF-8 JSON，含整数 `height`、整数 `width` 和按行展开的整数数组 `values`。不改 `height`、`width` 和数组长度，把 `values` 的每个整数加 1。加 1 之后仍至少两个不同类别值。`model_golden.<用例名>` 退出码 1，失败输出含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`，也不是尺寸或类型不符。把该文件恢复成原字节后，退出码 0。
7. 任选 `modnet_matting` 或 `ppmatting_matting`。黄金格式与验收 6 相同，`values` 是 0–255 的 alpha。不改 `height`、`width` 和数组长度，把 `values` 全部写成 0，再把下标 0 写成 255。前景像素数为 1。该条退出码 1，失败输出含 `mask overlap` 和 `0.90`，不含 `fingerprint drift`，也不是尺寸或类型不符。恢复原字节后，退出码 0。
8. `python scripts/golden_drift_check.py --check` 退出码 0。基线里不再有这四个 `.png` 键，改为这四个 `.json` 键。`case_names` 与 `case_count` 不变。`enlightengan_enhancement.png`、`fastsam_segmentation.png`、`sam_prompt_prediction.png`、`yolov8_detection.json` 的哈希与改前基线相同。`test/golden/` 里除这四条 png 换成 json 之外，不删除其他 id 的黄金。

## 测试义务

QA 按验收 1 至 8 做黑盒。验收 6 两条分割做一条，验收 7 两条抠图做一条。验收 6、7 改黄金，结束后恢复原字节。`MORTRED_UPDATE_GOLDEN=1` 那一次退出 0 不算验收 3 通过。

白盒核对：`compare_scene_segmentation` 与 `compare_matting` 不再调用 `expect_fingerprint`；比较的是 `segmentation_result` 与 `matting_result`，不比较 `colorized_seg_mask`；源码里写明 `0.90` 和前景 `128`；这两条比较不引用 P1-8 未合入的 overlap 符号。这四条仍经 `run_case`、`load_model_cfg`、`force_cpu_backend`。`expect_fingerprint` 的其他调用还在。`golden_process_exit` 的 77 分支还在。分割黄金至少两个类别值，抠图黄金前景像素数至少为 2，黄金来自该条 CPU onnx，不是手写或复制。

## 自测命令

在仓库根目录执行。`<bin>` 与 `<filter>` 同验收标准。二进制尚未建成时，只使用 `cmake --preset full-cpu` 与 `cmake --build --preset full-cpu --target model_golden_test`。不构建 `full`。

| 命令 | 前提 | 期望退出码 |
| --- | --- | --- |
| `<bin> --gtest_list_tests` | 已用 `full-cpu` 建成 | 0 |
| `<bin> --gtest_filter=<filter>` | 四个 onnx 与四份 JSON 都在；未设置 `MORTRED_UPDATE_GOLDEN` 与 `MORTRED_CI_REQUIRE_WEIGHTS` | 0，且标准输出含 `NUMERICAL_PASSED` |
| `<bin> --gtest_filter=<filter>` | 这四个 onnx 都不存在；未设置 `MORTRED_CI_REQUIRE_WEIGHTS` | 77，且标准输出含 `NUMERICAL_NOT_RUN` |
| `MORTRED_CI_REQUIRE_WEIGHTS=1 <bin> --gtest_filter=model_golden.bisenetv2_segmentation` | `bisenetv2_cityscapes.static_bs1.onnx` 不存在 | 1，且标准输出不含 `NUMERICAL_NOT_RUN` |
| `python scripts/golden_drift_check.py --check` | 基线只替换了这四个键，其余黄金文件都在 | 0 |

生成黄金的命令是 `MORTRED_UPDATE_GOLDEN=1 <bin> --gtest_filter=<filter>`。它退出码可以是 0，标准输出不含 `NUMERICAL_PASSED`。它不是上表第二行，不能当作数值通过。只跑这四条过滤。

## RD 不得自行决定

- 四个 id、测试名、配置、输入图、onnx 路径。
- 阈值 `0.90`，抠图前景 `128`。分割用黄金中各类别 IoU 的平均；抠图用前景 IoU。通过线是 `>= 0.90`，失败线是 `< 0.90`。
- 把 `modnet` 的 `CV_8UC1` 或 `ppmatting_512` 的 `CV_32SC1` 改成另一种输出类型来迁就比较。
- 手写黄金、全 0 掩码、单一类别、前景少于 2 像素，或把另一个 id 的黄金复制过来。
- 调用或等待 P1-8 未合入的 overlap 符号。给这四条以外的 fingerprint 用例改成掩码重合。
- 把缺权重或然缺黄金写成通过；改掉 `weights not available`、退出码 77 或 `NUMERICAL_NOT_RUN`。
- 用 `--record` 改写其他黄金的哈希，或从基线删掉其他 id 的黄金。
- 加载产品 TensorRT toml，生成 engine，构建 `full`，跑 19 条同进程过滤，或生成 `msocrnet` 黄金。
- 改 HTTP、产品 toml、模型张量契约、进程和 engine。
