# 技术方案

- 问题: P0-2
- 版本: v2
- 对应列表: v3

## 边界

只覆盖这一条：不设置 `MORTRED_PACK`，默认 pack 仍是 `conf/packs/demo.toml`。范围内 39 个 id 里，某一个产品 engine 缺失时，`mortredctl doctor --strict` 退出码为 1，失败输出点名该 engine 的仓库相对路径，并有一行 `next:`。执行该行，在本机本卡生成这一个 id 的产品 engine。39 个 id 以冻结名单为准，含 `yolov8s` 与 `metric3d_512`。

产品 engine 的路径以 `conf/product_static_bs1.json` 里该 id 的 `engine` 为准。范围内产品 toml 的 `model_file_path` 已是 `../` 加上这段路径。`yolov8s` 为 `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16`。`metric3d_512` 为 `weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16`。`mobilenetv2` 的产品 engine 是表内 `weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32`，与 demo pack 所用的 CPU MNN 配置无关。表内 fp32 的六条是 `mobilenetv2`、`resnet`、`densenet`、`bisenetv2`、`attentive_gan`、`realesrgan`，其余为 fp16。

文件不存在，或文件大小为 0，都算缺失。doctor 只做存在性检查，不调用 trtexec。生成仍走现有本机 TensorRT 转换，gpu profile 为 CUDA 12.x 与 TensorRT 10.x。cpu profile 不链 TensorRT。全库自动构建保持另选：不带 `--id` 的 `mortredctl prepare` 仍只处理当前 pack。

上一版方案审查条件：无。本条从 v1 起。

## 做法

### 推荐

未设置 `MORTRED_PACK` 时，doctor 在现有 pack 占用率与 pack TensorRT 段之后，增加一道产品 engine 门。这道门读取上述 39 个 id 在 `conf/product_static_bs1.json` 中的 `engine`。demo pack 没有 TensorRT backend 时，原有的 pack 段仍可打印 pack 内无 TensorRT；产品 engine 门另行失败，进程以退出码 1 结束，并且结束在活体检查之前。

每一个缺失 id 的失败输出都含该 `engine` 字符串，以及一行：

`next: mortredctl prepare --id <id>`

同一次运行里，缺失几个就点名几个，并各有一行上式。39 个都缺失时，一次输出含全部 39 条路径和 39 行 `next:`。只缺失一个时，输出里的 `prepare --id` 只有这一个 id。这道门与现有 pack TensorRT 门一样，在 `doctor` 与 `doctor --strict` 上都会导致退出码 1。验收命令使用冻结的 `mortredctl doctor --strict`。

`mortredctl prepare --id <id>` 在不设置 `MORTRED_PACK` 时可用。它只解析该 id 的产品 onnx 与产品 engine，用现有产品 manifest 和 `convert_trt_engines.sh --only` 的完整 engine 相对路径，在本机本卡写出这一个文件，再按存在且非空检查。它不读取 demo pack 的模型列表，不转换其余 id，不拉起常驻 catalog，也不把 `/ready` 写成这行的完成条件。`--id` 与 `--pack` 同时出现时，退出码 1，不生成文件。id 不在这 39 个之中时，退出码 1，不生成文件。产品 onnx 已在表内 `onnx` 路径、且本机有 trtexec 时，该命令退出码 0，对应 engine 存在且大小大于 0。

### 否决

把 39 个 id 写进 `conf/packs/demo.toml`，或把未设置 `MORTRED_PACK` 时的默认 pack 换成全量 pack。默认起服集合、占用率戳和未带参数的 `mortredctl prepare` 都会变成全库构建。边界要求全库自动构建保持另选。一条 `prepare --pack` 也不能只生成被点名的那一个 id。demo 里的 `mobilenetv2` 仍指向 CPU MNN 配置。

为 39 个 id 各新增一份 pack，把 `next:` 写成 `mortredctl prepare --pack` 该文件。39 份 pack 与 `product_static_bs1` 双写；选中后又要占用率戳。未设置 `MORTRED_PACK` 时，doctor 仍只看见 demo.toml，产品 engine 缺失继续退出 0。

已设置 `MORTRED_PACK` 时也改查全部 39 个 engine。冻结场景是未设置该变量。选中 `conf/packs/yolov8.toml` 或 `conf/packs/metric3d.toml` 时，现有行为已是该 pack 的 engine 缺失则退出非 0，并给出 `next: mortredctl prepare --pack` 该 pack。

## 改动面

- `scripts/mortredctl_doctor.sh`：未设置 `MORTRED_PACK` 时增加上述产品 engine 门。
- `scripts/mortredctl_prepare.sh`：接受 `--id`，并在与 `--pack` 同时出现或 id 非法时于入口失败。
- `scripts/prepare_pack.sh`：`--id` 时只转换该 id 的产品 engine。
- 新增 `scripts/test_p0_2_unset_pack_doctor.sh`：自测脚本。它只搬移或放置占位文件并调用 doctor / prepare 入口，不调用 trtexec。
- 读取 `conf/product_static_bs1.json` 即可。若在 `scripts/product_static_bs1.py` 增加只读查询，不得改表、不得执行 `--emit` 或 `--apply-models`。

## 明确不改

- `conf/packs/demo.toml`、`conf/packs/yolov8.toml`、`conf/packs/metric3d.toml`、`conf/packs/gpu_yolo_metric3d.toml`。
- `conf/product_static_bs1.json` 的内容，以及范围内产品 toml 的 `type` 与 `model_file_path`。
- 已设置 `MORTRED_PACK` 时，doctor 仍只检查该 pack 的 TensorRT engine 与占用率戳；`next:` 仍是 `mortredctl prepare --pack` 该 pack。
- 不带 `--id` 的 `mortredctl prepare` 仍只处理当前 pack。demo pack 无 TensorRT 时仍跳过转换并以退出码 0 结束。
- 占用率、校准、ASan 段、安全告警、supervisor 不可达时的活体回退。engine 都存在时，不要求 `doctor --strict` 退出 0。
- HTTP、tensor、gateway `:8080`、supervisor `:8787`、一 catalog id 一个 OS 进程。
- `src/`、cpu profile 的链接内容、CUDA 与 TensorRT 版本。
- P0-3，P1-7 至 P1-12，D-8 至 D-12。
- 已关闭的 P1-3、P1-4、D-2。不重做遗留 engine 对齐、占用率实测容差，也不改 `attentive_gan` 的输入尺寸。

## 契约

未设置 `MORTRED_PACK` 时，`mortredctl doctor --strict` 在任一上述产品 engine 缺失或为空时退出码为 1。合并后的标准输出与标准错误含该 `engine` 相对路径，并含 `next: mortredctl prepare --id <id>`。

`mortredctl prepare --id <id>` 只生成该 id 的产品 engine。成功时退出码 0，文件在表内 `engine` 路径上且大小大于 0。`--help` 退出码 0，并写出 `--id`。

HTTP 路径、请求与响应、产品 toml 的 backend `type`、tensor 形状都不变。

## 性能与资源

产品 engine 门只对 39 个路径做存在与大小检查，不占 GPU，不留下进程。`prepare --id` 一次只跑该 id 的本机转换，转换进程结束后释放显存，不常驻模型进程，也不把其余 id 的 engine 载入显存。

## 验收标准

1. 不设置 `MORTRED_PACK`。仅让 `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16` 缺失，其余 38 个产品 engine 为非空文件。`mortredctl doctor --strict` 退出码为 1。合并输出含该相对路径，含一行 `next: mortredctl prepare --id yolov8s`，且不含其它 id 的 `prepare --id`。
2. 同样不设置 `MORTRED_PACK`。仅让 `weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16` 缺失，其余为非空文件。退出码为 1。合并输出含该相对路径，含一行 `next: mortredctl prepare --id metric3d_512`，且不含其它 id 的 `prepare --id`。
3. 不设置 `MORTRED_PACK`。39 个产品 engine 全部缺失或全部为空。同一次 `mortredctl doctor --strict` 退出码为 1。合并输出含这 39 个 id 在 `conf/product_static_bs1.json` 中的 `engine` 字符串，且每个 id 各有一行 `next: mortredctl prepare --id <该 id>`。
4. 不设置 `MORTRED_PACK`。对冻结名单中每一个 id 单独让该产品 engine 缺失，其余为非空文件。每次退出码为 1，合并输出含该 id 的 `engine` 字符串和对应的 `next: mortredctl prepare --id <该 id>`，且不含其它 id 的 `prepare --id`。其中 `yolov8n` 必须出现。`mobilenetv2` 点名的路径必须是 `weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32`。
5. 不设置 `MORTRED_PACK`。把 `yolov8s` 的产品 engine 换成 0 字节文件，其余为非空文件。`mortredctl doctor --strict` 退出码为 1，合并输出含该相对路径和 `next: mortredctl prepare --id yolov8s`。
6. 不设置 `MORTRED_PACK`。`yolov8s` 的产品 onnx 已在表内 `onnx` 路径，本机有 trtexec。执行 `mortredctl prepare --id yolov8s`，退出码为 0，`weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16` 存在且大小大于 0。执行前若 `yolov8n` 的产品 engine 不存在，执行后仍不存在。
7. 同样执行 `mortredctl prepare --id metric3d_512`。退出码为 0，`weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16` 存在且大小大于 0。执行前若 `yolov8s` 的产品 engine 不存在，执行后仍不存在。
8. 同样执行 `mortredctl prepare --id yolov8n`。退出码为 0，`weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16` 存在且大小大于 0。
9. `mortredctl prepare --id not_a_catalog_id` 退出码为 1，且不新增 engine 文件。
10. `mortredctl prepare --id yolov8s --pack conf/packs/demo.toml` 退出码为 1，且不生成 `yolov8s` 的产品 engine。
11. `bash scripts/mortredctl_prepare.sh --help` 退出码为 0，输出含 `--id`。
12. 不设置 `MORTRED_PACK`。冻结名单中每一个 id，在该 id 的产品 onnx 已位于 `conf/product_static_bs1.json` 的 `onnx` 路径且本机有 trtexec 时，执行该 id 这次失败输出里的 `next:` 命令 `mortredctl prepare --id <该 id>`，退出码为 0，表内该 `engine` 存在且大小大于 0；执行前其余 id 的产品 engine 若不存在，执行后仍不存在。`yolov8n`、`yolov8l`、`yolov8x` 各自执行，一次 `yolov8s` 生成不代替这三条。缺该 id 的 onnx、缺 trtexec 或缺 GPU 时，该 id 记挂起，写明缺的环境和未跑的验收，未生成不得记成通过。该 id 的产品 engine 若大小为 0，转换前先去掉该文件，或对该次 `mortredctl prepare --id <该 id>` 传入现有 `--force`，避免被当成已存在而跳过。

## 测试义务

QA 按第 1 条到第 12 条做黑盒。doctor 比较合并后的标准输出与标准错误。第 1 条到第 5 条用搬移原文件或 0 字节文件制造缺失，结束后恢复原文件。第 6 条到第 8 条在本机已有对应产品 onnx 且 trtexec 可用时执行；缺 onnx、缺 trtexec 或缺 GPU 时，这三条记挂起，写明缺的环境和未跑的验收，不得把未生成记成通过。

第 12 条覆盖冻结名单全部 39 个 id，含 `yolov8n`、`yolov8l`、`yolov8x`，不能用 `yolov8s` 代替。每一个 id：该 id 的产品 onnx 已位于 `conf/product_static_bs1.json` 的 `onnx` 路径且本机有 trtexec 时，执行该 id 这次失败输出里的 `next:` 命令 `mortredctl prepare --id <该 id>`，退出码为 0，表内该 `engine` 存在且大小大于 0；执行前其余 id 的产品 engine 若不存在，执行后仍不存在。大小为 0 的 engine 在转换前先去掉，或对该次命令传入现有 `--force`，避免被当成已存在而跳过。缺该 id 的 onnx、缺 trtexec 或缺 GPU 时，该 id 记挂起，写明缺的环境和未跑的验收，未生成不得记成通过。

另测一条已有行为：设置 `MORTRED_PACK` 为 `conf/packs/yolov8.toml`，且 `yolov8s` 的产品 engine 缺失时，`mortredctl doctor --strict` 退出码为 1，合并输出含该 engine 相对路径，并含 `next: mortredctl prepare --pack` 指向该 pack。`metric3d.toml` 对 `metric3d_512` 做同样核对。

白盒核对：`conf/packs/demo.toml`、`conf/product_static_bs1.json`、范围内产品 toml 的 `type` 与 `model_file_path` 无内容变更；`--only` 收到的是该 id 的完整 `engine` 相对路径；未改 `src/`；未改 P0-3、P1-7 至 P1-12、D-8 至 D-12 的代码与文档。

## 自测命令

在仓库根目录执行。

| 命令 | 期望退出码 |
| --- | --- |
| `env -u MORTRED_PACK bash scripts/test_p0_2_unset_pack_doctor.sh` | 0 |
| `python3 scripts/product_static_bs1.py --check` | 0 |
| `python3 scripts/pack_trt.py --self-test` | 0 |
| `python3 scripts/pack_occupancy.py --self-test` | 0 |
| `bash scripts/mortredctl_prepare.sh --help` | 0 |
| `bash scripts/mortredctl_prepare.sh --id not_a_catalog_id` | 1 |
| `bash scripts/mortredctl_prepare.sh --id yolov8s --pack conf/packs/demo.toml` | 1 |

自测脚本覆盖第 1 条到第 5 条，以及「已设置 `MORTRED_PACK` 时仍指向该 pack」的核对。脚本恢复它搬移过的 engine，不调用 trtexec。

本机该 id 的产品 onnx 已在表内 `onnx` 路径、且有 trtexec 与 GPU 时，再对冻结名单全部 39 个 id 各执行一行。`yolov8s`、`metric3d_512`、`yolov8n` 三行仍对应第 6 条到第 8 条。三十九行共同对应第 12 条。`yolov8s` 一行不代替 `yolov8n`、`yolov8l`、`yolov8x`。执行前若该 id 的产品 engine 大小为 0，先去掉该文件，或把该行改为带现有 `--force` 的 `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id <该 id> --force`，期望退出码仍为 0。退出码为 0 还要核对表内该 `engine` 存在且大小大于 0；执行前其余 id 的产品 engine 若不存在，执行后仍不存在。缺该 id 的 onnx、缺 trtexec 或缺 GPU 时，该行记挂起，写明缺的环境和未跑的验收，不得改脚本让未生成以退出码 0 表示已生成。

| 命令 | 期望退出码 |
| --- | --- |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id mobilenetv2` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id resnet` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id densenet` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov5l` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov6s` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id nanodet_1x5` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id nanodet_416` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov7` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov7x` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov8n` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov8s` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov8l` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id yolov8x` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id libface` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id centerface` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id bisenetv2` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id pphuman_mobile` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id pphuman_lite` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id pphuman_server` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id hrnet` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id msocrnet` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id dbnet` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id attentive_gan` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id enlightengan` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id realesrgan` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id superpoint` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id dinov2_vits14` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id dinov2_vitb14` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id dinov2_vitl14` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id modnet` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id ppmatting_512` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id ppmatting_1024` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id ppmatting_resnet34` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id ppmatting_v2` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id depth_vits14` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id depth_vitb14` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id depth_vitl14` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id metric3d_512` | 0 |
| `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id metric3d_1088` | 0 |

## RD 不得自行决定

- 39 个 id 与 engine 相对路径。以冻结名单和 `conf/product_static_bs1.json` 的 `engine` 为准，fp32 六条保持 fp32。
- 未设置 `MORTRED_PACK` 时检查这 39 个产品 engine。已设置时仍只检查该 pack。
- `next:` 行的生成命令是 `mortredctl prepare --id <id>`。不得改成对 demo pack 或全量 pack 的 `prepare --pack`。
- `--id` 与 `--pack` 同时出现时的退出码是 1。未知 id 的退出码是 1。doctor 产品 engine 门失败时的退出码是 1。
- 缺失包含不存在和大小为 0。
- `--only` 使用完整 engine 相对路径。
- engine 只在本机本卡生成。本条不下载权重，不复制其它机器的 engine。
- 不把 39 个 id 写入 demo pack，不把未带 `--id` 的 prepare 改成全库构建。
- 不改 HTTP、tensor、占用率，不改 P0-3、P1-7 至 P1-12、D-8 至 D-12，不重开 P1-3、P1-4、D-2。
