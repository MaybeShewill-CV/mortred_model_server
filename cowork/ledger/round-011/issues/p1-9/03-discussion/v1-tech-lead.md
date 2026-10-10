# 技术方案

- 问题: P1-9
- 版本: v1
- 对应列表: v3

## 边界

上一版审查意见无。本方案只覆盖冻结句 H3：`docs/benchmark-set-a.md` 里，范围内 39 个 catalog id 各自有一行，同一行里同时有硬件、可直接执行的复现命令、指标和日期。

这 39 个 id 按冻结句的顺序是：`mobilenetv2`、`resnet`、`densenet`、`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`、`libface`、`centerface`、`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dbnet`、`attentive_gan`、`enlightengan`、`realesrgan`、`superpoint`、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。

现在硬件和日期在「机器」表，指标在表 2 和表 4，复现位置是 `logs/bench/set_a/trtexec/<id>.cmd.txt` 这种路径。`logs/` 被忽略，仓库里没有这些 cmd 文件。表 1、表 3、表 4 把 `libface` 拆成 `libface_320` 和 `libface_640`。`msocrnet` 在表 3、表 4 为空。这些旧表保持原样。

指标只来自本方案在这台机器上刚跑完的那次 `trtexec` 标准输出里最后一条 `Throughput:`。表 2、表 3、表 4 的数字，`logs/bench` 里已有的 json，以及 P1-8 尚未生成的黄金，都不能写成这一行的已测指标。命令非 0、没有 `Throughput:`、engine 文件不在、或硬件探测失败时，不写该 id 的指标，也不写表 5。39 个都成功才发表表 5。缺 engine 或没有 GPU 时，实现说明点名没跑成的 id，交给调度者按环境挂起；挂起不算这一条通过。

不覆盖 P0-2、P1-8、P1-10、P1-11、P1-12、D-8 至 D-12。不重开已关闭原句。

## 做法

### 推荐

新增一次只跑本机、一次一个 id 的采集，把结果写进 `docs/benchmark-set-a.md` 文末的新节 `## 表 5 — 一行基准`。每一行是一个 markdown 表格数据行，单元格依次是 catalog id、硬件、命令、指标、日期。

采集开始时执行一次 `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`，取第一行作为全部 39 行共用的硬件字符串。然后按冻结句顺序，对每个 id 在仓库根目录执行下面这一条命令，工作目录是仓库根，进程结束再跑下一个：

`3rd_party/bin/trtexec --loadEngine=<下表路径> --warmUp=500 --duration=15 --avgRuns=100 --useSpinWait`

路径以 `.fp16` 结尾时，命令末尾再加 `--fp16`。下表「shapes」非空时，再加 `--shapes=` 接该单元格。除此之外不加别的参数。

指标单元格写成 `trtexec_rps=` 加上该进程输出里最后一条 `Throughput:` 与 `qps` 之间的数字原样，不再格式化。日期是该进程退出时的本地日期，格式 `YYYY-MM-DD`。同一轮 39 个都成功后，才一次性写入表 5，并为每个 id 写入 `docs/benchmark-h3/<id>.txt`。记录文件含四行：硬件、日期、与表格相同的命令、以及进程输出里那条 `Throughput:` 原行。

`scripts/bench_set_a/run.py` 重写英文基准文档时，若旧文件已有 `## 表 5 — 一行基准` 直到文末，把这段原样接回，不根据 `logs/bench` 重算。中文副本不接表 5。

代价：每个 id 约 15 秒时长加 500 毫秒预热，39 个串行，大约十几分钟；`msocrnet` 和 `metric3d_1088` 仍被 `--duration=15` 封顶。需要本机已有 `3rd_party/bin/trtexec`、一次成功的 `nvidia-smi` 查询，以及下表里的 engine 文件。缺任一条件则整张表 5 不发表。

### 否决

把「机器」表的 GPU 和 `2026-09-24`、表 2 的 `trtexec_rps`、以及 cmd 文件路径拼成一行。代价看起来只是改文档。那条路径不是可执行命令，cmd 文件不在仓库里，表 2 的数字也不是这次进程打出来的。冻结句要求的可执行命令和已测指标都不会出现。`write_docs` 若用已有 `logs/bench` json 填表 5，并盖上今天的日期，同样是把旧数字写成这次的已测指标，所以表 5 不走这条生成路径。

改用 `mortred-model-benchmark.out` 的 `mean` 毫秒。代价是要先有完整构建，每个 id 固定 100 次循环，`--loops` 在没有 `--batch` 时不起作用；`msocrnet` 在 P1-8 里已经因内存不足被挂起，这条命令仍会加载产品 toml 里的 TensorRT engine。它还要改计时循环才能变成短命令，那超出这一行文档的可观察变化。

## 改动面

- `scripts/bench_set_a/h3_lines.py`：按本方案探测硬件、串行执行 39 条命令、只从本次输出取 `Throughput:`，成功后写表 5 和记录文件。
- `scripts/bench_set_a/check_h3_lines.py`：只读表 5 和记录文件，不启动 `trtexec`。
- `scripts/bench_set_a/run.py`：重写英文文档时原样保留已有表 5；中文副本仍由原正文生成，不含表 5。
- `docs/benchmark-set-a.md`：只在文末增加表 5。
- `docs/benchmark-h3/<id>.txt`：39 个记录文件，每个 id 一个。

## 明确不改

- 「机器」节、表 1、表 2、表 3、表 4、Gaps，以及其中已有的数字、空单元格和 `2026-09-24`。
- `docs/benchmark-set-a.zh-cn.md`、两份 README、`docs/deployment.md`、`docs/openapi.json`。
- 产品 toml、`conf/bench`、`conf/trt_profiles`、engine 文件的生成方式、`mortredctl prepare`、`mortredctl doctor`。
- 推理、HTTP 契约、gateway、supervisor、黄金用例、`model_golden_test`、数值阈值。
- `src/apps/common/benchmark_runner.h` 和 `mortred-model-benchmark.out` 的循环次数。
- `.github/`、CPU profile 的构建依赖。CPU profile 继续不链 TensorRT。本方案只调用已经放在 `3rd_party/bin/trtexec` 的二进制。

## 契约

HTTP 契约和 tensor 契约不变。表 5 是文档行，列名固定为 `id`、`hardware`、`command`、`metric`、`date`。命令单元格不含 `|`。`libface` 只占一行，行 id 是 `libface`。`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x` 各占一行，`--loadEngine` 各用各的文件。指标名是 `trtexec_rps`，与文档已有说明一致，它是 `trtexec` 的 Throughput，不是 HTTP RPS。

`--loadEngine` 使用下列仓库相对路径。shapes 为空的 id 不加 `--shapes`。

| id | engine | shapes |
| --- | --- | --- |
| `mobilenetv2` | `weights/classification/mobilenetv2/mobilenetv2.engine.static_bs1.fp32` | |
| `resnet` | `weights/classification/resnet/resnet.engine.static_bs1.fp32` | |
| `densenet` | `weights/classification/densenet/densenet.engine.static_bs1.fp32` | |
| `yolov5l` | `weights/object_detection/yolov5/yolov5l.engine.static_bs1.fp16` | |
| `yolov6s` | `weights/object_detection/yolov6/yolov6s.engine.static_bs1.fp16` | |
| `nanodet_1x5` | `weights/object_detection/nanodet/nanodet_plus_m_1x5.engine.static_bs1.fp16` | |
| `nanodet_416` | `weights/object_detection/nanodet/nanodet_plus_m_416.engine.static_bs1.fp16` | |
| `yolov7` | `weights/object_detection/yolov7/yolov7.engine.static_bs1.fp16` | |
| `yolov7x` | `weights/object_detection/yolov7/yolov7x.engine.static_bs1.fp16` | |
| `yolov8n` | `weights/object_detection/yolov8/yolov8n.engine.static_bs1.fp16` | |
| `yolov8s` | `weights/object_detection/yolov8/yolov8s.engine.static_bs1.fp16` | |
| `yolov8l` | `weights/object_detection/yolov8/yolov8l.engine.static_bs1.fp16` | |
| `yolov8x` | `weights/object_detection/yolov8/yolov8x.engine.static_bs1.fp16` | |
| `libface` | `weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16` | `input:1x3x480x640` |
| `centerface` | `weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16` | `input:1x3x640x640` |
| `bisenetv2` | `weights/scene_segmentation/bisenetv2/bisenetv2_cityscapes.engine.static_bs1.fp32` | |
| `pphuman_mobile` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_mobile.engine.static_bs1.fp16` | |
| `pphuman_lite` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_192x192_lite.engine.static_bs1.fp16` | |
| `pphuman_server` | `weights/scene_segmentation/pphuman_seg/pp_humanseg_512x512_server.engine.static_bs1.fp16` | |
| `hrnet` | `weights/scene_segmentation/hrnet/hrnetw48_ccd.engine.static_bs1.fp16` | |
| `msocrnet` | `weights/scene_segmentation/msocrnet/msocrnet_fp16.engine.static_bs1.fp16` | |
| `dbnet` | `weights/ocr/db_text_detector/db_model_large.engine.static_bs1.fp16` | |
| `attentive_gan` | `weights/enhancement/attentive_gan_derain/attentive_gan_derain.engine.static_bs1.fp32` | |
| `enlightengan` | `weights/enhancement/enlighten_gan/enlightengan.engine.static_bs1.fp16` | `input_src:1x3x256x256,input_gray:1x1x256x256` |
| `realesrgan` | `weights/enhancement/real_esrgan/realesr-general-x4v3.engine.static_bs1.fp32` | `input:1x256x256x3` |
| `superpoint` | `weights/feature_point/superpoint/superpoint_120x160.engine.static_bs1.fp16` | |
| `dinov2_vits14` | `weights/classification/dinov2/dinov2_vits14_pretrain.engine.static_bs1.fp16` | |
| `dinov2_vitb14` | `weights/classification/dinov2/dinov2_vitb14_pretrain.engine.static_bs1.fp16` | |
| `dinov2_vitl14` | `weights/classification/dinov2/dinov2_vitl14_pretrain.engine.static_bs1.fp16` | |
| `modnet` | `weights/matting/modnet/modnet_hrnet_w18.engine.static_bs1.fp16` | |
| `ppmatting_512` | `weights/matting/ppmatting/ppmatting_hrnet_w18_human_512.engine.static_bs1.fp16` | |
| `ppmatting_1024` | `weights/matting/ppmatting/ppmatting_hrnet_w18_human_1024.engine.static_bs1.fp16` | |
| `ppmatting_resnet34` | `weights/matting/ppmatting/pp_humanmatting-resnet34_vd.engine.static_bs1.fp16` | |
| `ppmatting_v2` | `weights/matting/ppmatting/ppmattingv2_stdc1_human_512.engine.static_bs1.fp16` | |
| `depth_vits14` | `weights/mono_depth_estimation/depth_anything/depth_anything_vits14.engine.static_bs1.fp16` | |
| `depth_vitb14` | `weights/mono_depth_estimation/depth_anything/depth_anything_vitb14.engine.static_bs1.fp16` | |
| `depth_vitl14` | `weights/mono_depth_estimation/depth_anything/depth_anything_vitl14.engine.static_bs1.fp16` | |
| `metric3d_512` | `weights/mono_depth_estimation/metric3d/metric3d_750k_512x1088.engine.static_bs1.fp16` | |
| `metric3d_1088` | `weights/mono_depth_estimation/metric3d/metric3d_750k_1088x1920.engine.static_bs1.fp16` | |

四个动态 id 的 shapes 沿用现有 Set A profile 的 opt。`libface` 的 engine 与 `conf/model/object_detection/libfacedetection/libface.toml` 的 `model_file_path` 同一文件。

## 性能与资源

不拉起 gateway（:8080）和 supervisor（:8787），不常驻进程。同一时间只有一个 `trtexec`，结束即退出，显存只留给当前这一份 static batch=1 engine。不新建 engine，不改 workspace，不加 `--memPoolSize`。39 次串行的墙钟时间由 `--duration=15` 和 `--warmUp=500` 决定，再加进程启动。采集脚本不调用 `killall`。

## 验收标准

1. `docs/benchmark-set-a.md` 有一节 `## 表 5 — 一行基准`。该节有且仅有 39 个数据行，id 与顺序与边界中的名单一致。没有 `libface_320` 和 `libface_640`。
2. 每个数据行是一行，同时含：非空硬件，且硬件不是 `name, memory.total` 这个表头模板，也不含 `N/A`；命令以 `3rd_party/bin/trtexec` 开头，并含该 id 的 `--loadEngine=` 路径、`--warmUp=500`、`--duration=15`、`--avgRuns=100`、`--useSpinWait`；指标匹配 `trtexec_rps=` 后接一个数字；日期匹配 `YYYY-MM-DD`。
3. 39 行的硬件字符串彼此相同。命令彼此不同。命令单元格不是 `logs/bench` 路径，也不以 `.cmd.txt` 结尾。
4. engine 路径以 `.fp16` 结尾的行，命令含 `--fp16`。以 `.fp32` 结尾的六行（`mobilenetv2`、`resnet`、`densenet`、`bisenetv2`、`attentive_gan`、`realesrgan`）不含 `--fp16`。
5. `libface`、`centerface`、`enlightengan`、`realesrgan` 的命令含上表对应的 `--shapes=`。其余 35 个 id 的命令不含 `--shapes`。
6. `yolov8n`、`yolov8s`、`yolov8l`、`yolov8x` 四行的 `--loadEngine` 分别是上表四个文件。`msocrnet` 有自己的一行，指标是 `trtexec_rps`。
7. 每个 id 有 `docs/benchmark-h3/<id>.txt`。其中硬件、日期、命令与该行相同，并且有一条 `Throughput:` 原行，其数字与该行 `trtexec_rps=` 后面的数字逐字相同。
8. 「机器」节、表 1 至表 4、Gaps 的原文还在。表 5 的日期不要求等于 `2026-09-24`。
9. 下面的检查命令退出码为 0。

## 测试义务

QA 对照边界名单读表 5 和 39 个记录文件，逐条核对验收标准 1 到 8。重点看四条 YOLOv8 的 engine 路径各不相同，`libface` 只有一行，`msocrnet` 的指标来自记录文件里的 `Throughput:` 行。

白盒确认 `h3_lines.py` 的指标只取自它刚刚启动的那个进程的标准输出里最后一条 `Throughput:`；退出码非 0、没有该行、硬件探测失败或 engine 不在时，不写表 5。确认它不读取表 2 的单元格，也不读取 `logs/bench` 的 json 来填指标。确认 `run.py` 重写英文文档时原样接回已有表 5，中文副本没有表 5。

交付树上的检查命令退出码为 0。不要求 QA 再跑 39 次 `trtexec`，不要求黄金对比、`doctor --strict` 或 README 对照。

## 自测命令

在仓库根目录执行。本机入口是 `python`。

`python scripts/bench_set_a/h3_lines.py`

39 个 id 都在本次进程输出里打出 `Throughput:` 时，期望退出码 0，并写出表 5 与记录文件。任一 id 没有跑成时，期望退出码 1，且 `docs/benchmark-set-a.md` 不出现新的表 5。

`python scripts/bench_set_a/check_h3_lines.py`

表 5 和记录文件满足验收标准 1 到 8 时，期望退出码 0。缺行、命令不是上述 argv、或记录文件里的数字与 `trtexec_rps` 不一致时，期望退出码 1。这条检查不启动 `trtexec`。

## RD 不得自行决定

- 不得把表 2、表 3、表 4 或 `2026-09-24` 那次文档里的数字抄进表 5，不得把日期写成采集进程没有退出的那一天。
- 不得在命令失败、没有 `Throughput:`、`nvidia-smi` 失败、输出含 `N/A`、或 engine 文件不在时填写指标或发表残缺表 5。
- 不得用 `libface_320`、`libface_640` 代替 `libface`，不得用 `yolov8s` 的行代替 `yolov8n`、`yolov8l`、`yolov8x`。
- 不得增删 `trtexec` 参数，不得改 `--duration`、`--warmUp`、`--avgRuns`，不得并行跑多个 id，不得改 engine 路径和四条 shapes。
- 不得改产品 toml、engine 生成、`mortredctl doctor`、黄金用例、README、部署文档、鉴权、镜像用户、GPU 采样或 fork 与 exec 之间的检查来凑这一行。
- 不得把 `trtexec_rps` 写成 HTTP RPS，不得用 P1-8 的黄金或未生成文件编指标。
