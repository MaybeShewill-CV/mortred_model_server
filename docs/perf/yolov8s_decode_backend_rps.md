# YOLOV8s 解码后端吞吐对比（2026-09-22）

分支 `perf/per-worker-gpu-decoder`。每个 worker 独占一份 jpeggpu decoder
（独立 decoder、stream、event 和缓冲），热路径不再共享一把解码锁。
本文记录这次改造之后、在本机上对 YOLOV8s 做的四组闭环压测，以及怎样复现。

四组 RPS 是当次报告的单次数值，不是三轮中位数。GPU 占用和 CPU 占用是
压测过程中对 `nvidia-smi` / `htop` 的目测，不是积分平均值。

## 结果

条件固定：8 worker，`max_batch_size=1`，直接打模型端口，C=16，`--raw`，
输入各一张 JPEG。GPU 列是 `image_decode_backend=gpu`，CPU 列是
`image_decode_backend=cpu`。CPU 列仍走 `image_decode_mode=budget` 的
`IMREAD_REDUCED`，不是全分辨率 libjpeg。

| 输入 | 解码 | RPS | GPU 占用 | CPU 占用（htop） | 闭环平均延迟 |
|---|---|---:|---:|---:|---:|
| bus.jpg，810×1080，476.0 KiB | GPU（jpeggpu 全分辨率） | 296.63 | ~87% | ~40% | 53.9 ms |
| jidu_face.jpg，3840×2160，699.2 KiB | GPU（jpeggpu 全分辨率） | 293.63 | ~80% | ~40% | 54.5 ms |
| bus.jpg | CPU（budget，r=2，405×540） | 264 | ~54% | ~55% | 60.6 ms |
| jidu_face.jpg | CPU（budget，r=4，960×540） | 232 | ~52% | ~63% | 69.0 ms |

闭环平均延迟由 Little 定律从并发和 RPS 推出：`C / RPS`。它是窗口内的
平均驻留时间，不是客户端 P50。前提是窗口内几乎全是 HTTP 200，客户端
是闭环、没有自己空转。

相对关系（同一张图，GPU 相对 CPU）：

- bus.jpg：296.63 / 264 = **+12.4%**
- jidu_face.jpg：293.63 / 232 = **+26.6%**

同一条路径、换图：

- GPU：293.63 / 296.63 = **−1.0%**（两张图几乎一样）
- CPU：232 / 264 = **−12.1%**

htop 的百分比是 16 个逻辑核的平均值。40% 大约是 6.4 个核在忙，63% 大约是
10 个核。`nvidia-smi` 的 GPU-Util 是 SM 忙闲采样，jpeggpu 走的就是 SM，
不是 NVDEC。

## 本机环境

2026-09-22 在 WSL2 里测，GPU 从 Windows 透传。桌面显示接在这张卡上
（`nvidia-smi` 的 `Disp.A = On`），合成器占用也算进 GPU-Util。

| 项 | 值 |
|---|---|
| 宿主机 | Windows 10 22H2（10.0.19045） |
| WSL | Ubuntu 22.04.1，内核 6.18.33.2-microsoft-standard-WSL2 |
| WSL 可见内存 | 11.6 GiB |
| CPU | Intel Core i7-10700 @ 2.90 GHz，8 核 16 线程，L3 16 MiB |
| GPU | NVIDIA GeForce RTX 2070 SUPER，8 GiB，compute capability 7.5，SM 最高 2115 MHz |
| 驱动 | 616.92（nvidia-smi 报 CUDA UMD 13.4） |
| 编译用 CUDA | 12.6（`/usr/local/cuda-12.6`） |
| TensorRT | 10.3.0（`libnvinfer.so.10.3.0`） |
| OpenCV | 4.5.4 |
| libjpeg | libjpeg-turbo 2.1.2（`libjpeg.so.8`） |
| GPU JPEG | jpeggpu，静态库 `3rd_party/libs/libjpeggpu.a`。这张消费卡没有 NVDEC JPEG 引擎 |
| 引擎 | `yolov8s.engine.static_bs1.fp16in`，输入 f16 NCHW `[1,3,640,640]` |
| 服务 | `mortred-model-server.out`，监听 `127.0.0.1:9056`，URI `/mortred_ai_server_v1/obj_detection/yolov8` |
| worker | `[pack.YOLOV8] worker_nums = 8`（`conf/local/local_server.toml`），覆盖 server toml 里的 `worker_nums=1` |
| 二进制时间 | `libmodels.so` 13:37，`mortred-model-server.out` 13:38（该分支工作区编出，改动当时未提交） |

pack 里还有 `cpu_decode_us_per_kb = 9.98`、`decode_gpu_min_cpu_ms = 6.0`。
这两个数只参与 `auto` 的分流，`gpu` / `cpu` 强制模式不用它们决定走哪边。

日志和这次二进制对得上：

- 13:41，pid 25059，`decode fork [YOLOV8]: mode=gpu`，紧挨着 8 条 `jpeggpu decoder ready`。对应 GPU 两列。
- 13:58，pid 47433，`mode=cpu`，没有新的 `jpeggpu decoder ready`。对应 CPU 两列。`cpu` 模式不分配 slot。

中间 13:53 还有一次 `mode=auto` 的启动（也是 8 条 ready）。那次不是这四格。
`auto` 不会让两张图走同一条路，见后面的分流计算。

## 两条路径实际在做什么

网络输入 640×640，`image_decode_budget_upscale = 1.25`。CPU 档位由
`jpeg_reduce_factor_for` 决定：先按 `min(640/W, 640/H)` 选 1/2/4/8，再用
预算上采样允许再降一档。

| 图 | 全图像素 | 字节 | CPU 档 | CPU 实际解码 | GPU 实际解码 |
|---|---:|---:|---:|---|---|
| bus.jpg | 0.875 MP | 487438 | r=2（预算从 1 抬上来的；2×缩放比 = 1.185 ≤ 1.25） | 405×540，像素为全图的 1/4 | 810×1080 全图，设备端 letterbox 到 640 |
| jidu_face.jpg | 8.29 MP | 716019 | r=4（再降到 8 会有 1.333 倍上采样，超过 1.25） | 960×540，像素为全图的 1/16 | 3840×2160 全图，设备端 letterbox 到 640 |

jidu 的像素是 bus 的 9.5 倍，文件只大 1.47 倍。它是一张压得很小的 4K JPEG。

CPU 路径：libjpeg-turbo `IMREAD_REDUCED` → CPU letterbox → pinned H2D →
本 worker 的 TRT stream。GPU 路径：本 worker 的 jpeggpu 全分辨率解码 →
CUDA preprocess → 同一条 stream 上 `cudaEventRecord`，TRT 等到这个 event
再 `enqueueV3`。两条路径的检测器看到的像素不同。历史上 bus.jpg 的 budget
降档相对全分辨率有约 2.95 px 的坐标差（`docs/perf/latency_iterations.md`
W4）；全分辨率 GPU 解码的坐标差大约 0.32 px。这次只比吞吐，没有重跑 golden。

## 复现

1. 用本分支的 gpu profile 编出 `mortred-model-server.out`，经 supervisor 拉起。
   确认进程环境里 `MORTRED_WORKER_NUMS=8`。server toml 写的 `worker_nums=1`
   会被 pack 覆盖。
2. 改 `conf/model/object_detection/yolov8/yolov8_config.toml` 的
   `image_decode_backend`。GPU 两列写 `gpu`，CPU 两列写 `cpu`。
   `image_decode_mode` 保持 `budget`。改完重启模型进程：

   ```bash
   mortredctl restart YOLOV8
   ```

3. 看 `logs/YOLOV8.log` 里这一次启动的 `decode fork` 行。
   - GPU 列：8 行 `mode=gpu`，并且有 8 行 `jpeggpu decoder ready`。
   - CPU 列：8 行 `mode=cpu`，没有新的 ready 行。
   - 不要用 `gpu_path_eligible=yes` 判断。`cpu` 模式下引擎仍然合格，
     这面旗子保持 true，但请求不会进 GPU。
4. 直接打模型端口，不要打 gateway。`--token` 用模型进程的
   `MORTRED_AUTH_TOKEN`。脚本默认读的是 `MORTRED_GATEWAY_AUTH_TOKEN`，
   打 9056 时对不上。

   ```bash
   python3 scripts/server/http_infer_rps.py \
     --url http://127.0.0.1:9056/mortred_ai_server_v1/obj_detection/yolov8 \
     --image demo_data/model_test_input/object_detection/bus.jpg \
     --raw -c 16 --warmup 5s --duration 15s \
     --token "$MORTRED_AUTH_TOKEN"
   ```

   jidu 把 `--image` 换成
   `demo_data/model_test_input/object_detection/jidu_face.jpg`。
   建议同一格跑 3 轮取中位。上面的表是单次读数，复现时允许大约 ±1% 的
   抖动；差出 10% 以上就先核对 mode 和计数器，不要直接和本表比。
5. 压测前后各抓一次 `/metrics`，只看增量。值为 0 的计数器不会打印。

   ```bash
   curl -s -H "Authorization: Bearer $MORTRED_AUTH_TOKEN" \
     http://127.0.0.1:9056/metrics | grep mortred_jpeg_decode
   ```

   - GPU 列：`mortred_jpeg_decode_total{backend="jpeggpu"}` 的增量应约等于
     成功请求数，`cpu-reduced` / `cpu-full` 不动。
   - CPU 列：只增加 `cpu-reduced`（这两张图的档位都大于 1）。
   - `mortred_jpeg_decode_ladder` 的数值是已打开的 slot 数。它证明 decoder
     创建了，不证明请求走了 GPU。
6. 占用率在压测窗口里看：`nvidia-smi dmon -s u`，以及 htop 的平均 CPU。
   采样是目测，和表里的百分号同一精度。

`auto` 复现不出这张表。按 pack 的 9.98 µs/KB、6 ms 门槛：

- bus.jpg：`9.98 × (487438/1024) / 1000 = 4.75 ms`，低于 6 ms，留在 CPU。
- jidu_face.jpg：`9.98 × (716019/1024) / 1000 = 6.98 ms`，进 GPU。

所以 `auto` 会把这两张图拆到两列去，不会两张都出现 ~85% 的 GPU 占用。

## 分析

这四格符合「每个 worker 一块 GPU decoder」之后的预期，不符合改造之前
「GPU 解码比 CPU 慢」的那个观察。改造前所有 worker 共用一个 decoder、
一条 stream、一个 event 和一把锁；解码在 GPU 上串行，还和解码之后的
TRT 抢同一块 SM。那时候强制 GPU 的服务吞吐低于 CPU 降档路径。现在锁没了，
8 条 stream 可以把「下一张图的解码」叠到「上一张图的 TRT」后面。

### GPU 两列几乎一样，而且 GPU 是瓶颈

296.63 和 293.63 只差 1%，落在单次读数的噪声里。GPU 占用 80–87%，CPU
大约 6 个核，16 个逻辑核没有打满。限制在 GPU 上，不在主机解码。

同一条请求里，jpeggpu、preprocess 和 TRT 有先后：TRT 必须等到这张图的
event。能藏起来的是跨请求的重叠。C=16、8 个 worker 时队列是满的，重叠
足够把 4K 多出来的解码填进空隙。jidu 的全图像素是 bus 的 9.5 倍，RPS
却没有掉，说明在这张卡、这个 worker 数下，全分辨率 jpeggpu 还没把重叠
吃完。占用率从 87% 看到 80%，差距在目测采样的误差里，不能据此说 4K
更空闲。

### CPU 两列被主机喂不饱，4K 更明显

CPU 路径上 GPU 只有大约 52–54%。TRT 有一半时间在等主机把 letterbox 之后
的张量送上来。8 个 worker 的 `imdecode` 和 CPU letterbox 是这段等待。

降档省的是 IDCT，省不掉 Huffman。bus 从全图 6.2 ms 降到大约 5.0 ms，
只有 1.27 倍（`latency_iterations.md` W4），因为 Huffman 必须扫完整份
码流。jidu 的文件只比 bus 大 1.47 倍，Huffman 是同一量级；但降档之后
仍要做 960×540 的 IDCT 和 letterbox，像素是 bus 降档结果的 2.4 倍。
所以 CPU 从大约 55% 升到 63%，RPS 从 264 掉到 232（−12%）。这和
2026-09-21 那次 campaign 里「4K raw 比 bus raw 低大约 14%」是同一件事：
主机喂图深度不够，GPU 吃不饱。

GPU 路径把这截喂图从主机挪走了。4K 的收益（+26.6%）大于 bus（+12.4%），
差出来的部分就是这 12% 的主机缺口。

### 和旧的单图结论、和 337 RPS 怎么放在一起

`latency_iterations.md` 迭代 6 的 5.41 ms（nvJPEG 全解）对 4.83 ms
（CPU 降档）是单请求、单 decoder 的延迟。它说明「一张图、解码器还要
加锁」时，全分辨率 GPU 解码比降档 CPU 慢。它不预测 8 条独立 stream 在
C=16 下的吞吐。这次 bus 的 GPU 列比 CPU 列高 12%，和那个单图结论不矛盾：
比的不是同一种并发结构。

`docs/perf/decode_pipeline_redesign.md` 里，同一台 2070 SUPER、同样
C=16 直接打 9056、15 s × 3 轮的中位数是：bus raw **337.4**（约为当时
观测到的纯推理上限 341.8 的 98.7%），jidu 4K raw **288.6**。那次 auto
把两张图都留在 CPU 降档路径。今天这张表的 CPU 列（264 / 232）和 GPU 列
（296.63 / 293.63）都低于 337.4。

不要把这个差距读成「per-worker decoder 把 CPU 路径做退了」。CPU 路径的
解码代码这次没改；今天 CPU 列也没有创建 jpeggpu slot。绝对数对不上的
原因至少包括：本表是单次读数，不是三轮中位；GPU 占用是目测；桌面合成
挂在同一张卡上；两次实验的客户端时长没有对齐。同一次实验里的四格比较
仍然成立，跨 campaign 的绝对 RPS 不能直接当回归。

在这个前提下，有一个和代码结构一致的读法。337 RPS 对应的 GPU 帧大约
是 3 ms 量级的 TRT（降档图的主机预处理已经快到能喂满）。今天 GPU 列
296 RPS、SM 大约 87%，相当于在这条关键路径上又叠了全分辨率 jpeggpu 和
全图 preprocess，帧变长大约 `337/297 − 1 ≈ 14%`。bus 这种已经能喂满
TRT 的图，把解码搬上 GPU 会比「降档 CPU + 喂满的 TRT」更慢。4K 原来
喂不满，搬上去之后 293.63 和旧的 288.6 打平。今天这次实验里 GPU 列
仍然赢过今天的 CPU 列，是因为今天的 CPU 列本身只有 264 / 232，没有
回到 337。

### 值得接着量的

- 每个格子 15 s × 3 轮取中位，同时记下 `jpeggpu` 与 `cpu-reduced` 的增量。
  没有计数器的 RPS 不能区分 `gpu` 和 `auto`。
- `worker_nums=1` 对 `8`，都强制 `gpu`。1 个 worker 时跨请求重叠消失，
  RPS 应该明显低于 8，直到撞上 TRT 本身。这是「锁已经去掉」的直接检验。
- 负载中和空闲时的检测框是否一致。per-worker slot 的正确性靠 lease：
  下一次解码复用缓冲之前，上一次 TRT 已经同步完。吞吐数字看不出张量被串。
- bus 上如果目标是绝对 RPS 而不是全分辨率精度，CPU budget 路径的历史上限
  仍然更高。`auto` 用字节门槛把 bus 留在 CPU、把 4K 送去 GPU，方向和
  这四格一致；6 ms 这个具体门槛只是一次校准，不是这张表测出来的。
