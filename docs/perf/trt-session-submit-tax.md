# TensorRT 提交路径公共税（静态 batch=1 GPU 管线）

状态：**方案，尚未落地。** 数字来自 2026-09-24 Set A 实测（RTX 2070 SUPER 8 GiB，静态 batch=1）。HTTP RPS 是发布值；trtexec Throughput 不是 HTTP RPS。

本文记录一套针对 `TrtSession::run` 的提交路径改造：在 **不改 decode fork、不加 worker、不把后处理搬上 GPU** 的前提下，缩短每票请求占用 GPU 的时间，使其靠近同引擎、同形状的 trtexec 紧循环。方案从 yolov8s 的 72% 达成率分析导出，实现落在 `src/models/backend/trt_session.{h,cpp}`，对满足前提的静态引擎模型共用。

相关文档：

- Set A 表：`docs/benchmark-set-a.md`
- yolov8s 解码后端对比：`docs/perf/yolov8s_decode_backend_rps.md`
- 端到端 stage 埋点：`docs/perf/latency_iterations.md`（`/tmp/mortred_stage_timing`）

## 1. 问题

### 1.1 达成率

\[
\text{达成率} = \frac{\max(\text{HTTP CPU rps},\;\text{HTTP GPU rps})}{\text{trtexec qps}}
\]

batch=1 时，一次 HTTP 成功对应一次 `enqueueV3`。同形状下 trtexec 的 qps 是服务 RPS 的近似上限（多 stream 拷贝重叠可略超 Throughput，过不去 `1 / GPU Compute Time`）。达成率明显低于 90%，说明服务每票占用 GPU 的时间长于 trtexec 一次迭代。

yolov8s（GPU 解码、`w*=8`、并发 32、`bus.jpg`）：

| 量 | 值 |
|---|---|
| trtexec GPU compute | 2.04 ms |
| trtexec latency（含 H2D/D2H） | 2.50 ms |
| trtexec qps | 413.7 |
| HTTP GPU rps | 297.8 |
| 服务每票节拍 `1000/297.8` | **3.36 ms** |
| 达成率 | **72.00%** |

多出来的约 0.86 ms 不是 YOLO 头扫描（那一段在 worker 归还前，高并发时可与其他请求的 `exec` 重叠；标定 `w=8→16` RPS 不再涨，说明已顶在共享 GPU 上）。

### 1.2 公共税在代码里是什么

GPU 零拷贝路径（jpeggpu 写出 `device_data` + `device_ready_event`）进入 `TrtSession::run` 后，当前实现每票仍会：

1. 按**输入字节**向 `_m_pinned_h2d_pool` `acquire` 一块 host 缓冲（yolov8s fp16 约 2.4 MiB），即使后面走 `device_data` 分支、根本不用这块。
2. 每次 `setInputShape`（静态 `[1,3,640,640]` 也设）。
3. 每次 `cudaStreamWaitEvent` + `setTensorAddress`（每个 worker 的 `d_output` 指针实际不变）。
4. `enqueueV3`。
5. **`cudaStreamSynchronize` 一次**。
6. 从 `_m_pinned_d2h_pool` 再借一块，`cudaMemcpyAsync` D2H（yolov8s `[1,84,8400]` fp32 约 2.8 MiB），再 **`synchronize` 一次**，再 `memcpy` 到 `std::vector`。

jpeggpu 在自己的 stream 上 `eventRecord`，不等待。把解码和推理串起来的是推理 stream 上的 `WaitEvent` 和第一次全流同步。trtexec 是单流 `--useSpinWait` 紧循环，没有 JPEG、没有双 sync、没有每票绑地址。服务里不要用 spin wait（占满 CPU）。

同一套税出现在所有走 `TrtSession::run` 的模型上。`exec` 越短，达成率越差（yolov8n 50%、superpoint 44%、libface_640 21%），因为分母变小、分子（HTTP 节拍）被同一笔提交税托在 ~3 ms。

### 1.3 明确不做什么

| 不做 | 原因 |
|---|---|
| 改回 CPU 解码来「避免抢 SM」 | yolov8s 已测：GPU 297.8 > CPU 276.4。换 CPU 会把 72% 降到约 67%。 |
| 加 worker | 标定到平台期后 RPS 不再涨。 |
| 把 YOLO/YuNet 后处理搬上 GPU | 与短 `exec` 抢 SM；高并发下 CPU 后处理已能重叠。 |
| 在 `.static_bs1` 引擎上开 `max_batch_size>1` | 打包会被静态 N=1 拒绝，退回更慢的逐张 `run()`。 |
| 把 JPEG 解码打进 CUDA graph | 码流长度每张图不同。 |

## 2. 技术方案

改造集中在 `TrtSession`。模型侧继续 `set_gpu_preprocess` + 现有 `run_impl` 零拷贝入口。四刀有依赖：3 稳定后才能 4。

### 刀 1 — GPU 零拷贝不要付 H2D 主机池

**现状：** `run()` 在绑定循环之前对所有输入累加 `pinned_h2d_total`（含 `device_data`），再 `pinned_h2d_stage`。零拷贝分支 `continue` 掉，缓冲仍占用到 stream sync 之后才 `release`，并抢池上的 mutex。

**改法：** 所有输入都是 `device_data` 时 `pinned_h2d_total = 0`，不碰 `_m_pinned_h2d_pool`。CPU 路径（主机 tensor → H2D）行为不变。

**收益：** 去掉每票一次无用的 ~2.4 MiB pinned 借还和 mutex。不改数值。

### 刀 2 — 同一条推理 stream 上合并同步

**现状：** `enqueueV3 → cudaStreamSynchronize → D2H → cudaStreamSynchronize`。第一次 sync 只为「算完再拷回」。

**改法：** 输出形状在 enqueue **之前** 已知时（无 `IOutputAllocator`）：

```text
cudaStreamWaitEvent(jpeg_ready)   // 仅 GPU 零拷贝
enqueueV3
cudaMemcpyAsync D2H               // 同一 _m_stream，顺序保证发生在 compute 之后
cudaStreamSynchronize             // 仅此一次，之后主机可读
```

动态输出仍走 allocator 的模型（YuNet 等）**必须**先 sync 再读形状，不能套这一刀，见 §4.3。

**收益：** 去掉一次全设备阻塞。8 个 worker、多 context 时，这次 sync 是最大的提交空窗。预期对短 `exec` 模型（n / superpoint / pphuman-192）相对收益最大。

**不要：** `cudaDeviceScheduleSpin` / trtexec `--useSpinWait`。

### 刀 3 — 静态 I/O 只绑一次

前提：该 session 生命周期内输入形状不变，且每个 worker 的 device 指针不变（jpeggpu slot 的 `d_output` 复用）。

**改法：**

- 形状与上次相同 → 跳过 `setInputShape`。
- 输入/输出 device 指针未变 → 跳过 `setTensorAddress`。
- 静态输出在该 worker 第一次成功 `run` 时 `ensure` 并绑死。
- 输出 host 侧用 **session 自持 pinned**，不要每票从 `_m_pinned_d2h_pool` 借还。D2H 直接进这块；后处理读 pinned，去掉 sync 之后那次整段 `memcpy`。

多静态输出（如 YOLOv7 三个 head）同样适用：第一次全部绑死，graph 里多次 D2H。

**收益：** 砍每票 TensorRT 绑定和 host 二次拷贝。为刀 4 提供不变的地址。

### 刀 4 — 只捕获推理 stream 上的短图

在刀 3 之后、单个 worker 上预热若干次普通 `run`，再对 `_m_stream` `cudaStreamBeginCapture`：

```text
cudaStreamWaitEvent(jpeg_ready)
enqueueV3
cudaMemcpyAsync D2H（一路或多路静态输出）
```

JPEG 解码、letterbox kernel 仍在 **jpeggpu stream** 上跑完再 `eventRecord`。图只负责「等这个 event + 推理 + 拷回」。每个 worker 一张图（现有模型已是一 worker 一 context 一流）。输入指针若永远是该 slot 的 `d_output`，replay 不必改 graph。地址或形状一变必须销毁重捕。

**收益：** 砍 `enqueueV3` 的 CPU launch。对 0.2–2 ms 级 `exec` 相对更值钱。注意：Set A 的 trtexec 也是 `cuda_graph=no`；服务先上图会抬达成率的**分子**，分母那次 trtexec 不会一起变。若对照也打开 graph，两边都会涨，比值另算。

### 落地顺序

1. 刀 1 + 刀 2（改动面小，对着无效 pinned 和双 sync）。
2. 打开 `/tmp/mortred_stage_timing`，在目标模型 GPU 管线、原 `w*` / 原并发下看 `h2d` / `exec` / `d2h` 空窗。
3. 刀 3（静态引擎）。
4. 刀 4（绑定稳定之后）。

精度：不改预处理公式、不改引擎、不改后处理数学。应用 golden / 检测框对比作门禁（与 `docs/perf/latency_iterations.md` 相同）。

## 3. 预期技术收益

收益对象是 **高并发 HTTP RPS**（发布值），不是并发=1 的端到端延迟总和。并发=1 时四刀也会缩短 `T_serial`；高并发时只有缩短了 `T_GPU,独占` 才会抬吞吐。

粗模型（GPU 已打满、CPU 后处理已重叠）：

\[
\text{HTTP 节拍} \approx T_{\text{exec}} + T_{\text{jpeg/letterbox 抢 SM}} + T_{\text{提交税（双 sync / 绑定 / launch）}}
\]

\[
\text{达成率} \approx \frac{T_{\text{trtexec latency}}}{\text{HTTP 节拍}}
\]

四刀主要减 \(T_{\text{提交税}}\)。JPEG/letterbox 与 TRT 抢 SM **不在本方案范围**（已确认 CPU 解码更慢，不退回）。

用 Set A 已完成行作**方向性**估计（不是承诺值）。「服务节拍」= `1000 / 较高一侧 HTTP rps`。若提交税从当前空窗里收回 **0.3–0.8 ms**（双 sync + 绑定，不含 graph），短 `exec` 模型的达成率升幅大于长 `exec`。

| 模型 | 当前达成率 | trtexec latency (ms) | 当前 HTTP 节拍 (ms) | 为何预期有效 |
|---|---:|---:|---:|---|
| yolov8n | 50.16% | 1.75 | 3.21 | `exec` 1.27 ms，税占比最大 |
| superpoint | 43.70% | 0.25 | 0.65 | `exec` 0.20 ms，与 libface 同类但 I/O 静态 |
| pphuman_lite | 35.66% | 0.83 | 2.47 | 192 输入，trtexec ~1137 qps |
| pphuman_mobile | 44.04% | 0.96 | 2.30 | 同上 |
| dinov2_vitb14 | 42.15% | 2.42 | 5.67 | 静态 NCHW，HTTP 远低于 trtexec |
| dinov2_vits14 | 55.18% | 3.42 | 6.15 | 同上 |
| yolov8s | 72.00% | 2.50 | 3.36 | 原分析对象，收回 ~0.8 ms 则节拍接近 2.5–2.7 ms |
| yolov7 | 81.41% | 5.43 | 6.29 | 三静态 head，税相对较小 |
| dinov2_vitl14 | 83.42% | 6.74 | 8.04 | `exec` 长，升幅小于 s/b |
| dbnet | 83.11% | 3.14 | 3.67 | 大图 D2H/后处理仍在 |
| bisenetv2 | 58.66% | 12.75 | 18.81 | 分割图 D2H 为主因，本方案只削提交税 |
| yolov8l | 67.20% | 6.54 | 9.59 | 见 §4.2：表上走 CPU HTTP |

Graph（刀 4）在刀 2/3 之后额外砍 launch，短 `exec` 上可能再收回零点几毫秒。应用后应用 **同一套** Set A HTTP 协议复测（`w*`、并发、`MORTRED_IMAGE_DECODE_BACKEND=gpu`、同一张 JPEG），并与当时的 `logs/bench/set_a/trtexec/<id>.json` 比达成率。不要用 spin-wait trtexec 当服务必须达到的硬指标。

## 4. 可应用的模型范围

范围限于 Set A 中 **达成率 < 90%** 且已跑完转换 + trtexec + 标定 + HTTP 的行。产品 toml 不改；对照仍用 `conf/bench/set_a/<id>.toml`。

四刀前提：

| 前提 | 刀 1 | 刀 2 | 刀 3 | 刀 4 |
|---|---|---|---|---|
| GPU 零拷贝（`device_data`） | 需要 | 可选（CPU 路径无 WaitEvent） | 指针稳定时需要 | graph 含 WaitEvent 时需要 |
| 输出在 enqueue 前可知（无 allocator） | — | **硬条件** | 静态输出 | 同左 |
| 进程内输入形状不变 | — | — | **硬条件** | **硬条件** |

### 4.1 整套四刀（实现落在 `TrtSession`，模型不用各写一份）

| id | catalog | 达成率 | 引擎 | 说明 |
|---|---|---:|---|---|
| yolov8n | YOLOV8 | 50.16% | 静态 640，单输出 `[1,84,8400]` | **优先。** 与 s 同一 `YoloV8Detector` |
| yolov8s | YOLOV8 | 72.00% | 同上 | 原方案 |
| yolov7 | YOLOV7 | 81.41% | 静态 640，**三个**静态 head | 刀 2/3/4 按多路静态 D2H 推广，不是逐张 `run()` |
| superpoint | SUPERPOINT | 43.70% | 静态 `[1,1,120,160]` | GRAY 预处理；`exec` 极短 |
| pphuman_mobile | PPHUMAN_SEG | 44.04% | 静态 192×192 | 单输出 |
| pphuman_lite | PPHUMAN_SEG | 35.66% | 静态 192×192 | 单输出 |
| dinov2_vits14 | DINOV2 | 55.18% | 静态 NCHW | `set_gpu_preprocess` DIRECT_RESIZE |
| dinov2_vitb14 | DINOV2 | 42.15% | 静态 NCHW | 同上 |
| dinov2_vitl14 | DINOV2 | 83.42% | 静态 NCHW | 能套；升幅小于 s/b |
| dbnet | DBNET | 83.11% | 静态 544×960，单输出 | 能套；后处理/大图仍占用节拍 |
| bisenetv2 | BISENETV2 | 58.66% | 静态 **NHWC**，单输出 | graph 同样成立；分割图 D2H 很大 |

以上均已 `set_gpu_preprocess`，HTTP 较高一侧为 GPU 解码（yolov8l 除外，见下）。

### 4.2 代码能套、表上的达成率未必动

**yolov8l（67.20%）** 与 n/s 同一检测器和静态 I/O，四刀都能编进 `TrtSession`。表上比值用的是 **CPU HTTP 104.3 rps**（GPU 仅 77.7）。刀 1 只作用于零拷贝；刀 2/3/4 对 CPU 路径的 H2D+enqueue+D2H 有效。在 GPU 管线成为更快一侧之前，四刀抬的不是表上那一格。不要把 yolov8l 当作本方案的首要验收模型。

### 4.3 不能整套套用

| id | 达成率 | 原因 | 本方案能做的 | 应另开的工作 |
|---|---:|---|---|---|
| libface_640 | 21.44% | 12 路动态 YuNet 头 + `IOutputAllocator`；必须先 sync 再 D2H；无 pinned D2H | 仅刀 1（GPU 零拷贝仍可能误借 H2D pinned） | 按 overlay 固定 480×640 在 enqueue 前绑死 12 路、一次 sync、热路径去掉全量 `require_finite`，再考虑 graph |
| centerface | 75.19% | `ALIGN_TO_MULTIPLE /32`，H/W 随原图变，输出随输入变 | 刀 1 | 按对齐后的 H/W 推断输出再绑，或按分辨率分 graph；换图必须重捕 |

libface_640 的 toml 虽把内容尺寸钉死，**当前 `TrtSession` 仍当动态输出处理**，所以不能直接上刀 2–4。那是 YuNet I/O 契约改造，不是本文四刀的子集。

### 4.4 不在「不足 90%」清单里的模型

达成率 ≥ 90% 的行（yolov8x、pphuman_server、attentive_gan、depth_* 等）`exec` 已接近或超过 HTTP 节拍，提交税不是主因。代码路径若满足静态 I/O，四刀仍可跟随 `TrtSession` 一并生效，**不要作为本方案的验收目标**。未完成 Set A 全流程的模型（nanodet、分类 NHWC 失败行、matting 转换失败等）先完成基准再谈达成率。

## 5. 验收

1. 数值门禁：目标模型 demo 图 golden（检测框 / 分割 / embedding）与改造前逐字段或约定容差一致。
2. 性能：同一 overlay、`MORTRED_WORKER_NUMS=<原 w*>`、`MORTRED_IMAGE_DECODE_BACKEND=gpu`、原并发、同一 JPEG，15 s HTTP；达成率相对改造前 **上升**，且 GPU HTTP 不低于改造前。
3. 回归：CPU 解码路径 RPS 不无故下降（刀 1 不得破坏 H2D；刀 2 不得在 allocator 模型上提前 D2H）。
4. 埋点：`/tmp/mortred_stage_timing` 下 `exec` 与 `d2h` 之间不应再出现一次独立的全流 sync 间隔（allocator 模型除外）。

首轮建议验收：**yolov8n + yolov8s + superpoint**（短 `exec`、静态单输出或等价、GPU 解码已是更快一侧）。
