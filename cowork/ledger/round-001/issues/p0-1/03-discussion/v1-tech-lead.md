# 技术方案

- 问题: P0-1
- 版本: v1
- 对应列表: v3

## 边界

本方案只让已拉起的 `libface` 对一张合法 JPEG 返回成功状态，并让下面三处的高和宽相同：

- `conf/model/object_detection/libfacedetection/libface.toml` 的 `model_input_image_size`。`src/models/object_detection/detection_params.h` 把该数组定为 `[height, width]`。
- `conf/trt_profiles/set_a_libface.json` 里名为 `input` 的 NCHW profile。
- `docs/benchmark-h3/libface.txt` 的 `--shapes=input:1x3x480x640`。

对齐后的形状是高 480、宽 640、batch=1。该文件已经用这个形状对 `weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16` 跑出 Throughput 1668.97 qps。当前 toml `[640, 480]` 与 profile `[1, 3, 640, 480]` 会送出 `1×3×640×480`，静态引擎拒绝后返回 `MODEL_RUN_SESSION_FAILED`。

TensorRT 保持 static shape、batch=1。每个 worker 仍各自持有一个 execution context。不要求动态 shape。不改 `realesrgan`。

## 做法

### 推荐

把产品 toml 和 profile 改到基准文件已经测通的静态形状。检测器继续按 toml 的 `[height, width]` 做 resize，再打成 NCHW。480 和 640 都能被 32 整除，现有的右/下补齐不会再放大这张量。GPU 预处理和 CPU 预处理读的是同一个 toml 尺寸，一处修改同时覆盖两条被拒绝的路径。

代价：改两份配置，并改 `scripts/product_static_bs1.py` 里锁死 `model_input_image_size = [640, 480]` 的那一句守卫，否则 `python3 scripts/product_static_bs1.py --check` 会失败。现有 engine 若已接受 `input:1x3x480x640`，不重编，基准文件的硬件、日期、命令和 1668.97 qps 保持不动。只有这次 `trtexec --loadEngine` 对该形状退出非 0 时，才在 profile 已经改成三档相同的 `[1, 3, 480, 640]` 之后执行一次 `mortredctl prepare --id libface --force`，再用同形状的 `trtexec` 重写 `docs/benchmark-h3/libface.txt`。第二次探测仍失败就停，不再换尺寸。

### 否决

把三处改成高 640、宽 480，并按现有 profile 重编 engine。代价是必须重编，1668.97 qps 作废。`scripts/convert_trt_engines.sh` 对静态输入的 ONNX 会丢掉 profile；若这份 YuNet ONNX 的静态输入已是 `1×3×480×640`，重编结果仍是高 480、宽 640，toml 继续送出 `1×3×640×480`，请求仍然失败。`scripts/bench_set_a/h3_lines.py`、`scripts/bench_set_a/check_h3_lines.py` 和 `docs/benchmark-set-a.md` 也已经写着 `input:1x3x480x640`。

在 `libface` 预处理里对调高宽，toml 仍写 `[640, 480]`。三份文件的高宽仍然不同，锁定的变化达不到。对调也和所有检测器共用的 `[height, width]` 约定冲突。把 profile 改成动态范围则超出 static shape、batch=1。

## 改动面

- `conf/model/object_detection/libfacedetection/libface.toml`：`model_input_image_size` 改为 `[480, 640]`。`type` 仍是 `tensorrt`，engine 路径、输入名 `input`、12 个输出名、`input_layout = "nchw"` 不动。
- `conf/trt_profiles/set_a_libface.json`：唯一绑定 `input` 的 `min`、`opt`、`max` 都改为 `[1, 3, 480, 640]`。
- `scripts/product_static_bs1.py`：libface 尺寸守卫改为要求 toml 含 `model_input_image_size = [480, 640]`。
- `docs/benchmark-h3/libface.txt`：现有 engine 对 `input:1x3x480x640` 探测退出 0 时不改。只有按上面的条件重编之后，才用这次 `trtexec` 的硬件、日期、命令和 Throughput 覆盖该文件，形状串保持 `input:1x3x480x640`。

## 明确不改

- `src/models/object_detection/libface_detector.inl`、`src/models/object_detection/detection_params.h`、`src/models/backend/trt_session.cpp`。
- `src/server/generic_cv_server.h` 里按 worker 各自 `init` 的循环。每个 worker 继续创建自己的 `TrtSession`，并在该会话里 `createExecutionContext`。
- `conf/server/object_detection/libface_det/libface_server_config.toml` 的 `worker_nums = 1`、`max_batch_size = 1`、端口 9051 和 URI。
- `conf/ci/libface_onnx_hosted.toml`、`conf/onnx_sources.json`、其他 id 的 profile 和 `docs/benchmark-h3/`。
- `realesrgan` 的配置、profile、engine 和代码。
- HTTP 路径、字段名和响应信封。本条不包含 calibrate，也不包含 `http_infer_rps.py` 的发布 RPS 报告。

## 契约

- `model_input_image_size = [480, 640]` 表示高 480、宽 640。
- profile 的 `min`、`opt`、`max` 同为 NCHW `[1, 3, 480, 640]`。
- 送入会话的输入名仍是 `input`，形状是 `1×3×480×640`，batch=1。12 个输出名不变。
- 一张合法 JPEG 仍走现有请求：模型端口接受 `Content-Type: image/jpeg` 的原始字节，也接受现有 JSON `images`。成功响应仍是 HTTP 200，JSON `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0。
- 框和关键点仍按现有解码映射回原图像素。本条不规定框的数值。

## 性能与资源

- 继续使用本机本卡上的 `face_detection_yunet_2026may.engine.static_bs1.fp16`。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。
- `worker_nums` 保持 1。该 worker 持有自己的 execution context。方案不引入共享 context，也不引入跨 worker 的 context 池。
- 不重编时，`docs/benchmark-h3/libface.txt` 的 Throughput 1668.97 qps 仍是这个 id 的引擎参照。重编后，该文件里的新 Throughput 才是参照。本条不产生 HTTP RPS。
- 显存仍是这一份 static batch=1 engine 加上这一个 context。不另建 context 池。

## 验收标准

1. 打开 `libface.toml`，`model_input_image_size` 为 `[480, 640]`，`type` 为 `tensorrt`，`model_file_path` 仍指向 `face_detection_yunet_2026may.engine.static_bs1.fp16`。
2. 打开 `conf/trt_profiles/set_a_libface.json`，名为 `input` 的 `min`、`opt`、`max` 都是 `[1, 3, 480, 640]`，三者相同。
3. 打开 `docs/benchmark-h3/libface.txt`，命令含 `--shapes=input:1x3x480x640`，同一文件有硬件、日期和 `Throughput:`。未重编时，日期为 2026-10-10，Throughput 为 1668.97 qps。重编后，这三行来自同一次退出码为 0 的 `trtexec`，形状串仍是 `input:1x3x480x640`。
4. 用改完后的 toml 拉起 `libface`。向 `http://127.0.0.1:9051/mortred_ai_server_v1/obj_detection/libface` 提交 `demo_data/model_test_input/object_detection/face_wo_mask.jpg`。HTTP 状态为 200，响应 JSON 的 `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0。
5. `libface_server_config.toml` 的 `worker_nums` 为 1，`max_batch_size` 为 1。
6. 在 `/mnt/g/Codex/mortred_model_server` 执行 `python3 scripts/product_static_bs1.py --check`，退出码为 0。

## 测试义务

QA 按白盒核对 diff：改动停在上面的改动面；`trt_session.cpp` 与按 worker 建会话的循环没有改成共享 context 或 context 池；profile 的三档彼此相等且 N 为 1；`realesrgan` 的文件不在 diff 里。再按验收标准 1 至 6 从外部判定。一张 JPEG 的成功状态即通过，不要求框非空，不要求 HTTP RPS。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面每条的期望退出码都是 0。使用 gpu profile 二进制 `build/full/bin/mortred-model-server.out`。cpu profile 二进制不用于本条。

```bash
cd /mnt/g/Codex/mortred_model_server
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/_lib:/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/g/Codex/mortred_model_server")
toml = (root / "conf/model/object_detection/libfacedetection/libface.toml").read_text(encoding="utf-8")
assert "model_input_image_size = [480, 640]" in toml
prof = json.loads((root / "conf/trt_profiles/set_a_libface.json").read_text(encoding="utf-8-sig"))
assert prof[0]["name"] == "input"
assert prof[0]["min"] == prof[0]["opt"] == prof[0]["max"] == [1, 3, 480, 640]
bench = (root / "docs/benchmark-h3/libface.txt").read_text(encoding="utf-8")
assert "--shapes=input:1x3x480x640" in bench and "Throughput:" in bench
print("shape-agree")
PY
python3 scripts/product_static_bs1.py --check
3rd_party/bin/trtexec --loadEngine=weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16 --warmUp=200 --duration=2 --avgRuns=10 --useSpinWait --fp16 --shapes=input:1x3x480x640
```

`trtexec` 退出 0 后，再对改完 toml 之后拉起的进程做一次请求。9051 上若已有旧进程，先停掉旧进程，再启动这一次。

```bash
cd /mnt/g/Codex/mortred_model_server
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/_lib:/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build/full/bin/mortred-model-server.out --model libface conf/server/object_detection/libface_det/libface_server_config.toml >/tmp/libface-p0-1.log 2>&1 &
echo $! >/tmp/libface-p0-1.pid
for _ in $(seq 1 60); do
  curl -sf -o /dev/null http://127.0.0.1:9051/ready && break
  sleep 1
done
curl -sS -D /tmp/libface-p0-1.hdr -o /tmp/libface-p0-1.json -H "Content-Type: image/jpeg" --data-binary @demo_data/model_test_input/object_detection/face_wo_mask.jpg http://127.0.0.1:9051/mortred_ai_server_v1/obj_detection/libface
python3 - <<'PY'
from pathlib import Path
hdr = Path("/tmp/libface-p0-1.hdr").read_text(encoding="utf-8", errors="replace")
body = Path("/tmp/libface-p0-1.json").read_text(encoding="utf-8")
assert " 200" in hdr.splitlines()[0]
assert '"status":0' in body.replace(" ", "") and '"status_str":"OK"' in body.replace(" ", "")
print("jpeg-ok")
PY
kill "$(cat /tmp/libface-p0-1.pid)"
```

这条请求脚本的期望退出码是 0。`/ready` 在 60 秒内没有返回 HTTP 200，或响应不是上面的成功状态，则退出码非 0。

## RD 不得自行决定

- 高和宽只取高 480、宽 640。不得改回高 640、宽 480，也不得另选尺寸。
- `min`、`opt`、`max` 必须相同，N 必须为 1。不得做成动态 shape 或动态 batch。
- 现有 engine 对 `input:1x3x480x640` 的 `trtexec` 退出 0 时，不得 `--force` 重编，不得手改 `docs/benchmark-h3/libface.txt` 的 Throughput。
- 探测失败时只允许按推荐做法重编一次。第二次探测失败即停。
- 不得把 `type` 改成 `onnx` 或 `mnn`，不得改 `[height, width]` 的解析顺序，不得让多个 worker 共享一个 execution context 或增加 context 池。
- 不得改 `realesrgan`，不得把本条扩成 calibrate 或 HTTP RPS 报告。
