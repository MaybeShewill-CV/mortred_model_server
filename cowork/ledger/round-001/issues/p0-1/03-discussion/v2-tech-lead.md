# 技术方案

- 问题: P0-1
- 版本: v2
- 对应列表: v3

## 边界

本方案只让已拉起的 `libface` 对一张合法 JPEG 返回成功状态，并让下面三处的高和宽相同：

- `conf/model/object_detection/libfacedetection/libface.toml` 的 `model_input_image_size`。`src/models/object_detection/detection_params.h` 把该数组定为 `[height, width]`。
- `conf/trt_profiles/set_a_libface.json` 里名为 `input` 的 NCHW profile。
- `docs/benchmark-h3/libface.txt` 的 `--shapes=input:1x3x480x640`。

对齐后的形状是高 480、宽 640、batch=1。当前 toml `[640, 480]` 与 profile `[1, 3, 640, 480]` 会送出 `1×3×640×480`，静态引擎拒绝后返回 `MODEL_RUN_SESSION_FAILED`。基准文件里的命令已经是 `input:1x3x480x640`。该文件的通过条件只有一种，不把其中某个 qps 或某一天写成另一套通过条件。

TensorRT 保持 static shape、batch=1。每个 worker 仍各自持有一个 execution context。验收打在模型端口 `9051` 上，不经过 gateway `:8080`。

## 做法

### 推荐

把产品 toml 和 profile 改到基准文件已经写出的静态形状。检测器继续按 toml 的 `[height, width]` 做 resize，再打成 NCHW。480 和 640 都能被 32 整除，现有的右/下补齐不会再放大这张量。GPU 预处理和 CPU 预处理读的是同一个 toml 尺寸，一处修改同时覆盖两条被拒绝的路径。

拉起进程时工作目录用仓库的 `_bin`，可执行文件用 `_bin/mortred-model-server.out`。server toml 里的 `../conf/...` 和产品 toml 里的 `../weights/...` 按这个工作目录打开。请求带上该进程正在使用的 Bearer。`libface_server_config.toml` 没有 `auth_token`；非空的 `MORTRED_AUTH_TOKEN` 就是这个进程的 token。推理 URI 会走鉴权，空 token 在会话之前返回 401。`/ready` 不查这个头。

代价：改两份配置，并改 `scripts/product_static_bs1.py` 里锁死 `model_input_image_size = [640, 480]` 的那一句守卫，否则 `python3 scripts/product_static_bs1.py --check` 会失败。`trtexec --loadEngine` 对 `input:1x3x480x640` 退出 0 时，不改 `docs/benchmark-h3/libface.txt`。该命令退出非 0 时，在 profile 已经改成三档相同的 `[1, 3, 480, 640]` 之后，执行一次 `mortredctl prepare --id libface --force`，再用含 `--shapes=input:1x3x480x640` 的 `trtexec` 覆盖该文件。覆盖后仍是四行结构：第一行硬件，第二行 `YYYY-MM-DD`，第三行命令，以及含 `Throughput:` 的一行。覆盖结果与原文件都用验收标准 3 这一组判定。第二次探测退出非 0 即停，不再换尺寸。

### 否决

把三处改成高 640、宽 480，并按现有 profile 重编 engine。代价是必须重编。`scripts/convert_trt_engines.sh` 对静态输入的 ONNX 会丢掉 profile；若这份 YuNet ONNX 的静态输入已是 `1×3×480×640`，重编结果仍是高 480、宽 640，toml 继续送出 `1×3×640×480`，请求仍然失败。`scripts/bench_set_a/h3_lines.py`、`scripts/bench_set_a/check_h3_lines.py` 和 `docs/benchmark-set-a.md` 也已经写着 `input:1x3x480x640`。

在 `libface` 预处理里对调高宽，toml 仍写 `[640, 480]`。三份文件的高宽仍然不同，锁定的变化达不到。对调也和所有检测器共用的 `[height, width]` 约定冲突。把 profile 改成动态范围则超出 static shape、batch=1。

## 改动面

- `conf/model/object_detection/libfacedetection/libface.toml`：`model_input_image_size` 改为 `[480, 640]`。`type` 仍是 `tensorrt`，engine 路径、输入名 `input`、12 个输出名、`input_layout = "nchw"` 不动。
- `conf/trt_profiles/set_a_libface.json`：唯一绑定 `input` 的 `min`、`opt`、`max` 都改为 `[1, 3, 480, 640]`。
- `scripts/product_static_bs1.py`：libface 尺寸守卫改为要求 toml 含 `model_input_image_size = [480, 640]`。只改这一句守卫。
- `docs/benchmark-h3/libface.txt`：仅在推荐做法里那一次 `prepare --force` 之后，用这次 `trtexec` 覆盖。探测退出 0 时这个文件不改。

## 明确不改

- `src/models/object_detection/libface_detector.inl`、`src/models/object_detection/detection_params.h`、`src/models/backend/trt_session.cpp`。
- `src/server/generic_cv_server.h` 里按 worker 各自 `init` 的循环。每个 worker 继续创建自己的 `TrtSession`，并在该会话里 `createExecutionContext`。
- `conf/server/object_detection/libface_det/libface_server_config.toml` 的 `worker_nums = 1`、`max_batch_size = 1`、端口 9051、URI，以及其中没有 `auth_token` 这一事实。token 由启动环境的 `MORTRED_AUTH_TOKEN` 提供。
- 鉴权函数、`/ready` 的豁免、推理 URI 必须带 Bearer 的现有顺序。
- `src/control/supervisor.cpp`。supervisor 已在 `execve` 之前把工作目录换到 `<project_root>/_bin`；本条自测与它对齐，不改它。
- `conf/ci/libface_onnx_hosted.toml`、`conf/onnx_sources.json`、其他 id 的 profile 和 `docs/benchmark-h3/`。
- `realesrgan` 的配置、profile、engine 和代码。
- HTTP 路径、字段名和响应信封。本条不包含 calibrate，也不包含 `http_infer_rps.py` 的发布 RPS 报告。本条不走 gateway。

## 契约

- `model_input_image_size = [480, 640]` 表示高 480、宽 640。
- profile 的 `min`、`opt`、`max` 同为 NCHW `[1, 3, 480, 640]`。
- 送入会话的输入名仍是 `input`，形状是 `1×3×480×640`，batch=1。12 个输出名不变。
- 可执行文件是 `_bin/mortred-model-server.out`。`cmake --preset full` 把它写到这里。启动时工作目录是仓库的 `_bin`，`../conf` 与 `../weights` 按这个目录打开。
- 该进程正在使用的 Bearer 是它环境里的 `MORTRED_AUTH_TOKEN`。server toml 没有 `auth_token`，这个变量非空时覆盖空配置。请求头是 `Authorization: Bearer <该值>`。
- 验收请求是 `POST http://127.0.0.1:9051/mortred_ai_server_v1/obj_detection/libface`。正文是 `demo_data/model_test_input/object_detection/face_wo_mask.jpg` 的原始字节。`Content-Type` 为 `image/jpeg`。成功响应是 HTTP 200，JSON `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0。
- 缺这个头、头里的 token 与进程环境不一致，或进程的 token 为空，推理 URI 返回 401。这次 401 发生在会话之前。`/ready` 不要求这个头。
- 框和关键点仍按现有解码映射回原图像素。本条不规定框的数值。

## 性能与资源

- 继续使用本机本卡上的 `face_detection_yunet_2026may.engine.static_bs1.fp16`。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。engine 在本机本卡生成。
- `worker_nums` 保持 1。该 worker 持有自己的 execution context。方案不引入共享 context，也不引入跨 worker 的 context 池。
- 该 id 的引擎参照就是 `docs/benchmark-h3/libface.txt` 里的 `Throughput:`。本条不产生 HTTP RPS，也不把某个固定 qps 写成通过条件。
- 显存仍是这一份 static batch=1 engine 加上这一个 context。

## 验收标准

1. 打开 `libface.toml`，`model_input_image_size` 为 `[480, 640]`，`type` 为 `tensorrt`，`model_file_path` 仍指向 `face_detection_yunet_2026may.engine.static_bs1.fp16`。
2. 打开 `conf/trt_profiles/set_a_libface.json`，名为 `input` 的 `min`、`opt`、`max` 都是 `[1, 3, 480, 640]`，三者相同。
3. 打开 `docs/benchmark-h3/libface.txt`，只按这一组同时成立判定通过：命令含 `--shapes=input:1x3x480x640`；同一文件有一行 `YYYY-MM-DD` 日期；同一文件有一行含 `Throughput:`；同一文件另有一行硬件描述，该行不是 trtexec 命令，不是 `Throughput:` 行，也不是单独的日期行。高 480、宽 640 与 `model_input_image_size = [480, 640]`、profile 里 `input` 的 `[1, 3, 480, 640]` 相同。打开这一份文件就能判定这一项。
4. 这一次 `libface` 的可执行文件是仓库 `_bin/mortred-model-server.out`，`/proc/<pid>/cwd` 是仓库的 `_bin`。
5. 向该进程提交 `demo_data/model_test_input/object_detection/face_wo_mask.jpg` 的原始字节，`Content-Type` 为 `image/jpeg`，`Authorization: Bearer` 的值等于该进程环境里的 `MORTRED_AUTH_TOKEN`。响应是 HTTP 200，JSON `status` 为 0，`status_str` 为 `OK`，`results[0].status` 为 0。
6. `libface_server_config.toml` 的 `worker_nums` 为 1，`max_batch_size` 为 1。
7. 在 `/mnt/g/Codex/mortred_model_server` 执行 `python3 scripts/product_static_bs1.py --check`，退出码为 0。

## 测试义务

QA 按白盒核对 diff：改动停在上面的改动面；`trt_session.cpp` 与按 worker 建会话的循环没有改成共享 context 或 context 池；profile 的三档彼此相等且 N 为 1；鉴权代码和 server toml 的 token 字段不在 diff 里；`realesrgan` 的文件不在 diff 里。再按验收标准 1 至 7 从外部判定。一张带该进程 Bearer 的 JPEG 返回上述成功状态即通过。401、工作目录是仓库根、或可执行文件不是 `_bin/mortred-model-server.out`，都不通过。不要求框非空，不要求 HTTP RPS。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面三条的期望退出码都是 0。使用 gpu profile 已生成的 `_bin/mortred-model-server.out`。cpu profile 二进制不用于本条。

第一条，工作目录是仓库根。它核对 toml、profile，以及基准文件的那一种通过结果。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import json
import re
from pathlib import Path
root = Path("/mnt/g/Codex/mortred_model_server")
toml = (root / "conf/model/object_detection/libfacedetection/libface.toml").read_text(encoding="utf-8")
assert "model_input_image_size = [480, 640]" in toml
assert 'type = "tensorrt"' in toml
assert "face_detection_yunet_2026may.engine.static_bs1.fp16" in toml
prof = json.loads((root / "conf/trt_profiles/set_a_libface.json").read_text(encoding="utf-8-sig"))
assert len(prof) == 1 and prof[0]["name"] == "input"
assert prof[0]["min"] == prof[0]["opt"] == prof[0]["max"] == [1, 3, 480, 640]
bench = (root / "docs/benchmark-h3/libface.txt").read_text(encoding="utf-8")
assert "--shapes=input:1x3x480x640" in bench
assert "Throughput:" in bench
assert re.search(r"\d{4}-\d{2}-\d{2}", bench)
lines = [ln.strip() for ln in bench.splitlines() if ln.strip()]
assert any(
    "trtexec" not in ln and "Throughput" not in ln and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ln)
    for ln in lines
)
print("shape-agree")
PY
python3 scripts/product_static_bs1.py --check
```

第二条，工作目录仍是仓库根。期望退出码是 0。退出非 0 时按推荐做法重编一次后再执行同一条，期望退出码仍是 0。第二次非 0 即停。

```bash
cd /mnt/g/Codex/mortred_model_server
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/_lib:/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
3rd_party/bin/trtexec --loadEngine=weights/object_detection/libfacedetection/face_detection_yunet_2026may.engine.static_bs1.fp16 --warmUp=200 --duration=2 --avgRuns=10 --useSpinWait --fp16 --shapes=input:1x3x480x640
```

第三条，工作目录是仓库的 `_bin`。可执行文件是 `./mortred-model-server.out`。9051 上若已有进程，先停掉，再启动这一次。Bearer 的取法：启动前在同一 shell 导出非空 `MORTRED_AUTH_TOKEN=libface-p0-1`；该值写入 `/tmp/libface-p0-1.token`，请求头使用它。验收时用 `tr '\0' '\n' < /proc/<pid>/environ` 读出 `MORTRED_AUTH_TOKEN`，请求里的 Bearer 必须与这一行相同。server toml 里没有这个 token。`/ready` 不带这个头。推理请求带上。期望退出码是 0。`/ready` 在 60 秒内没有返回 HTTP 200、进程路径不对、或响应不是上面的成功状态，则退出码非 0。

```bash
set -euo pipefail
cd /mnt/g/Codex/mortred_model_server/_bin
unset MORTRED_WORKER_NUMS MORTRED_MODEL_CONFIG_FILE
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/_lib:/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MORTRED_AUTH_TOKEN=libface-p0-1
printf '%s' "$MORTRED_AUTH_TOKEN" >/tmp/libface-p0-1.token
pidfile=/tmp/libface-p0-1.pid
cleanup() {
  if [ -f "$pidfile" ]; then
    kill -INT "$(cat "$pidfile")" 2>/dev/null || true
    wait "$(cat "$pidfile")" 2>/dev/null || true
  fi
}
trap cleanup EXIT
./mortred-model-server.out --model libface ../conf/server/object_detection/libface_det/libface_server_config.toml >/tmp/libface-p0-1.log 2>&1 &
echo $! >"$pidfile"
ready=0
for _ in $(seq 1 60); do
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:9051/ready; then
    ready=1
    break
  fi
  sleep 1
done
test "$ready" -eq 1
readlink -f /proc/"$(cat "$pidfile")"/cwd >/tmp/libface-p0-1.cwd
readlink -f /proc/"$(cat "$pidfile")"/exe >/tmp/libface-p0-1.exe
python3 - <<'PY'
from pathlib import Path
root = "/mnt/g/Codex/mortred_model_server"
cwd = Path("/tmp/libface-p0-1.cwd").read_text(encoding="utf-8").strip()
exe = Path("/tmp/libface-p0-1.exe").read_text(encoding="utf-8").strip()
assert cwd == root + "/_bin", cwd
assert exe == root + "/_bin/mortred-model-server.out", exe
print("process-path-ok")
PY
curl -sS -D /tmp/libface-p0-1.hdr -o /tmp/libface-p0-1.json \
  -H "Authorization: Bearer $(cat /tmp/libface-p0-1.token)" \
  -H "Content-Type: image/jpeg" \
  --data-binary @/mnt/g/Codex/mortred_model_server/demo_data/model_test_input/object_detection/face_wo_mask.jpg \
  http://127.0.0.1:9051/mortred_ai_server_v1/obj_detection/libface
python3 - <<'PY'
import json
from pathlib import Path
hdr = Path("/tmp/libface-p0-1.hdr").read_text(encoding="utf-8", errors="replace")
code = int(hdr.splitlines()[0].split()[1])
body = json.loads(Path("/tmp/libface-p0-1.json").read_text(encoding="utf-8"))
assert code == 200
assert body["status"] == 0
assert body["status_str"] == "OK"
assert body["results"][0]["status"] == 0
print("jpeg-ok")
PY
```

## RD 不得自行决定

- 高和宽只取高 480、宽 640。不得改回高 640、宽 480，也不得另选尺寸。
- `min`、`opt`、`max` 必须相同，N 必须为 1。不得做成动态 shape 或动态 batch。
- 基准文件只按验收标准 3 那一组通过。不得再写一套「某个日期、某个 qps」和另一套「重编后的日期、qps」。
- `trtexec` 对 `input:1x3x480x640` 退出 0 时，不得 `--force` 重编，不得手改 `docs/benchmark-h3/libface.txt`。探测失败时只允许按推荐做法重编一次。第二次探测失败即停。
- 启动的可执行文件必须是 `_bin/mortred-model-server.out`，工作目录必须是仓库的 `_bin`。不得从仓库根执行 `build/full/bin/mortred-model-server.out`。
- 推理请求必须带该进程的 `MORTRED_AUTH_TOKEN`。不得把 token 写进 server toml，不得改鉴权代码，不得把空 token 当成通过。
- 自测不得继承非空的 `MORTRED_WORKER_NUMS` 或 `MORTRED_MODEL_CONFIG_FILE`。
- 不得把 `type` 改成 `onnx` 或 `mnn`，不得改 `[height, width]` 的解析顺序，不得让多个 worker 共享一个 execution context 或增加 context 池。
- 不得运行 `product_static_bs1.py --apply-models` 来重写 `libface.toml`。不得改 `realesrgan`，不得把本条扩成 calibrate、HTTP RPS 报告或 gateway 验收。
