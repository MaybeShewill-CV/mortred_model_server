# 技术方案

- 问题: P2-7
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` v3 的 P2-7

## 边界

本方案只改这一件可观察的事：已经拉起的 `realesrgan`，对一张解码后宽或高不是 256 的合法 JPEG，一次 HTTP 响应是输入错误，正文写明尺寸须为 256×256。引擎保持 static shape、batch=1。不要求动态 shape。

成功返回增强图，且送入会话的形状为 `1×256×256×3`，不作为通过。非法 JPEG、缺图、鉴权失败不写入通过条件。P2-9、P2-10、P2-11 以及其他 id 不在本方案内。

现在的路径是：`realesrgan.toml` 指向 `realesr-general-x4v3.engine.static_bs1.fp32`。`docs/benchmark-h3/realesrgan.txt` 含 `--shapes=input:1x256x256x3`。`preprocess` 不做 resize，按工作图的行数和列数组 NHWC。静态引擎上 GPU 零拷贝不合格：`Resize::NONE` 通不过静态维检查，输入维也不是动态，请求落到 CPU 预处理。宽或高不是 256 时，`setInputShape` 失败，`TrtSession::run` 返回状态 2，HTTP 500，`status_str` 为 `model run session failed`。这句正文没有 `256×256`。预处理若只交回空张量，基类会收成状态 3，`status_str` 为 `model input empty`，正文同样没有这句。

## 做法

### 推荐

在 `realesrgan` 解码得到工作图之后、组 NHWC 之前判断行数和列数。宽或高不是 256 时，不组张量，不调用 `session().run`，因此也不调用 `setInputShape`。`run` 返回一条新的输入状态，整数值固定为 8，枚举文案全文是 `尺寸须为 256×256`。HTTP 层把 8 映到 400。统一响应里 `status` 与 `results[0].status` 都是 8，`status_str` 就是这句。调用方一次响应就能看到输入错误和尺寸句子。

判断用的是解码后的工作图：行数是高，列数是宽。边长小于 10 的已解码图同样走状态 8，不再先交回空张量。通道不符、解码失败仍走现有状态，那些情况不在本条通过里。宽和高都是 256 时，这一张图只解码一次，仍走现有预处理和现有会话，不做 resize。

代价：全局状态表增加一条只由 `realesrgan` 返回的线协议码。非 256 的请求不再进入这次 `enqueue`。正好 256 的驻留引擎、显存和 execution context 保持原样。

### 否决

把图缩放到 256 再推理。代价是一次缩放和一次成功推理，调用方拿到增强图。锁定的通过是输入错误并写明 256×256。现有预处理不做 resize。不采用。

按 `conf/trt_profiles/set_a_realesrgan.json` 的 64 到 640 重编动态引擎，或在 `setInputShape` 失败之后把状态 2 改写成尺寸句子。动态引擎会让非 256 成功返回增强图，显存按 profile 的 max 常驻，并离开 static shape、batch=1。事后改写会把 dtype 不匹配、`enqueue` 失败，以及其他模型的形状拒绝都写成「256×256」。尺寸判断放在送入会话之前，并且只针对 `realesrgan` 这一次已解码的工作图。不采用。

## 改动面

- `src/common/status_code.h`：增加一条状态，名字 `MODEL_INPUT_SIZE_REJECTED`，整数值 8，文案全文 `尺寸须为 256×256`。已有状态的数值和文案保持不动。
- `src/server/http_status.h`：状态 8 返回 HTTP 400，与 `MODEL_EMPTY_INPUT_IMAGE` 同一类。
- `src/models/enhancement/real_esrgan.h` 与 `src/models/enhancement/real_esrgan.inl`：在基类把空张量收成状态 3 之前返回状态 8。两边都是 256 时仍走现有的一次预处理和一次会话。
- `test/status_code_unittest.cc`：为数值 8 增加稳定值断言。文案断言沿用现有的全表检查。

## 明确不改

- `src/models/backend/trt_session.cpp`。`setInputShape` 失败时仍返回 `MODEL_RUN_SESSION_FAILED`。每个 session 仍在自己的 `init` 里 `createExecutionContext()` 一次。不共享 context，不加跨 worker 的 context 池。
- `src/models/backend/backend_cv_model.h` 里空张量收成 `MODEL_EMPTY_INPUT_IMAGE` 的规则，以及其他模型的预处理。
- `realesrgan` 的 GPU 预处理描述：仍是 `Resize::NONE`、`dynamic_size = true`、NHWC。不把它改成拉到 256 的 resize，也不把这条静态引擎的零拷贝改成合格。
- `conf/model/enhancement/real_esrgan/realesrgan.toml` 的 `type = "tensorrt"` 与 `realesr-general-x4v3.engine.static_bs1.fp32`。
- `conf/server/enhancement/real_esrgan/realesrgan_server_config.toml` 的 `worker_nums=1`、`max_batch_size=1`、端口 9012、URI。
- `conf/trt_profiles/set_a_realesrgan.json`、`docs/benchmark-h3/realesrgan.txt`。不执行 `mortredctl prepare`。
- 统一响应的字段集合。不新增 JSON 字段。`errors` 可以不出现。句子放在已有的 `status_str`。
- `src/server/openapi_doc.h`、gateway、鉴权、`worker_nums` 的环境覆盖。

## 契约

请求仍是现有的统一信封或 raw body。验收用 `Content-Type: image/jpeg`，正文是这一张 JPEG 的字节。URI 仍是 `/mortred_ai_server_v1/enhancement/realesrgan`。

对解码后宽或高不是 256 的这一张合法 JPEG，响应同时是：

- HTTP 400
- JSON `status` 为 8
- `status_str` 全文等于 `尺寸须为 256×256`。乘号是 U+00D7，`为` 与 `256` 之间一个空格
- `results` 长度为 1，`results[0].status` 为 8，`results[0].data` 为 null
- `partial` 为 false
- `model.name` 为 `realesrgan`

状态 2 与 HTTP 500 不再是这一张图的结果。状态 3 的文案仍是 `model input empty`，不能代替这句。状态 66 与 HTTP 422 留给信封和参数校验，本条不用。已有状态的数值和文案不变。`status_str` 仍由枚举文案生成。

送入会话的引擎形状仍是名为 `input` 的 `1×256×256×3`，batch=1。非 256 的请求不送入会话。正好 256×256 时张量仍与该引擎一致；这一支不作为本条通过。

## 性能与资源

常驻的仍是一个 `realesrgan` 进程。server toml 的 `worker_nums=1`，进程内一个 worker，持有它自己的那一个 execution context。产品文件仍指向 `realesr-general-x4v3.engine.static_bs1.fp32`。不生成新引擎，不按 profile 的 max 增加显存。

尺寸拒绝发生在组 NHWC 和 `enqueue` 之前，不增加 CUDA 分配。GPU 零拷贝保持不合格，这次请求仍不走那条路径。`docs/benchmark-h3/realesrgan.txt` 的 Throughput `50.7951 qps` 保持不动，不作为本条数字。不跑 calibrate，不跑 `http_infer_rps.py`。

## 验收标准

1. 启动前 `127.0.0.1:9012/ready` 连不上。随后拉起的可执行文件是仓库 `_bin/mortred-model-server.out`，`/proc/<pid>/cwd` 是仓库的 `_bin`。该进程环境没有 `MORTRED_AUTH_TOKEN`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_IMAGE_DECODE_BACKEND`、`MORTRED_MODEL_CONFIG_FILE`。`127.0.0.1:9012/ready` 在 90 秒内返回 HTTP 200。
2. 使用一张无方向标记的彩色 JPEG。用 OpenCV 解码后高为 240、宽为 320、通道为 3。高和宽都不是 256，且都大于等于 10。
3. 向 `/mortred_ai_server_v1/enhancement/realesrgan` POST 这一次，`Content-Type` 为 `image/jpeg`，正文是这张 JPEG 的字节，不带 `Authorization`。HTTP 状态为 400。JSON `status` 为 8，`status_str` 全文等于 `尺寸须为 256×256`，`partial` 为 false，`model.name` 为 `realesrgan`。`results` 长度为 1，`results[0].status` 为 8，`results[0].data` 为 null。
4. 同一次响应的 HTTP 状态不是 500，JSON `status` 不是 0 也不是 2，`results[0].data` 里没有增强图。日志里的句子不算通过。
5. `conf/model/enhancement/real_esrgan/realesrgan.toml` 仍含 `realesr-general-x4v3.engine.static_bs1.fp32`。`docs/benchmark-h3/realesrgan.txt` 仍含 `--shapes=input:1x256x256x3`。

## 测试义务

QA 先跑下面两条自测命令，再按验收标准 1 至 5 做外部判定。外部判定就是已拉起的 `realesrgan` 上这一次 HTTP 请求。不另发正好 256×256 的成功请求，不把成功增强图当作通过。不发送非法 JPEG、空正文或错误 token 来充当通过。不跑 calibrate，不跑 `http_infer_rps.py`，不跑 `mortredctl prepare`。

白盒核对 diff：改动停在改动面列出的四个位置。`trt_session.cpp` 里 `setInputShape` 失败仍返回 `MODEL_RUN_SESSION_FAILED`，`createExecutionContext()` 仍是每个 session 一次。尺寸条件是解码后的宽或高不等于 256，不是只拒绝 320×240，也不是无条件拒绝。正好 256×256 仍组 NHWC 并进入现有会话，且不做 resize。GPU 描述仍是 `Resize::NONE`。产品 toml 的 engine 路径不在 diff 里。

`status_code_unittest` 里数值 8 的断言通过，全表文案检查里这条的文案是 `尺寸须为 256×256`。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面两条的期望退出码都是 0。

第一条编译服务器和状态码测试，并跑整份 `status_code_unittest`。

```bash
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target mortred-model-server.out
cmake --build build/full --target status_code_unittest
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs:/mnt/g/Codex/mortred_model_server/_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./_bin/status_code_unittest
```

第二条覆盖验收标准 1 至 5。开始时 `127.0.0.1:9012/ready` 必须连不上，连得上则退出码非 0，不杀掉已有进程。结束时只停止本命令启动的 pid。工作目录是仓库的 `_bin`，可执行文件是 `./mortred-model-server.out`。JPEG 写到 `/tmp/p2-7-320x240.jpg`，响应用 `/tmp/p2-7-body.json`。`python3` 使用本机已有的 `cv2`（`scripts/calibrate_pack.py` 已导入它）。不设置上面列出的五个环境变量。不执行 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。

```bash
cd /mnt/g/Codex/mortred_model_server
grep -q 'realesr-general-x4v3.engine.static_bs1.fp32' conf/model/enhancement/real_esrgan/realesrgan.toml
grep -q -- '--shapes=input:1x256x256x3' docs/benchmark-h3/realesrgan.txt
curl -sf --max-time 1 http://127.0.0.1:9012/ready >/dev/null && exit 1
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs:/mnt/g/Codex/mortred_model_server/_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
model_pid=""
cleanup() {
  if [ -n "$model_pid" ]; then kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
cd /mnt/g/Codex/mortred_model_server/_bin
env -u MORTRED_AUTH_TOKEN -u MORTRED_WORKER_NUMS -u MORTRED_DECODE_AUTO -u MORTRED_IMAGE_DECODE_BACKEND -u MORTRED_MODEL_CONFIG_FILE \
  ./mortred-model-server.out --model realesrgan ../conf/server/enhancement/real_esrgan/realesrgan_server_config.toml >/tmp/p2-7-realesrgan.log 2>&1 &
model_pid=$!
cd /mnt/g/Codex/mortred_model_server
python3 - << 'PY'
import json, os, time, urllib.request
import cv2
import numpy as np

root = "/mnt/g/Codex/mortred_model_server"
pid = int(open("/proc/self/status").read())  # replaced below
PY
python3 - << 'PY'
import json, os, time, urllib.request
import cv2
import numpy as np

root = "/mnt/g/Codex/mortred_model_server"
pid = int(os.environ["P2_7_PID"])
exe = os.path.realpath(f"/proc/{pid}/exe")
cwd = os.path.realpath(f"/proc/{pid}/cwd")
assert exe == root + "/_bin/mortred-model-server.out", exe
assert cwd == root + "/_bin", cwd
env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
keys = {e.split(b"=", 1)[0].decode() for e in env if b"=" in e}
for name in ("MORTRED_AUTH_TOKEN", "MORTRED_WORKER_NUMS", "MORTRED_DECODE_AUTO",
             "MORTRED_IMAGE_DECODE_BACKEND", "MORTRED_MODEL_CONFIG_FILE"):
    assert name not in keys, name

img = np.zeros((240, 320, 3), np.uint8)
img[:, :] = (20, 40, 60)
path = "/tmp/p2-7-320x240.jpg"
assert cv2.imwrite(path, img)
got = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
assert got is not None and got.shape == (240, 320, 3), got.shape if got is not None else None

deadline = time.time() + 90
ready = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:9012/ready", timeout=2) as resp:
            ready = resp.status
            if ready == 200:
                break
    except Exception:
        time.sleep(0.5)
assert ready == 200, ready

data = open(path, "rb").read()
req = urllib.request.Request(
    "http://127.0.0.1:9012/mortred_ai_server_v1/enhancement/realesrgan",
    data=data,
    headers={"Content-Type": "image/jpeg"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        status = resp.status
        body = resp.read()
except urllib.error.HTTPError as err:
    status = err.code
    body = err.read()
open("/tmp/p2-7-body.json", "wb").write(body)
assert status == 400, status
payload = json.loads(body.decode("utf-8"))
phrase = "尺寸须为 256×256"
assert payload["status"] == 8, payload["status"]
assert payload["status_str"] == phrase, payload["status_str"]
assert payload["status_str"][5] == "\u00d7"
assert payload["partial"] is False
assert payload["model"]["name"] == "realesrgan"
assert len(payload["results"]) == 1
assert payload["results"][0]["status"] == 8
assert payload["results"][0]["data"] is None
assert payload["status"] not in (0, 2)
print("p2-7-ok")
PY
```

第二条里，把服务器放到后台之后，用同一 shell 的 `model_pid` 传给 Python：在 `python3` 之前加 `export P2_7_PID="$model_pid"`。上面第一段只为占位的 Python 不要执行。完整顺序是：grep、确认 9012 空闲、启动服务器、`export P2_7_PID="$model_pid"`、再执行从 `import json` 开始的那一个 Python。该 Python 退出码为 0，并打印 `p2-7-ok`。整条命令的期望退出码是 0。

## RD 不得自行决定

- 状态整数值只取 8，文案只取 `尺寸须为 256×256`（乘号 U+00D7）。不得改成状态 2、3 或 66，不得改成 HTTP 422 或 500，不得把句子只写进日志。
- 不得把图缩放到 256 再推理，不得把动态 shape 或重编引擎当作本条的实现。
- 不得修改 `setInputShape` 的全局失败码，不得共享 execution context，不得增加 context 池。
- 不得把正好 256×256 的请求也返回状态 8。那一支仍进现有会话；成功增强图不作为本条通过，验收也不发送那张图。
- 不得把非法 JPEG、缺图、401 写进通过条件，不得改其他 catalog id。
- 验收 JPEG 的尺寸固定为解码后 320×240。实现上的判断是任意已解码工作图的宽或高不等于 256，不得只识别 320×240。
