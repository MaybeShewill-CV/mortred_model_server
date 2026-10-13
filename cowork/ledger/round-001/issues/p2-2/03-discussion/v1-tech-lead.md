# 技术方案

- 问题: P2-2
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-2

## 边界

锁定句照录：打开 gateway 限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。场景是把 `[gateway.rate_limit]` 打开，或设置 `MORTRED_RATE_LIMIT=enforce`。同一 IP 先用错误 token 打满 burst，再带正确 token 推理。现在 `conf/mortred.toml` 里 `enabled = false` 时不计量。开关打开后，`check_ip_rate_limit` 在路由和鉴权之前调用 `gcra_admit`，允许的请求会推进 TAT。随后鉴权失败仍返回 401，但这部分请求已经消耗来源 IP 的配额。同一 IP 用错误 token 打满 burst 后，带正确 token 的请求收到 429 和 `Retry-After`。按 key 的配额只在 key 已允许推理之后扣减。改完后能观察到的变化：打开同一限流开关，用错误 token 打满原先的 burst 次数，同一 IP 紧接着的合法推理仍然成功。401 不改变该来源的剩余配额。

本方案只做这一处：最终 HTTP 状态是这条鉴权 401 时，该来源的剩余瞬时名额与这次请求未曾计入时相同。`enabled = false` 且未设置 `MORTRED_RATE_LIMIT` 时仍不计量。

不把 P2-5 的 token 启动检查拉进来。不改 TensorRT execution context，不改 `worker_nums`，不做动态 batch。不跑 calibrate，不跑 `http_infer_rps.py`。`conf/server` 的 `worker_nums` 保持 1。健康检查豁免不在此列。上一版审查意见无。

## 做法

### 推荐

来源限流仍在路由解析和鉴权之前做一次 admit。admit 结果为拒绝时，行为与现在相同：enforce 下返回 429，TAT 不推进。admit 结果为允许、而这次请求最终在进入转发之前以 401 结束时，在处理函数返回之前，按该策略的一个发射间隔归还这一次占用。归还后的 TAT 等于「这次允许没有发生」。已经计入的更早请求仍留在 TAT 里。

没有 Authorization 和错误 Bearer 都走现有的「未授权」分支，都返回现有的 401，都归还。外部验收使用错误 Bearer。按 key 的 `consume_rate_limit` 仍只在 key 已允许推理之后扣减，错误 token 到不了那里。

代价：从这次 admit 成功到归还完成，该来源暂时少一个瞬时名额。401 不访问上游，这段只覆盖路由查找和鉴权比较。验收规定下一次请求在上一次状态行已经收到之后才发出，验收序列看不到这段暂扣。归还是该来源所在分片锁内的一次整数减法，不新增线程，不提高 `max_tracked`，不分配显存，不生成 engine。

### 否决

把 admit 挪到鉴权成功之后。附带写法是先看 Bearer，错误或缺失就跳过 admit。路由未命中的 404 会不再占用配额。来源已经打满时，错误 token 得到 401，L1 不再在鉴权之前丢掉这批请求。锁定句只要求 401 不改变剩余配额。

401 仍推进 TAT，另放一个「鉴权失败不计入」的并行计数，允许决策改读并行计数。`Retry-After` 仍按 TAT 计算，会和剩余名额分成两套数。现有 GCRA 单测以一个 TAT 单元为配额来源，两套状态会漂。

## 改动面

- `src/control/rate_limit/gcra.h`：在现有分片存储上增加「归还一次已允许的占用」。`gcra_admit` 的允许条件、拒绝时不写 TAT、发射间隔、burst 容差、空闲清扫、容量打满时拒绝新来源、分配失败时放行，保持现有公式。
- `src/control/gateway/gateway_app.cpp`：仅在「这次 admit 为允许，且最终在进入转发之前返回 401」的路径上调用归还。429 的响应头保持现有写法。
- `test/rate_limit_gcra_unittest.cc`：按下述测试义务增加归还用例。
- `test/gateway_multiinstance_test.cc`：改掉把 401 当成「L1 已占用」的断言，并按下述测试义务增加两个用例。

## 明确不改

- `conf/mortred.toml` 的 `[gateway.rate_limit]`：`enabled = false`、`shadow = false`、`rate_per_sec = 100000`、`burst = 200000`、`trusted_proxies = "127.0.0.1,::1"`。
- `conf/server` 全部 toml 的 `worker_nums`。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。
- `MORTRED_RATE_LIMIT` 的覆盖规则：`enforce` 打开 enforce，`shadow` 打开 shadow，其他非空值把两个开关都清掉。未设置该变量时沿用 toml。
- 来源身份：`src/control/rate_limit/subject.h` 的可信代理、`Forwarded`、`X-Forwarded-For` 与前缀折叠。
- 调用顺序里限流之前的三段：`OPTIONS`、`/healthz`、`/metrics`。`/metrics` 鉴权失败的 401 仍在 admit 之前返回，不归还，也不占用。
- 路由未命中的 404 仍占用。已鉴权之后的 405 仍占用。已鉴权并已转发之后的 502、503 仍占用。有效 key 触发的按 key 429 仍占用来源配额，且仍返回 429 与 `Retry-After`，不返回 401。
- `src/control/api_key_manager.h` 的 `consume_rate_limit`。
- supervisor 进程树、catalog 的产品 id、TensorRT execution context、engine 文件、`docs/benchmark-h3/`。不执行 `mortredctl prepare`。
- 推理 token、管理 token、metrics token 的启动检查。本条不增加、不放宽启动拒绝条件。

## 契约

计数关闭时（`enabled` 与 `shadow` 都为假，包括 `MORTRED_RATE_LIMIT` 把二者清掉）：请求不进入 admit，不建来源表项，不写 429。错误 Bearer 仍是 401。

计数打开时，enforce 与 shadow 共用同一条归还规则：这次 admit 为允许，且处理函数在进入转发之前把状态写成 401，则按一个发射间隔归还，然后返回。归还不删除表项，不清空该来源已有的成功占用。shadow 下已鉴权的请求仍计入，且响应仍不是 429。不得只在 enforce 分支归还。

401 响应保持现有的 `WWW-Authenticate` 与统一错误 JSON，不带 `Retry-After`，不带 `RateLimit-Limit`、`RateLimit-Remaining`、`RateLimit-Reset`。

enforce 下真正超过瞬时名额的请求仍是 429，仍带 `Retry-After`（GCRA 毫秒数向上取整到秒）以及 `RateLimit-Limit`、`RateLimit-Remaining: 0`、`RateLimit-Reset`。

已鉴权请求的来源配额仍在转发之前扣下。验收里的「合法推理成功」指：客户端收到 HTTP 200，响应头 `X-Mortred-Model` 为该路由的 catalog id，正文为上游原样返回的 `ok`。上游是验收脚本里的临时进程，对任意 POST 固定回 200 和正文 `ok`。不要求模型 JSON 的 `status`。

产品进程布局不变。gateway 仍是单独进程。验收不启动 supervisor，不把临时上游写进驻留集。

## 性能与资源

不生成 engine，不增加 execution context，不加 context 池，不改 `worker_nums`。来源表的分片数和 `max_tracked` 不变。归还只发生在 401 路径，锁内一次减法。常驻显存与本条无关。

验收使用临时 toml：`rate_per_sec = 1`，`burst = 3`，`trusted_proxies` 为空。发射间隔为 1000 毫秒，瞬时名额为 3。这组数字只用于验收，不写回产品 toml。序列必须串行，连接地址固定 `127.0.0.1`，不发送 `Forwarded` 和 `X-Forwarded-For`。从该序列第一次推理请求发出，到该序列最后一次响应收完，墙钟须小于 900 毫秒，避免间隔补进一次新的允许。

## 验收标准

1. 打开仓库 `conf/mortred.toml`，`[gateway.rate_limit]` 这一节同时含有 `enabled = false`、`shadow = false`、`rate_per_sec = 100000`、`burst = 200000`、`trusted_proxies = "127.0.0.1,::1"`。
2. 新的 gateway 进程，临时 toml 为 `enabled = false`、`shadow = false`、`rate_per_sec = 1`、`burst = 3`、`trusted_proxies` 为空，环境中没有 `MORTRED_RATE_LIMIT`。同一 `127.0.0.1` 串行 POST `/v1/models/p2_2_fake/infer`：3 次 Bearer `wrong-p2-2` 都是 401，且没有 `Retry-After`；随后 4 次 Bearer `infer-p2-2` 都是 200，`X-Mortred-Model` 为 `p2_2_fake`，正文为 `ok`。这 7 次里没有 429。该序列墙钟小于 900 毫秒。
3. 另一个新的 gateway 进程，临时 toml 改为 `enabled = true`，其余与第 2 条相同，环境中没有 `MORTRED_RATE_LIMIT`。同一地址串行：3 次 `wrong-p2-2` 都是 401 且没有 `Retry-After`；紧接着 3 次 `infer-p2-2` 都是 200，头和正文与第 2 条的 200 相同；再一次 `infer-p2-2` 是 429，`Retry-After` 为 `1`，`RateLimit-Limit` 为 `1`，`RateLimit-Remaining` 为 `0`，`RateLimit-Reset` 为 `1`。该序列墙钟小于 900 毫秒。
4. 另一个新的 gateway 进程，临时 toml 为 `enabled = false`，其余数字与第 2 条相同，环境 `MORTRED_RATE_LIMIT=enforce`。请求序列和判定与第 3 条相同。
5. 另一个新的 gateway 进程，配置与第 3 条相同。同一地址串行：1 次 `infer-p2-2` 为 200；随后 3 次 `wrong-p2-2` 为 401 且没有 `Retry-After`；随后 2 次 `infer-p2-2` 为 200；再一次 `infer-p2-2` 为 429，四个限流头与第 3 条的 429 相同。该序列墙钟小于 900 毫秒。

第 2 至第 5 条各自使用独立进程和空的来源表。临时目录不包含 `conf/api_keys.toml`。三个 token 彼此不同：推理 `infer-p2-2`，管理 `admin-p2-2`，metrics `metrics-p2-2`。请求不使用管理 token。就绪只使用一次 `GET /healthz`，这次探测不计入上述次数。

## 测试义务

QA 先跑下面的白盒，再跑自测命令里的外部脚本。外部脚本覆盖验收标准 1 至 5。不跑 calibrate，不跑 `http_infer_rps.py`，不跑 `mortredctl prepare`，不启动 `mortred-model-server.out` 和 supervisor。

- `release_undoes_one_admission_and_keeps_earlier_ones`：`rate_per_sec = 10`、`burst = 5`、时间固定为 0。连续两次 admit 都允许；归还一次；再连续四次 admit 都允许；下一次拒绝。从未 admit 的 key 上归还，不新建表项，随后仍可连续允许 5 次，第 6 次拒绝。
- `wrong_token_preserves_full_burst`：进程内 gateway，`enabled = true`，`rate_per_sec = 1`，`burst = 3`，`trusted_proxies` 为空，上游端口无人监听。3 次错误 Bearer 为 401。随后 3 次正确推理 token 为 503。再一次正确推理 token 为 429，且带 `Retry-After`。
- `wrong_token_does_not_restore_spent_admission`：同一限流数字。1 次正确推理 token 为 503；3 次错误 Bearer 为 401；随后 2 次正确推理 token 为 503；再一次为 429，且带 `Retry-After`。
- 已有的 `ip_rate_limit_sheds_and_reports_retry_after` 改为：`rate_per_sec = 3`、`burst = 3`。8 次不带 Authorization 的请求都是 401，0 次 429。随后 8 次带该实例推理 token 的请求里，3 次为 503，5 次为 429；每次 429 都有 `Retry-After` 和 `RateLimit-Limit`、`RateLimit-Remaining`、`RateLimit-Reset`。
- 已有的 `xff_from_trusted_peer_buckets_by_client`：占用改为带推理 token 的请求。同一 `X-Forwarded-For` 连续 4 次为 3 次 503 和 1 次 429。三个不同客户端地址各 1 次，都是 503。
- 已有的 `xff_from_untrusted_peer_is_ignored`：6 次轮换 `X-Forwarded-For` 且带推理 token，为 3 次 503 和 3 次 429。
- 已有的 `shadow_mode_never_rejects`、`healthz_is_exempt_from_ip_limit`、`rate_limited_key_gets_429_not_401` 仍按当前断言通过。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面每一条的期望退出码都是 0。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。不执行 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。

第一条核对验收标准 1。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
from pathlib import Path
text = Path("conf/mortred.toml").read_text(encoding="utf-8")
start = text.index("[gateway.rate_limit]")
block = text[start:].split("\n[", 1)[0]
need = [
    "enabled = false",
    "shadow = false",
    "rate_per_sec = 100000",
    "burst = 200000",
    'trusted_proxies = "127.0.0.1,::1"',
]
missing = [line for line in need if line not in block]
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
PY
```

第二条编译并跑白盒。两个新的 gateway 用例名和归还用例名必须出现在列表里，随后两份测试都通过。

```bash
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target rate_limit_gcra_unittest gateway_multiinstance_test mortred-gateway.out
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./_bin/rate_limit_gcra_unittest --gtest_list_tests > /tmp/p2-2-gcra.txt
./_bin/gateway_multiinstance_test --gtest_list_tests > /tmp/p2-2-gw.txt
grep -F 'release_undoes_one_admission_and_keeps_earlier_ones' /tmp/p2-2-gcra.txt
grep -F 'wrong_token_preserves_full_burst' /tmp/p2-2-gw.txt
grep -F 'wrong_token_does_not_restore_spent_admission' /tmp/p2-2-gw.txt
./_bin/rate_limit_gcra_unittest
./_bin/gateway_multiinstance_test
```

第三条覆盖验收标准 2 至 5。gateway 二进制是 `_bin/mortred-gateway.out`。临时根在 `/tmp`，结束时删除。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import os, socket, subprocess, time
from pathlib import Path

ROOT = Path("/tmp/p2-2-rate-limit")
BIN = Path("/mnt/g/Codex/mortred_model_server/_bin/mortred-gateway.out")
LIB = "/mnt/g/Codex/mortred_model_server/3rd_party/libs"

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

class Upstream:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["python3", "-c", UPSTREAM],
            stdout=subprocess.PIPE, text=True)
        self.port = int(self.proc.stdout.readline().strip())

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)

UPSTREAM = r"""
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0))
s.listen(64)
print(s.getsockname()[1], flush=True)
while True:
    c, _ = s.accept()
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = c.recv(4096)
        if not chunk:
            break
        data += chunk
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
    c.close()
"""

def write_root(gw_port, up_port, enabled):
    if ROOT.exists():
        subprocess.check_call(["rm", "-rf", str(ROOT)])
    (ROOT / "conf" / "server").mkdir(parents=True)
    (ROOT / "conf" / "server" / "p2_2_fake.toml").write_text(
        "\n".join([
            "[p2_2_fake_SERVER]",
            'model = "p2_2_fake"',
            "port = %d" % up_port,
            'server_uri = "/upstream/p2_2_fake"',
            'server_exe = "fake_model_server.out"',
            'profile = "any"',
            "",
        ]),
        encoding="utf-8",
    )
    (ROOT / "conf" / "mortred.toml").write_text(
        "\n".join([
            "[gateway]",
            'host = "127.0.0.1"',
            "port = %d" % gw_port,
            "",
            "[gateway.rate_limit]",
            "enabled = %s" % ("true" if enabled else "false"),
            "shadow = false",
            "rate_per_sec = 1",
            "burst = 3",
            'trusted_proxies = ""',
            "",
        ]),
        encoding="utf-8",
    )

def http(port, token):
    s = socket.create_connection(("127.0.0.1", port), timeout=2)
    req = (
        "POST /v1/models/p2_2_fake/infer HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Connection: close\r\n"
        "Authorization: Bearer %s\r\n"
        "Content-Length: 0\r\n\r\n" % token
    )
    s.sendall(req.encode())
    raw = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        raw += chunk
    s.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return status, headers, body

def wait_ready(port):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.2)
            s.sendall(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            data = s.recv(256)
            s.close()
            if b" 200 " in data.split(b"\r\n", 1)[0]:
                return
        except OSError:
            time.sleep(0.05)
    raise SystemExit("gateway did not become ready")

def run_sequence(enabled, env_mode, steps, label):
    up = Upstream()
    gw_port = free_port()
    write_root(gw_port, up.port, enabled)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    env["MORTRED_PROJECT_ROOT"] = str(ROOT)
    env["MORTRED_PROFILE"] = "gpu"
    env.pop("MORTRED_CONTROL_CONFIG", None)
    env.pop("MORTRED_RATE_LIMIT", None)
    if env_mode:
        env["MORTRED_RATE_LIMIT"] = env_mode
    env["MORTRED_GATEWAY_AUTH_TOKEN"] = "infer-p2-2"
    env["MORTRED_API_TOKEN"] = "admin-p2-2"
    env["MORTRED_METRICS_TOKEN"] = "metrics-p2-2"
    env["MORTRED_INTERNAL_TOKEN"] = "internal-p2-2"
    gw = subprocess.Popen([str(BIN)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        wait_ready(gw_port)
        t0 = time.monotonic()
        seen = []
        for token, want in steps:
            status, headers, body = http(gw_port, token)
            seen.append((status, headers, body))
            if status != want:
                raise SystemExit("%s: want %s got %s" % (label, want, status))
            if want == 401 and "retry-after" in headers:
                raise SystemExit("%s: 401 has Retry-After" % label)
            if want == 200:
                if headers.get("x-mortred-model") != "p2_2_fake" or body != b"ok":
                    raise SystemExit("%s: forwarded 200 mismatch" % label)
            if want == 429:
                got = (
                    headers.get("retry-after"),
                    headers.get("ratelimit-limit"),
                    headers.get("ratelimit-remaining"),
                    headers.get("ratelimit-reset"),
                )
                if got != ("1", "1", "0", "1"):
                    raise SystemExit("%s: 429 headers %s" % (label, got))
        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms >= 900:
            raise SystemExit("%s: elapsed %.0f ms" % (label, elapsed_ms))
    finally:
        gw.kill()
        gw.wait(timeout=5)
        up.stop()

fresh = [("wrong-p2-2", 401)] * 3 + [("infer-p2-2", 200)] * 3 + [("infer-p2-2", 429)]
try:
    run_sequence(False, None, [("wrong-p2-2", 401)] * 3 + [("infer-p2-2", 200)] * 4, "disabled")
    run_sequence(True, None, fresh, "enabled")
    run_sequence(False, "enforce", fresh, "env-enforce")
    run_sequence(
        True,
        None,
        [("infer-p2-2", 200)] + [("wrong-p2-2", 401)] * 3 + [("infer-p2-2", 200)] * 2 + [("infer-p2-2", 429)],
        "partial",
    )
finally:
    subprocess.call(["rm", "-rf", str(ROOT)])
PY
```

## RD 不得自行决定

- 归还的量。一次归还只减去一个发射间隔。不得删除表项，不得把 TAT 清零，不得按 burst 重置。
- 哪些响应归还。只归还「这次 admit 为允许，且在进入转发之前返回 401」。404、405、502、503、按 key 的 429 不归还。admit 为拒绝的请求不归还。`/metrics` 的 401 不归还。
- 归还的位置。写在 401 返回路径上，处理函数返回之前。不放到转发回调里。
- shadow 与 enforce 是否分叉。两条模式都归还这种 401。已鉴权请求在 shadow 下仍计入且仍不返回 429。
- 产品 toml 的五个限流字段，以及验收用的 `rate_per_sec = 1`、`burst = 3` 是否写回产品文件。验收数字只留在临时 toml。
- 来源身份、GCRA 允许条件、拒绝路径不写 TAT、容量与 fail-open。
- 是否改 `worker_nums`、execution context、动态 batch，以及是否用真实 TensorRT 推理代替临时上游的 HTTP 200 作为通过条件。
- 是否在本条里改 token 启动检查。验收必须设置三个互不相同的非空 token，使现有启动条件能够听端口。
