# 技术方案

- 问题: P2-5
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-5

## 边界

锁定句照录：推理 token 或管理 token 缺失，或二者相同而 metrics token 不同时，gateway 仍然启动（OP4）。场景是推理 token 缺失、管理 token 缺失，或推理 token 与管理 token 相同而 metrics token 不同。直接启动 gateway 验收，不使用 Docker。现在 `scrape_token_usable` 只拒绝空的 metrics token，以及 metrics token 与推理 token 或管理 token 相同。推理 token 为空且管理 token 与 metrics token 不同时检查通过；管理 token 为空且推理 token 与 metrics token 不同时同样通过；推理 token 与管理 token 相同、metrics token 为第三个值时也通过。gateway 在「推理 token 或管理 token 至少有一个」或已加载 API key 文件时继续监听。`security_warn.sh` 在推理 token 与管理 token 相同时只打印警告。`docker-compose.yml` 的 `${VAR:?}` 只拒绝未设置，三个变量设成相同非空值时 compose 仍会启动容器。metrics token 为空，或与推理 token、管理 token 之一相同时，gateway 会拒绝启动。改完后能观察到的变化：推理 token 缺失、管理 token 缺失，或推理 token 与管理 token 相同而 metrics token 不同时，gateway 以非 0 退出，不监听，日志写明拒绝原因。三个 token 都已设置且彼此不同时，gateway 可以进入监听。

本方案只做这一处启动判定。三个名字固定为环境变量：推理 `MORTRED_GATEWAY_AUTH_TOKEN`，管理 `MORTRED_API_TOKEN`，metrics `MORTRED_METRICS_TOKEN`。未设置和空字符串都是缺失。比较是整串相等。短于 32 字符的非空值可以监听。

验收进程是 `mortred-gateway.out`。临时根里可以有一份含一个 key 的 `conf/api_keys.toml`，该文件不能使上述三种情况进入监听。三个彼此不同的非空 token 在同一份 key 文件存在时仍可监听。这条验收不需要 GPU，不需要模型推理。

不把 P2-6、P2-7、P2-11 拉进来。不改 TensorRT execution context 的归属，不改动态 batch，不改限流归还。上一版审查意见无。

## 做法

### 推荐

在 `src/control/trust_tokens.h` 增加一个启动判定，供 gateway 的 `init` 和 supervisor 拉起 `__gateway` 之前共用。判定顺序固定：

1. metrics token 为空，或与已设置的推理 token、管理 token 之一相同：保持现有拒绝和现有日志。
2. 推理 token 缺失：拒绝，原因短语为 `MORTRED_GATEWAY_AUTH_TOKEN is unset`。
3. 管理 token 缺失：拒绝，原因短语为 `MORTRED_API_TOKEN is unset`。
4. 推理 token 与管理 token 都非空且相同：拒绝，原因短语为 `MORTRED_GATEWAY_AUTH_TOKEN and MORTRED_API_TOKEN are identical`。
5. 三个都非空且两两不同：通过。

`GatewayApp::init` 在绑定端口之前调用该判定。失败时向 stderr 写 `mortred-gateway: refusing to start: ` 加上对应原因短语，返回失败，`run` 以非 0 结束，进程不监听。已加载的 API key 不跳过这次判定。判定函数不接收 key 文件是否加载。

`ProcessSupervisor::spawn_locked` 在 `exec` 之前、在「executable not found」之前调用同一判定。失败时不 `exec`。错误字符串含同一原因短语，并被现有永久失败判定认出：`__gateway` 状态为 `failed`，不进入 `backoff`。supervisor 主进程自身仍按现有规则启动管理面。`SupervisorApp` 在 autostart 前的提示改用该判定的原因短语，避免把推理 token 缺失或两者相同打印成 metrics 相撞。

代价：现在只设置其中一个静态 token、或只靠 `conf/api_keys.toml` 就能监听的进程，会改为启动失败。进程内测试和 `gateway_e2e_test` 里断言「已监听」的用例要补上三个彼此不同的 token。判定本身是启动前的几次字符串比较，不新开进程，不占显存，不生成 engine。

### 否决

只在 `gateway_app.cpp` 里把「至少一个静态 token」改成「两个都非空且不同」，supervisor 拉起 `__gateway` 时仍只调用 `scrape_token_usable`。直接启动 gateway 可以满足锁定句。supervisor 仍会 `exec` gateway，子进程随即以非 0 退出；`__gateway` 的重启策略是 `always`，失败会进入重试，而不是现有 metrics 错误那种永久失败。两处判定以后会分开。

只改 `scripts/security_warn.sh`、`mortredctl doctor --strict` 或 `docker-compose.yml`，让警告、doctor 或 compose 在 token 缺失或相同时报错。直接执行的 `mortred-gateway.out` 仍然监听。锁定场景要求直接启动 gateway，并且不使用 Docker。`security_warn.sh` 的 32 字符长度是 doctor 的警告，放进监听条件后，短于 32 字符的三个不同 token 不能监听。

## 改动面

- `src/control/trust_tokens.h`：保留 `scrape_token_usable` 的现有 metrics 规则。新增上述启动判定和三条原因短语。metrics 缺失短语仍是 `MORTRED_METRICS_TOKEN is unset`。metrics 相撞短语仍是 `MORTRED_METRICS_TOKEN matches an inference or management token`。
- `src/control/gateway/gateway_app.cpp`：`init` 在 `listen` 之前按该判定失败返回。stderr 使用 `mortred-gateway: refusing to start: ` 加原因短语。日志不写 token 的值。API key 已加载时同样执行该判定。
- `src/control/supervisor.cpp`：`spawn_locked` 对 `__gateway` 使用同一判定，失败不 `exec`，错误进入现有永久失败。
- `src/control/supervisor/supervisor_app.cpp`：gateway 二进制存在且判定失败时，管理面仍启动，提示含同一原因短语。
- `conf/api_keys.toml.example`：开头说明改为三个静态 token 都已设置且两两不同时 gateway 才能监听；key 文件不能代替缺失的推理 token 或管理 token。hash、scope、`enabled` 的写法不变。
- `test/trust_tokens_unittest.cc`、`test/supervisor_unittest.cc`、`test/gateway_e2e_test.cc`、`test/gateway_multiinstance_test.cc`：按下述测试义务修改。

## 明确不改

- `scrape_token_usable` 对空推理 token、空管理 token 的 metrics 碰撞规则。现有单测 `empty_infer_does_not_collide`、`empty_admin_does_not_collide` 仍针对这个函数。
- metrics 为空，或与推理 token、管理 token 之一相同时的拒绝日志和拒绝结果。
- 监听之后的请求鉴权：推理 Bearer、管理 Bearer、API key 的 200 与 401，`/healthz` 无 Bearer 返回 200，`/metrics` 仍只接受 metrics token。
- `[gateway.rate_limit]`、`MORTRED_RATE_LIMIT`、401 的配额归还。
- supervisor 主进程对空 `MORTRED_API_TOKEN` 的拒绝。这三种 gateway token 问题不使 supervisor 主进程退出。
- `scripts/security_warn.sh`、`scripts/mortredctl_doctor.sh`、`scripts/mortredctl_init-trust.sh`、`scripts/docker_entrypoint.sh`、`docker-compose.yml`、`deploy/`。
- `conf/mortred.toml`、`conf/server` 的 `worker_nums`、catalog 的产品 id、模型进程、TensorRT execution context、engine、动态 batch。
- token 长度、空白修剪、`MORTRED_INTERNAL_TOKEN`。内部 token 缺失不构成本条拒绝。

## 契约

环境变量未设置或值为空字符串，都算该 token 缺失。非空包括长度小于 32 的字符串，也包括只含空格的字符串。相等是字节串相等。

判定顺序就是上一节的 1 到 5。推理 token 和管理 token 都缺失时，日志使用推理 token 缺失的短语。metrics 问题先于这三条，沿用现有两句全文，不加新前缀。

新的三条在 gateway stderr 上的全文是：

- `mortred-gateway: refusing to start: MORTRED_GATEWAY_AUTH_TOKEN is unset`
- `mortred-gateway: refusing to start: MORTRED_API_TOKEN is unset`
- `mortred-gateway: refusing to start: MORTRED_GATEWAY_AUTH_TOKEN and MORTRED_API_TOKEN are identical`

这三句都不包含当次 token 的值。失败进程的退出码非 0，且不出现 `mortred-gateway listening on`。

通过时，三个 token 都非空且两两不同。gateway 打出现有监听日志 `mortred-gateway listening on http://127.0.0.1:<port>`，`GET /healthz` 为 200，正文为 `ok\n`。`conf/api_keys.toml` 里有一个非空 hash 时，上述拒绝仍然拒绝，上述通过仍然监听。

supervisor 拉起 `__gateway` 失败时，错误字符串含同一原因短语，状态为 `failed`。管理面继续监听。模型子进程的拉起条件不变。

## 性能与资源

启动前的字符串比较，发生在绑定端口之前。不生成 engine，不增加 execution context，不加 context 池，不改 `worker_nums`，不读 GPU。常驻显存与本条无关。验收不部署模型进程。

## 验收标准

四次都直接执行 `mortred-gateway.out`。临时根含 `conf/mortred.toml`（`[gateway]` 的 `host = "127.0.0.1"`）、空的 `conf/server/`，以及一份 `conf/api_keys.toml`：一个 `[keys.p2-5]`，`hash` 为 64 个 `a`，`scope = "inference"`，`enabled = true`。子进程环境只带指定变量，不继承调用方的 `MORTRED_*`。监听地址为 `127.0.0.1` 和当次空闲端口。token 取短于 32 字符的字面量。

1. 不设置 `MORTRED_GATEWAY_AUTH_TOKEN`。`MORTRED_API_TOKEN` 为 `p2-5-admin`，`MORTRED_METRICS_TOKEN` 为 `p2-5-metrics`。进程退出码非 0。stderr 含 `mortred-gateway: refusing to start: MORTRED_GATEWAY_AUTH_TOKEN is unset`，不含 `mortred-gateway listening on`，不含 `p2-5-admin` 和 `p2-5-metrics`。进程退出后，对该端口的 `GET /healthz` 失败。
2. 不设置 `MORTRED_API_TOKEN`。`MORTRED_GATEWAY_AUTH_TOKEN` 为 `p2-5-infer`，`MORTRED_METRICS_TOKEN` 为 `p2-5-metrics`。进程退出码非 0。stderr 含 `mortred-gateway: refusing to start: MORTRED_API_TOKEN is unset`，不含 `mortred-gateway listening on`，不含 `p2-5-infer` 和 `p2-5-metrics`。进程退出后，对该端口的 `GET /healthz` 失败。
3. `MORTRED_GATEWAY_AUTH_TOKEN` 与 `MORTRED_API_TOKEN` 都为 `p2-5-same`，`MORTRED_METRICS_TOKEN` 为 `p2-5-metrics`。进程退出码非 0。stderr 含 `mortred-gateway: refusing to start: MORTRED_GATEWAY_AUTH_TOKEN and MORTRED_API_TOKEN are identical`，不含 `mortred-gateway listening on`，不含 `p2-5-same` 和 `p2-5-metrics`。进程退出后，对该端口的 `GET /healthz` 失败。
4. `MORTRED_GATEWAY_AUTH_TOKEN` 为 `p2-5-infer`，`MORTRED_API_TOKEN` 为 `p2-5-admin`，`MORTRED_METRICS_TOKEN` 为 `p2-5-metrics`。进程保持运行。stderr 含 `mortred-gateway listening on http://127.0.0.1:` 和当次端口，不含这三个 token 字面量。`GET /healthz` 为 200，正文为 `ok\n`。探测完成后结束该进程。

## 测试义务

QA 先跑白盒，再跑自测命令里的外部脚本。外部脚本覆盖验收标准 1 至 4。不启动 supervisor，不启动 `mortred-model-server.out`，不跑 `mortredctl doctor`、`prepare`、`calibrate`，不跑 `http_infer_rps.py`，不使用 Docker。

- `trust_tokens_unittest`：`scrape_token_usable` 的现有用例保持。新增 `missing_infer_is_not_usable_for_listen`、`missing_admin_is_not_usable_for_listen`、`identical_infer_and_admin_is_not_usable_for_listen`、`distinct_three_are_usable_for_listen`。前三个分别对应三条原因短语；第三个的 metrics 与那两个相同值不同。第四个在三个两两不同的非空值上通过。两个都空时，原因短语是推理 token 缺失。
- `supervisor_unittest`：新增 `gateway_missing_infer_token_is_permanent_failure`、`gateway_missing_admin_token_is_permanent_failure`、`gateway_identical_infer_and_admin_is_permanent_failure`。三者都是 `start_server("__gateway")` 失败，错误含对应原因短语，状态为 `failed`，不是 `backoff`。已有的 metrics 缺失和 metrics 相撞永久失败保持。
- `gateway_e2e_test`：断言进程已监听的用例带上三个彼此不同的非空 token。只靠 API key 或只靠一个静态 token 就断言监听成功的用例改为断言启动失败。`KeysOnlyLoopbackRejectsAnonymousRequests`、`KeysOnlyLoopbackAcceptsValidKey`、`NonLoopbackKeysOnlyStartsAndEnforcesKeys`、`AdminScopeKeyMayInfer` 在三个 token 补齐后保持原请求断言。`AdminApiTokenIsAccepted`、`NonLoopbackWithOnlyApiToken` 补上不同的推理 token 后，管理 Bearer 仍为 200，错误 Bearer 仍为 401。`EmptyApiKeysWithTokenFallsBackToStaticToken`、`BrokenApiKeysWithTokenFallsBackToStaticToken`、`StaticTokenFallbackStillWorksWhenConfigured`、`MetricsTokenRejectsAnonymousAndWrongBearer` 补上不同的管理 token 后保持原请求断言。metrics 缺失，以及 metrics 与推理 token 或管理 token 相同，仍拒绝启动。
- `gateway_multiinstance_test`：每个断言 `init` 成功的实例补上与该实例推理 token、metrics token 都不同的 `admin_token`。限流次数、401、429、503 的断言保持。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面两条的期望退出码都是 0。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。不执行 `mortredctl prepare`、`mortredctl calibrate`、`http_infer_rps.py`，不使用 Docker。

第一条编译并跑白盒。四个新的判定名和三个新的永久失败名必须出现在列表里，随后四份测试都通过。

```bash
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target trust_tokens_unittest supervisor_unittest gateway_e2e_test gateway_multiinstance_test mortred-gateway.out
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./_bin/trust_tokens_unittest --gtest_list_tests > /tmp/p2-5-trust.txt
./_bin/supervisor_unittest --gtest_list_tests > /tmp/p2-5-sup.txt
grep -F 'missing_infer_is_not_usable_for_listen' /tmp/p2-5-trust.txt
grep -F 'missing_admin_is_not_usable_for_listen' /tmp/p2-5-trust.txt
grep -F 'identical_infer_and_admin_is_not_usable_for_listen' /tmp/p2-5-trust.txt
grep -F 'distinct_three_are_usable_for_listen' /tmp/p2-5-trust.txt
grep -F 'gateway_missing_infer_token_is_permanent_failure' /tmp/p2-5-sup.txt
grep -F 'gateway_missing_admin_token_is_permanent_failure' /tmp/p2-5-sup.txt
grep -F 'gateway_identical_infer_and_admin_is_permanent_failure' /tmp/p2-5-sup.txt
./_bin/trust_tokens_unittest
./_bin/supervisor_unittest
./_bin/gateway_e2e_test
./_bin/gateway_multiinstance_test
```

第二条覆盖验收标准 1 至 4。gateway 二进制是 `_bin/mortred-gateway.out`。临时根在 `/tmp/p2-5-tokens`，结束时删除。拒绝案例里 gateway 自身的退出码非 0；这条命令的退出码为 0。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import os, socket, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path("/tmp/p2-5-tokens")
BIN = Path("/mnt/g/Codex/mortred_model_server/_bin/mortred-gateway.out")
LIB = "/mnt/g/Codex/mortred_model_server/3rd_party/libs"
KEYS = "\n".join([
    "[keys.p2-5]",
    'hash = "%s"' % ("a" * 64),
    'scope = "inference"',
    "enabled = true",
    "",
])

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def write_root(port):
    if ROOT.exists():
        subprocess.check_call(["rm", "-rf", str(ROOT)])
    (ROOT / "conf" / "server").mkdir(parents=True)
    (ROOT / "conf" / "mortred.toml").write_text(
        "[gateway]\nhost = \"127.0.0.1\"\nport = %d\n" % port,
        encoding="utf-8",
    )
    (ROOT / "conf" / "api_keys.toml").write_text(KEYS, encoding="utf-8")

def healthz(port):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/healthz" % port, timeout=1
        ) as resp:
            return resp.status == 200 and resp.read() == b"ok\n"
    except Exception:
        return False

def base_env(port):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "LD_LIBRARY_PATH": LIB,
        "MORTRED_PROJECT_ROOT": str(ROOT),
        "MORTRED_GATEWAY_HOST": "127.0.0.1",
        "MORTRED_GATEWAY_PORT": str(port),
    }
    return env

def refuse(port, tokens, phrase, hidden):
    write_root(port)
    env = base_env(port)
    env.update(tokens)
    proc = subprocess.run(
        [str(BIN)], env=env, capture_output=True, text=True, timeout=10
    )
    err = proc.stderr
    if proc.returncode == 0:
        raise SystemExit("expected non-zero: %s" % phrase)
    if phrase not in err:
        raise SystemExit("missing phrase %r in %r" % (phrase, err))
    if "mortred-gateway listening on" in err:
        raise SystemExit("listening line present for %s" % phrase)
    for value in hidden:
        if value in err:
            raise SystemExit("token value leaked: %s" % value)
    if healthz(port):
        raise SystemExit("healthz still open after refuse: %s" % phrase)

def allow(port, tokens):
    write_root(port)
    env = base_env(port)
    env.update(tokens)
    proc = subprocess.Popen(
        [str(BIN)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read()
                raise SystemExit("gateway exited early: %s" % err)
            if healthz(port):
                break
            time.sleep(0.05)
        else:
            raise SystemExit("healthz not ready")
        err = ""
        # stderr is a pipe; the listening line is written before healthz returns
        time.sleep(0.2)
        proc.terminate()
        try:
            _, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate(timeout=5)
        line = "mortred-gateway listening on http://127.0.0.1:%d" % port
        if line not in err:
            raise SystemExit("missing listen line in %r" % err)
        for value in tokens.values():
            if value in err:
                raise SystemExit("token value leaked: %s" % value)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

port = free_port()
refuse(
    port,
    {"MORTRED_API_TOKEN": "p2-5-admin", "MORTRED_METRICS_TOKEN": "p2-5-metrics"},
    "mortred-gateway: refusing to start: MORTRED_GATEWAY_AUTH_TOKEN is unset",
    ["p2-5-admin", "p2-5-metrics"],
)
port = free_port()
refuse(
    port,
    {"MORTRED_GATEWAY_AUTH_TOKEN": "p2-5-infer", "MORTRED_METRICS_TOKEN": "p2-5-metrics"},
    "mortred-gateway: refusing to start: MORTRED_API_TOKEN is unset",
    ["p2-5-infer", "p2-5-metrics"],
)
port = free_port()
refuse(
    port,
    {
        "MORTRED_GATEWAY_AUTH_TOKEN": "p2-5-same",
        "MORTRED_API_TOKEN": "p2-5-same",
        "MORTRED_METRICS_TOKEN": "p2-5-metrics",
    },
    "mortred-gateway: refusing to start: MORTRED_GATEWAY_AUTH_TOKEN and MORTRED_API_TOKEN are identical",
    ["p2-5-same", "p2-5-metrics"],
)
port = free_port()
allow(
    port,
    {
        "MORTRED_GATEWAY_AUTH_TOKEN": "p2-5-infer",
        "MORTRED_API_TOKEN": "p2-5-admin",
        "MORTRED_METRICS_TOKEN": "p2-5-metrics",
    },
)
subprocess.check_call(["rm", "-rf", str(ROOT)])
PY
```

## RD 不得自行决定

- 不得用 `conf/api_keys.toml` 代替缺失的推理 token 或管理 token，也不得因此让相同的两个静态 token 进入监听。
- 不得只改 `security_warn.sh`、`mortredctl doctor` 或 `docker-compose.yml` 来充当本条拒绝。
- 不得把 token 长度、空白修剪，或 metrics 与另外两者相同的现有拒绝，改成另一套规则。
- 不得在拒绝或监听日志里打印 token 的值。
- 不得让 supervisor 主进程因这三种 gateway token 问题退出。`__gateway` 的这类失败不得进入 `backoff`。
- 不得改监听之后的 Bearer 和 API key 规则，不得改限流归还、TensorRT execution context 的归属、动态 batch、停止信号。
- 不得把 P2-6、P2-7、P2-11 的改动放进这次实现。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | cowork/ledger/round-001/issues/p2-5/03-discussion/v1-sign-pm.md |
| RD | 同意 | 无 | v1 | cowork/ledger/round-001/issues/p2-5/03-discussion/v1-sign-rd.md |
| 技术负责人 | 同意 | 无 | v1 | cowork/ledger/round-001/issues/p2-5/03-discussion/v1-sign-tech-lead.md |
