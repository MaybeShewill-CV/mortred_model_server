# 技术方案

- 问题: D-12
- 版本: v1
- 对应列表: `cowork/ledger/round-012/02-p0-p1-final.md` 的 D-12

## 边界

只做已经冻结的 D-12。冻结句不改写：一次拉起后，检查失败并点名 fork 与 exec 之间的调用。

改完后能观察到的变化只这一句：同一条检查通过，且模型仍监听 `127.0.0.1`，并带上 `MORTRED_AUTH_TOKEN` 或网关的 `MORTRED_METRICS_TOKEN`。

本条没有上一版方案，没有待并入的审查条件。

现在的窗口在 `src/control/supervisor.cpp` 的 `ProcessSupervisor::spawn_locked`。`fork` 在子进程分支之前。子进程从 `if (pid == 0)` 到对应的 `execl` 之间调用 `setenv`，网关端口用 `std::to_string`，并用 `signal` 把 `SIGINT`、`SIGQUIT` 设回 `SIG_DFL`。同段调用 `close_all_fds_except_stdio`，该函数体内调用 `opendir`。此时 `reaper`、`signal`、`monitor` 三个线程已经由 `start_threads` 启动。一次拉起网关或模型，失败输出会点名这四个调用。

模型侧现在在这个窗口里写入 `MORTRED_LISTEN_HOST=127.0.0.1` 和 `MORTRED_AUTH_TOKEN`（值是内部 token）。网关侧在 scrape token 非空时写入 `MORTRED_METRICS_TOKEN`。这两项注入要留下，只是不再发生在 `fork` 与 `exec` 之间。

不覆盖 P0-2、P1-8、D-8。不重开已关闭原句：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、P1-12、D-1 至 D-6、D-9、D-10、D-11。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`centerface` 的 engine 输入名 `input.1` 不重开。

## 做法

### 推荐

环境块和参数数组在父进程、`fork` 之前组装。子进程只保留 async-signal-safe 调用，并用 `execve` 带上这块环境。监督进程自己的环境不被 `spawn_locked` 修改。

父进程在创建完两对日志管道之后、`fork` 之前做完这些事：

- 用 `getrlimit(RLIMIT_NOFILE)` 得到 `max_fd`。`rlim_cur` 落在 3 到 1048576 时，`max_fd` 取这个值；失败、小于 3、或大于 1048576 时，`max_fd` 取 1024。
- 把现在子进程里算出的 `MORTRED_GATEWAY_PORT` 字符串提前算完。
- 从当时的 `environ` 复制一份环境块，再按下面的键覆盖或追加。空的可选键保持现状：现在子进程不 `setenv` 的键，环境块里也不要删掉继承来的同名项。
- 三个分支的 `argv` 仍是现在这三套：无配置参数时只有可执行名；`model` 非空时是可执行名、`--model`、model、配置路径；其余是可执行名加配置路径。网关可执行名仍是 `mortred-gateway.out`。
- `execve` 用的路径、`argv`、环境块、`chdir` 目录都先变成子进程可直接使用的指针。子进程不再调用 `c_str`、`empty`、`std::to_string`。

始终覆盖的键：`LD_LIBRARY_PATH`、`MORTRED_PROJECT_ROOT`、`MORTRED_CONTROL_CONFIG`。`LD_LIBRARY_PATH` 仍是项目 `lib_dir` 与 `libs_dir` 用冒号拼成的那一串，覆盖继承值。

网关再覆盖：`MORTRED_INTERNAL_TOKEN`、`MORTRED_GATEWAY_HOST`、`MORTRED_GATEWAY_PORT`。`MORTRED_METRICS_TOKEN`、`MORTRED_GATEWAY_AUTH_TOKEN`、`MORTRED_API_TOKEN` 仍只在对应字符串非空时写入。

模型再覆盖：`MORTRED_LISTEN_HOST` 的值固定是 `127.0.0.1`；`MORTRED_AUTH_TOKEN` 的值仍是内部 token。`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_MODEL_CONFIG_FILE` 的条件与现在相同。`MORTRED_MODEL` 仍只在 model 参数非空时写入，并与 `--model` 那套 `argv` 同时存在。

子进程在 `execve` 之前的顺序：把两根日志管道 `dup2` 到标准输出和标准错误，关掉四个管道端口，再对 `[3, max_fd)` 逐个 `close`；`prctl(PR_SET_PDEATHSIG, SIGINT)`，父进程已经消失则 `_exit(0)`；用 `sigaction` 把 `SIGINT` 和 `SIGQUIT` 设为 `SIG_DFL`；`sigemptyset` 之后 `sigprocmask` 清空掩码；`chdir` 到原来的 `bin_dir`，失败则 `_exit(127)`；`execve`，失败则 `_exit(127)`。

`signal` 不进入这个窗口。`sigaction` 的结构在子进程里用 `{}` 零初始化，再 `sigemptyset` 掩码、处理函数 `SIG_DFL`、flags 为 0。

从本文件删除 `close_all_fds_except_stdio`。子进程不再为关 fd 去读 `/proc/self/fd`。

子进程以及它在本文件里调用到的函数，调用名单只能是：`dup2`、`close`、`prctl`、`getppid`、`getpid`、`_exit`、`_Exit`、`sigemptyset`、`sigfillset`、`sigaddset`、`sigdelset`、`sigismember`、`sigprocmask`、`sigaction`、`chdir`、`execve`。`if`、`for`、`while`、`switch`、`return`、`sizeof`、`catch`、`alignof` 不算调用。窗口里必须有一次 `execve`。父进程日志行里的 `std::to_string(pid)` 留在子进程花括号之外。

代价：每次拉起在父进程复制一份环境块；子进程在 `exec` 前最多对 1048576 个 fd 调用一次 `close`，这个成本在拉起时发生一次。不新增常驻进程，不占显存，不生成 engine。产品端口仍是 gateway `:8080`、supervisor `:8787`。

### 否决

改成 `posix_spawn`。冻结的检查就是 `fork` 与 `exec` 之间的那一段；拿掉 `fork` 之后，这段窗口不存在，同一条检查无从对这一段点名。`PDEATHSIG` 也是这段里的 `prctl`。这条否决。

在 `spawn_locked` 里对监督进程调用 `setenv` / `putenv`，让子进程继承环境，同时子进程仍调用 `opendir`、`signal`、`std::to_string` 或 `execl`。监督进程自己的环境会被改掉，后拉起的子进程会看见上一次的键；窗口里的调用还在，同一条检查仍会点名。把检查改成不再点名这四个调用，也不是「同一条检查通过」。这条否决。

## 改动面

- `src/control/supervisor.cpp`：`spawn_locked` 的父进程在 `fork` 前组装环境块和 `argv`；子进程窗口改为上面的允许调用，并以 `execve` 结束。
- 同文件删除 `close_all_fds_except_stdio`。

## 明确不改

- `src/control/supervisor/supervisor_app.cpp` 在 `autostart` 之前对监督进程设置 `MORTRED_API_TOKEN` 和 `MORTRED_GATEWAY_CORS_ORIGINS`。这两处在 `fork` 之前。
- `src/control/gpu_sampler.h`、`src/control/occupancy_gate.h` 里对 `nvidia-smi` 的 `fork` / `execvp`。
- `src/control/cli/cli_app.cpp` 里的 `fork` / `execvp`。
- `src/control/supervisor.h`、catalog、就绪探测、重启与有序停机、一个 catalog id 一个 OS 进程。
- 模型进程读取 `MORTRED_LISTEN_HOST` 与 `MORTRED_AUTH_TOKEN` 的代码，网关读取 `MORTRED_METRICS_TOKEN` 的代码。
- 产品 toml、`type=mnn` / `type=tensorrt`、HTTP 与张量契约、gateway `:8080`、supervisor `:8787`。
- engine 仍在本机本卡用 `mortredctl prepare` 生成。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。
- 测试、脚本、镜像、部署文档、鉴权、限流、GPU 采样、默认镜像用户。
- P0-2、P1-8、D-8，以及已关闭原句。

## 契约

没有 HTTP 契约变化，没有张量契约变化，没有产品 toml 变化。

模型子进程的环境里仍有 `MORTRED_LISTEN_HOST=127.0.0.1` 和 `MORTRED_AUTH_TOKEN`。`MORTRED_AUTH_TOKEN` 的值仍是监督进程的内部 token；进程环境里若已有 `MORTRED_INTERNAL_TOKEN`，内部 token 仍取这个值。网关子进程在 scrape token 非空时仍有 `MORTRED_METRICS_TOKEN`。继承来的其他环境变量仍在，被上面列出的键覆盖。

`argv` 三套与现在相同。`PDEATHSIG` 仍是 `SIGINT`。`SIGINT` 与 `SIGQUIT` 在 `exec` 前仍是 `SIG_DFL`。信号掩码在 `exec` 前仍被清空。

## 性能与资源

每次拉起仍是一个子进程。父进程多一次环境块复制。子进程在 `exec` 前对 `[3, max_fd)` 调用 `close`，`max_fd` 不超过 1048576。不新增常驻进程，不占显存，不生成 engine。验收使用临时目录和临时端口，不占用产品端口 `:8080` 与 `:8787`。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

在 Linux 上、仓库根目录执行下面这一条。前置：`_bin/mortred-supervisor.out` 和 `python3` 已在。不需要 docker、权重、trtexec 或 GPU。临时根目录没有 pack，因此不会因 `nvidia-smi` 缺失而拒绝拉起。

这条命令就是那一条检查：它只扫描 `ProcessSupervisor::spawn_locked` 里 `fork` 之后的 `if (pid == 0)` 窗口，并跟进该窗口在 `src/control/supervisor.cpp` 内调用到的函数。窗口及其跟进函数里，凡是不在允许名单中的调用，各印一行 `fork-exec: ` 加名字后退出码 1，且不启动进程。`std::to_string` 印成 `fork-exec: std::to_string`。窗口里没有 `execve` 时另有一行 `fork-exec: missing execve`。`spawn_locked` 在去掉注释之后必须仍有字符串 `"MORTRED_LISTEN_HOST"`、`"127.0.0.1"`、`"MORTRED_AUTH_TOKEN"`、`"MORTRED_METRICS_TOKEN"`，缺一个就印 `fork-exec: missing` 加该字符串。子进程窗口若含有父进程日志里的 `spawn pid`，印 `fork-exec: missing`。

通过时退出码 0，标准错误为空，标准输出原文是下面三行。检查未通过时退出码 1，标准输出里能看见实际命中的 `fork-exec:` 行。

通过时的标准输出原文：

```
fork-exec: pass
listen=127.0.0.1
MORTRED_AUTH_TOKEN=d12-auth
```

命令：

```bash
bash <<'EOF'
set -euo pipefail
sup_pid=
tmp=
cleanup() {
  local rc=$?
  set +e
  if [[ -n "${sup_pid}" ]]; then
    kids=$(ps -o pid= --ppid "$sup_pid" 2>/dev/null || true)
    for k in $kids; do
      kill -KILL "$k" 2>/dev/null || true
    done
    kill -KILL "$sup_pid" 2>/dev/null || true
    wait "$sup_pid" 2>/dev/null || true
  fi
  if [[ -n "${tmp}" ]]; then
    rm -rf "$tmp"
  fi
  exit "$rc"
}
python3 - <<'PY'
import pathlib, sys
src = pathlib.Path("src/control/supervisor.cpp").read_text(encoding="utf-8")
ALLOW = {
    "dup2", "close", "prctl", "getppid", "getpid", "_exit", "_Exit",
    "sigemptyset", "sigfillset", "sigaddset", "sigdelset", "sigismember",
    "sigprocmask", "sigaction", "chdir", "execve",
}
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "alignof"}

def skip_noncode(text, i):
    n = len(text)
    if text.startswith("//", i):
        i += 2
        while i < n and text[i] != "\n":
            i += 1
        return i
    if text.startswith("/*", i):
        i += 2
        while i + 1 < n and not text.startswith("*/", i):
            i += 1
        return min(n, i + 2)
    if text[i] == '"':
        i += 1
        while i < n and text[i] != '"' and text[i] != "\n":
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            i += 1
        return min(n, i + 1)
    if text[i] == "'":
        i += 1
        while i < n and text[i] != "'" and text[i] != "\n":
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            i += 1
        return min(n, i + 1)
    return None

def extract_brace(text, open_idx):
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        jumped = skip_noncode(text, i)
        if jumped is not None:
            i = jumped
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
        i += 1
    return None

def scan_calls(text):
    names = []
    i = 0
    n = len(text)
    while i < n:
        jumped = skip_noncode(text, i)
        if jumped is not None:
            i = jumped
            continue
        c = text[i]
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            name = text[i:j]
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] == "(":
                names.append(name)
            i = j
            continue
        i += 1
    return names

def find_function_body(text, name):
    i = 0
    n = len(text)
    while i < n:
        jumped = skip_noncode(text, i)
        if jumped is not None:
            i = jumped
            continue
        c = text[i]
        if (c.isalpha() or c == "_") and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if text[i:j] == name:
                k = j
                while k < n and text[k] in " \t\r\n":
                    k += 1
                if k < n and text[k] == "(":
                    depth = 0
                    p = k
                    while p < n:
                        jumped = skip_noncode(text, p)
                        if jumped is not None:
                            p = jumped
                            continue
                        if text[p] == "(":
                            depth += 1
                        elif text[p] == ")":
                            depth -= 1
                            if depth == 0:
                                p += 1
                                break
                        p += 1
                    q = p
                    while q < n and text[q] in " \t\r\n":
                        q += 1
                    if text.startswith("const", q):
                        q += 5
                        while q < n and text[q] in " \t\r\n":
                            q += 1
                    if q < n and text[q] == "{":
                        return extract_brace(text, q)
            i = j
            continue
        i += 1
    return None

def strip_comments(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i) or text.startswith("/*", i):
            nxt = skip_noncode(text, i)
            i = n if nxt is None else nxt
            continue
        if text[i] == '"':
            nxt = skip_noncode(text, i)
            end = n if nxt is None else nxt
            out.append(text[i:end])
            i = end
            continue
        out.append(text[i])
        i += 1
    return "".join(out)

def show(name):
    return "std::to_string" if name == "to_string" else name

anchor = src.find("ProcessSupervisor::spawn_locked")
fn = None if anchor < 0 else find_function_body(src[anchor:], "spawn_locked")
fork_at = -1 if fn is None else fn.find("fork(")
pid_at = -1 if fork_at < 0 else fn.find("if (pid == 0)", fork_at)
brace_at = -1 if pid_at < 0 else fn.find("{", pid_at)
body = None if brace_at < 0 else extract_brace(fn, brace_at)
lines = []
if fn is None or body is None or "spawn pid" in body:
    lines.append("fork-exec: missing")
else:
    stripped = strip_comments(fn)
    for literal in (
        '"MORTRED_LISTEN_HOST"',
        '"127.0.0.1"',
        '"MORTRED_AUTH_TOKEN"',
        '"MORTRED_METRICS_TOKEN"',
    ):
        if literal not in stripped:
            lines.append("fork-exec: missing " + literal.strip('"'))
    direct = [n for n in scan_calls(body) if n not in KEYWORDS]
    if "execve" not in direct:
        lines.append("fork-exec: missing execve")
    bad = []
    seen = set()
    queue = [body]
    while queue:
        chunk = queue.pop()
        for name in scan_calls(chunk):
            if name in KEYWORDS:
                continue
            if name not in ALLOW:
                bad.append(show(name))
            if name not in seen:
                seen.add(name)
                found = find_function_body(src, name)
                if found is not None:
                    queue.append(found)
    lines.extend("fork-exec: " + item for item in bad)
if lines:
    for line in sorted(set(lines)):
        sys.stdout.write(line + "\n")
    sys.exit(1)
PY
tmp=$(mktemp -d)
trap cleanup EXIT
ports=$(python3 - <<'PY'
import socket
a = socket.socket(); a.bind(("127.0.0.1", 0))
b = socket.socket(); b.bind(("127.0.0.1", 0))
print(a.getsockname()[1])
print(b.getsockname()[1])
a.close(); b.close()
PY
)
api_port=$(printf '%s\n' "$ports" | sed -n '1p')
model_port=$(printf '%s\n' "$ports" | sed -n '2p')
mkdir -p "$tmp/conf/server/test" "$tmp/_bin" "$tmp/_lib" "$tmp/3rd_party/libs" "$tmp/logs"
cat > "$tmp/conf/mortred.toml" <<TOML
[supervisor]
api_host = "127.0.0.1"
api_port = ${api_port}
autostart_default = false
gpu_sample_interval_ms = 0
log_dir = "logs"
log_rotate_mb = 1
bin_dir = "_bin"
lib_dir = "_lib"
libs_dir = "3rd_party/libs"

[gateway]
host = "127.0.0.1"
port = 9
TOML
cat > "$tmp/conf/server/test/probe.toml" <<TOML
[PROBE_SERVER]
port=${model_port}
host="0.0.0.0"
server_uri="/mortred_ai_server_v1/test/probe"
server_exe="probe.out"
TOML
cat > "$tmp/_bin/probe.out" <<PY
#!/usr/bin/env python3
import os, socket, sys
host = os.environ["MORTRED_LISTEN_HOST"]
token = os.environ["MORTRED_AUTH_TOKEN"]
port = ${model_port}
body = ("listen=%s\nMORTRED_AUTH_TOKEN=%s\n" % (host, token)).encode()
hdr = ("HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body)).encode()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host, port))
s.listen(8)
while True:
    c, _ = s.accept()
    c.recv(4096)
    c.sendall(hdr + body)
    c.close()
PY
chmod 755 "$tmp/_bin/probe.out"
bin="$PWD/_bin/mortred-supervisor.out"
env -u MORTRED_AUTH_TOKEN -u MORTRED_LISTEN_HOST -u MORTRED_PACK \
  -u MORTRED_CONTROL_CONFIG -u MORTRED_PROFILE -u APP_BIN_DIR \
  -u APP_LIB_DIR -u APP_LIBS_DIR -u MORTRED_GATEWAY_HOST \
  -u MORTRED_GATEWAY_PORT -u MORTRED_EXPOSE -u MORTRED_METRICS_TOKEN \
  MORTRED_PROJECT_ROOT="$tmp" \
  MORTRED_API_TOKEN=d12-admin \
  MORTRED_API_HOST=127.0.0.1 \
  MORTRED_API_PORT="$api_port" \
  MORTRED_INTERNAL_TOKEN=d12-auth \
  MORTRED_AUTOSTART=false \
  "$bin" >"$tmp/sup.out" 2>"$tmp/sup.err" &
sup_pid=$!
ready=0
for _ in $(seq 1 50); do
  if python3 -c 'import sys,urllib.request
url=sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        ok=r.read()==b"{\"ok\":true}"
except Exception:
    ok=False
sys.exit(0 if ok else 1)' "http://127.0.0.1:${api_port}/api/v1/health"; then
    ready=1
    break
  fi
  sleep 0.2
done
[[ "$ready" == 1 ]]
python3 -c 'import sys,urllib.request
url=sys.argv[1]
req=urllib.request.Request(url, data=b"", method="POST", headers={"Authorization":"Bearer d12-admin","Content-Length":"0"})
with urllib.request.urlopen(req, timeout=15) as r:
    sys.exit(0 if r.status==200 and b"\"ok\":true" in r.read() else 1)' \
  "http://127.0.0.1:${api_port}/api/v1/servers/probe/start"
body_ok=0
for _ in $(seq 1 50); do
  if python3 -c 'import socket,sys
port=int(sys.argv[1])
s=socket.create_connection(("127.0.0.1", port), 2)
s.sendall(b"GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
data=b""
while True:
    chunk=s.recv(4096)
    if not chunk:
        break
    data+=chunk
s.close()
head,_,body=data.partition(b"\r\n\r\n")
sys.exit(0 if head.startswith(b"HTTP/1.1 200") and body==b"listen=127.0.0.1\nMORTRED_AUTH_TOKEN=d12-auth\n" else 1)' "$model_port"; then
    body_ok=1
    break
  fi
  sleep 0.2
done
[[ "$body_ok" == 1 ]]
printf '%s\n' 'fork-exec: pass' 'listen=127.0.0.1' 'MORTRED_AUTH_TOKEN=d12-auth'
EOF
```

## 测试义务

QA 只跑验收标准里的这一条命令。通过条件是退出码 0、标准输出与上面三行原文一致、标准错误为空。

不把读 `catalog()`、跑 `supervisor_unittest`、拉起产品 catalog 里的 id、生成 engine、启动产品端口上的 gateway 或 supervisor、覆盖 P0-2、P1-8、D-8 写成通过条件。

## 自测命令

与验收标准里的命令逐字相同。在 Linux 上、仓库根目录执行。期望退出码 0。标准错误为空。标准输出原文：

```
fork-exec: pass
listen=127.0.0.1
MORTRED_AUTH_TOKEN=d12-auth
```

```bash
bash <<'EOF'
set -euo pipefail
sup_pid=
tmp=
cleanup() {
  local rc=$?
  set +e
  if [[ -n "${sup_pid}" ]]; then
    kids=$(ps -o pid= --ppid "$sup_pid" 2>/dev/null || true)
    for k in $kids; do
      kill -KILL "$k" 2>/dev/null || true
    done
    kill -KILL "$sup_pid" 2>/dev/null || true
    wait "$sup_pid" 2>/dev/null || true
  fi
  if [[ -n "${tmp}" ]]; then
    rm -rf "$tmp"
  fi
  exit "$rc"
}
python3 - <<'PY'
import pathlib, sys
src = pathlib.Path("src/control/supervisor.cpp").read_text(encoding="utf-8")
ALLOW = {
    "dup2", "close", "prctl", "getppid", "getpid", "_exit", "_Exit",
    "sigemptyset", "sigfillset", "sigaddset", "sigdelset", "sigismember",
    "sigprocmask", "sigaction", "chdir", "execve",
}
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "alignof"}

def skip_noncode(text, i):
    n = len(text)
    if text.startswith("//", i):
        i += 2
        while i < n and text[i] != "\n":
            i += 1
        return i
    if text.startswith("/*", i):
        i += 2
        while i + 1 < n and not text.startswith("*/", i):
            i += 1
        return min(n, i + 2)
    if text[i] == '"':
        i += 1
        while i < n and text[i] != '"' and text[i] != "\n":
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            i += 1
        return min(n, i + 1)
    if text[i] == "'":
        i += 1
        while i < n and text[i] != "'" and text[i] != "\n":
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            i += 1
        return min(n, i + 1)
    return None

def extract_brace(text, open_idx):
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        jumped = skip_noncode(text, i)
        if jumped is not None:
            i = jumped
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
        i += 1
    return None

def scan_calls(text):
    names = []
    i = 0
    n = len(text)
    while i < n:
        jumped = skip_noncode(text, i)
        if jumped is not None:
            i = jumped
            continue
        c = text[i]
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            name = text[i:j]
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] == "(":
                names.append(name)
            i = j
            continue
        i += 1
    return names

def find_function_body(text, name):
    i = 0
    n = len(text)
    while i < n:
        jumped = skip_noncode(text, i)
        if jumped is not None:
            i = jumped
            continue
        c = text[i]
        if (c.isalpha() or c == "_") and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if text[i:j] == name:
                k = j
                while k < n and text[k] in " \t\r\n":
                    k += 1
                if k < n and text[k] == "(":
                    depth = 0
                    p = k
                    while p < n:
                        jumped = skip_noncode(text, p)
                        if jumped is not None:
                            p = jumped
                            continue
                        if text[p] == "(":
                            depth += 1
                        elif text[p] == ")":
                            depth -= 1
                            if depth == 0:
                                p += 1
                                break
                        p += 1
                    q = p
                    while q < n and text[q] in " \t\r\n":
                        q += 1
                    if text.startswith("const", q):
                        q += 5
                        while q < n and text[q] in " \t\r\n":
                            q += 1
                    if q < n and text[q] == "{":
                        return extract_brace(text, q)
            i = j
            continue
        i += 1
    return None

def strip_comments(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i) or text.startswith("/*", i):
            nxt = skip_noncode(text, i)
            i = n if nxt is None else nxt
            continue
        if text[i] == '"':
            nxt = skip_noncode(text, i)
            end = n if nxt is None else nxt
            out.append(text[i:end])
            i = end
            continue
        out.append(text[i])
        i += 1
    return "".join(out)

def show(name):
    return "std::to_string" if name == "to_string" else name

anchor = src.find("ProcessSupervisor::spawn_locked")
fn = None if anchor < 0 else find_function_body(src[anchor:], "spawn_locked")
fork_at = -1 if fn is None else fn.find("fork(")
pid_at = -1 if fork_at < 0 else fn.find("if (pid == 0)", fork_at)
brace_at = -1 if pid_at < 0 else fn.find("{", pid_at)
body = None if brace_at < 0 else extract_brace(fn, brace_at)
lines = []
if fn is None or body is None or "spawn pid" in body:
    lines.append("fork-exec: missing")
else:
    stripped = strip_comments(fn)
    for literal in (
        '"MORTRED_LISTEN_HOST"',
        '"127.0.0.1"',
        '"MORTRED_AUTH_TOKEN"',
        '"MORTRED_METRICS_TOKEN"',
    ):
        if literal not in stripped:
            lines.append("fork-exec: missing " + literal.strip('"'))
    direct = [n for n in scan_calls(body) if n not in KEYWORDS]
    if "execve" not in direct:
        lines.append("fork-exec: missing execve")
    bad = []
    seen = set()
    queue = [body]
    while queue:
        chunk = queue.pop()
        for name in scan_calls(chunk):
            if name in KEYWORDS:
                continue
            if name not in ALLOW:
                bad.append(show(name))
            if name not in seen:
                seen.add(name)
                found = find_function_body(src, name)
                if found is not None:
                    queue.append(found)
    lines.extend("fork-exec: " + item for item in bad)
if lines:
    for line in sorted(set(lines)):
        sys.stdout.write(line + "\n")
    sys.exit(1)
PY
tmp=$(mktemp -d)
trap cleanup EXIT
ports=$(python3 - <<'PY'
import socket
a = socket.socket(); a.bind(("127.0.0.1", 0))
b = socket.socket(); b.bind(("127.0.0.1", 0))
print(a.getsockname()[1])
print(b.getsockname()[1])
a.close(); b.close()
PY
)
api_port=$(printf '%s\n' "$ports" | sed -n '1p')
model_port=$(printf '%s\n' "$ports" | sed -n '2p')
mkdir -p "$tmp/conf/server/test" "$tmp/_bin" "$tmp/_lib" "$tmp/3rd_party/libs" "$tmp/logs"
cat > "$tmp/conf/mortred.toml" <<TOML
[supervisor]
api_host = "127.0.0.1"
api_port = ${api_port}
autostart_default = false
gpu_sample_interval_ms = 0
log_dir = "logs"
log_rotate_mb = 1
bin_dir = "_bin"
lib_dir = "_lib"
libs_dir = "3rd_party/libs"

[gateway]
host = "127.0.0.1"
port = 9
TOML
cat > "$tmp/conf/server/test/probe.toml" <<TOML
[PROBE_SERVER]
port=${model_port}
host="0.0.0.0"
server_uri="/mortred_ai_server_v1/test/probe"
server_exe="probe.out"
TOML
cat > "$tmp/_bin/probe.out" <<PY
#!/usr/bin/env python3
import os, socket, sys
host = os.environ["MORTRED_LISTEN_HOST"]
token = os.environ["MORTRED_AUTH_TOKEN"]
port = ${model_port}
body = ("listen=%s\nMORTRED_AUTH_TOKEN=%s\n" % (host, token)).encode()
hdr = ("HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body)).encode()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host, port))
s.listen(8)
while True:
    c, _ = s.accept()
    c.recv(4096)
    c.sendall(hdr + body)
    c.close()
PY
chmod 755 "$tmp/_bin/probe.out"
bin="$PWD/_bin/mortred-supervisor.out"
env -u MORTRED_AUTH_TOKEN -u MORTRED_LISTEN_HOST -u MORTRED_PACK \
  -u MORTRED_CONTROL_CONFIG -u MORTRED_PROFILE -u APP_BIN_DIR \
  -u APP_LIB_DIR -u APP_LIBS_DIR -u MORTRED_GATEWAY_HOST \
  -u MORTRED_GATEWAY_PORT -u MORTRED_EXPOSE -u MORTRED_METRICS_TOKEN \
  MORTRED_PROJECT_ROOT="$tmp" \
  MORTRED_API_TOKEN=d12-admin \
  MORTRED_API_HOST=127.0.0.1 \
  MORTRED_API_PORT="$api_port" \
  MORTRED_INTERNAL_TOKEN=d12-auth \
  MORTRED_AUTOSTART=false \
  "$bin" >"$tmp/sup.out" 2>"$tmp/sup.err" &
sup_pid=$!
ready=0
for _ in $(seq 1 50); do
  if python3 -c 'import sys,urllib.request
url=sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        ok=r.read()==b"{\"ok\":true}"
except Exception:
    ok=False
sys.exit(0 if ok else 1)' "http://127.0.0.1:${api_port}/api/v1/health"; then
    ready=1
    break
  fi
  sleep 0.2
done
[[ "$ready" == 1 ]]
python3 -c 'import sys,urllib.request
url=sys.argv[1]
req=urllib.request.Request(url, data=b"", method="POST", headers={"Authorization":"Bearer d12-admin","Content-Length":"0"})
with urllib.request.urlopen(req, timeout=15) as r:
    sys.exit(0 if r.status==200 and b"\"ok\":true" in r.read() else 1)' \
  "http://127.0.0.1:${api_port}/api/v1/servers/probe/start"
body_ok=0
for _ in $(seq 1 50); do
  if python3 -c 'import socket,sys
port=int(sys.argv[1])
s=socket.create_connection(("127.0.0.1", port), 2)
s.sendall(b"GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
data=b""
while True:
    chunk=s.recv(4096)
    if not chunk:
        break
    data+=chunk
s.close()
head,_,body=data.partition(b"\r\n\r\n")
sys.exit(0 if head.startswith(b"HTTP/1.1 200") and body==b"listen=127.0.0.1\nMORTRED_AUTH_TOKEN=d12-auth\n" else 1)' "$model_port"; then
    body_ok=1
    break
  fi
  sleep 0.2
done
[[ "$body_ok" == 1 ]]
printf '%s\n' 'fork-exec: pass' 'listen=127.0.0.1' 'MORTRED_AUTH_TOKEN=d12-auth'
EOF
```

## RD 不得自行决定

- 子进程窗口的允许名单。名单就是推荐做法里的那 16 个名字。不得把 `setenv`、`putenv`、`unsetenv`、`opendir`、`fdopendir`、`readdir`、`closedir`、`signal`、`std::to_string`、`execl`、`execle`、`execv`、`execvp`、`c_str`、`empty` 放进这个窗口，也不得放进窗口能调用到的本文件函数。
- `SIGINT` 与 `SIGQUIT` 回到 `SIG_DFL` 用 `sigaction`。不得因为手册把 `signal` 列为 async-signal-safe 而把 `signal` 留在窗口里。
- 关 fd 只用父进程事先算好的 `max_fd` 和子进程里的 `close`。`rlim_cur` 在 3 到 1048576 之间用该值，否则 `max_fd` 为 1024。不得改用 `opendir`、`close_range`、`closefrom`，也不得把上限改成别的数。Ubuntu 20.04 在支持范围内，没有可依赖的 `close_range`。
- 环境块在父进程、`fork` 之前从当时的 `environ` 复制再覆盖。`spawn_locked` 不得调用 `setenv`、`putenv`、`unsetenv`。不得只把注入的键交给子进程而丢掉继承的其余变量。
- 覆盖键和条件与现在一致。模型的 `MORTRED_LISTEN_HOST` 固定为 `127.0.0.1`，不得改成 toml 里的 `host`。`MORTRED_AUTH_TOKEN` 仍是内部 token。网关的 `MORTRED_METRICS_TOKEN` 仍只在 scrape token 非空时写入。
- `argv` 仍是现在的三套。`MORTRED_MODEL` 与 `--model` 仍同时出现。`chdir` 的目录仍是 `bin_dir`。`PDEATHSIG` 仍是 `SIGINT`。
- 子进程窗口留在 `src/control/supervisor.cpp` 的 `spawn_locked` 里。父进程那行 `spawn pid` 的 `std::to_string` 留在这个窗口外面。
- 不得改 `supervisor_app.cpp` 在拉起前对监督进程设置的 `MORTRED_API_TOKEN` 与 `MORTRED_GATEWAY_CORS_ORIGINS`，不得改 `gpu_sampler.h`、`occupancy_gate.h`、`cli_app.cpp` 里的 `fork`。
- 不得改产品 toml、HTTP 与张量契约、产品端口、测试、脚本、镜像、部署文档、鉴权、限流、GPU 采样或默认镜像用户。
- 不得把本条做成 P0-2、P1-8、D-8，或已关闭原句的重开。不得改验收命令来放过窗口里的调用。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 产品 | 同意 | 无 | v1 | `03-discussion/v1-sign-pm.md` |
| 研发 | 同意 | 无 | v1 | `03-discussion/v1-sign-rd.md` |
| 技术负责人 | 正文 | 无 | v1 | `03-discussion/v1-tech-lead.md` |
