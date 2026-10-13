# 技术方案

- 问题: P2-3
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-3

## 边界

锁定句照录：supervisor 在 `fork` 之后、`exec` 之前调用了不在 async-signal-safe 列表中的函数（ST5）。场景是一次启动 supervisor，拉起一个模型子进程，并让这次启动中的 `nvidia-smi` 探测跑完。现在拉起子进程时，`fork` 之后、`execve` 之前调用 `prctl(PR_SET_PDEATHSIG)`。探测 `nvidia-smi` 的子进程在 `fork` 之后调用 `execvp`。`prctl` 与 `execvp` 都不在 POSIX async-signal-safe 列表中。这段路径没有调用 `setenv`。改完后能观察到的变化：对这一次启动里 fork 出的子进程做一次库函数跟踪（ltrace 或等价工具）。通过条件是：`fork` 返回之后、随后的 `exec` 之前，跟踪里出现的库函数名没有 `prctl`、`execvp`、`setenv`。带绝对路径的 `execve` 可以作为随后那次 `exec` 本身出现，不算失败。只对照源码不算通过。同一次启动后模型子进程处于运行，`nvidia-smi` 探测有输出，或按现有规则标明失败。

本方案只做这三处调用：`src/control/supervisor.cpp` 里模型（以及同函数拉起的 gateway）子进程在 `fork` 返回之后的 `prctl`，以及 `src/control/occupancy_gate.h`、`src/control/gpu_sampler.h` 两处 `nvidia-smi` 子进程里的 `execvp`。父进程死亡时子进程仍收到 `SIGINT`。两处探测的查询参数、3 秒截止、超时 `SIGKILL` 和失败后的现有日志保持原样。

通过条件保持锁定句。窗口按下面的规则在一次外部验收里判定，不另加条件：

- 只看这次 supervisor 进程的直接子进程（跟踪行的 ppid 等于该 supervisor 的 pid）。模型进程在 `exec` 之后再 fork 出的进程不在窗口里。
- 某个直接子进程的跟踪里，第一条 `func=execve` 且路径以 `/` 开头的行是随后那次 `exec`。这一行不算失败。这一行之前的函数名不得是 `prctl`、`execvp`、`setenv`。
- `execvp` 没有这条例外。它出现在该行之前，或该子进程没有这样一条 `execve` 因而整段都算窗口时，判定失败。
- 跟踪里可以没有 `exec` 的函数名。没有这三个名字就不因此失败。有这三个名字就不能因为「没写出 exec 函数名」而放过。
- 父进程自己的 `setenv`（`supervisor_app.cpp` 在 `fork` 前写 `MORTRED_API_TOKEN` 和 `MORTRED_GATEWAY_CORS_ORIGINS`）不在子进程窗口里。`exec` 之后同一 pid 的后续调用不在窗口里。

附录里下面两句不是通过条件：只对照源码确认三处没有 `prctl` 和 `execvp`；把「nvidia-smi 改为带绝对路径的 `execve`」本身写成通过条件。列表阶段未选中的那一套也不写入通过条件：跟踪必须写出随后那次 `exec` 的函数名；探测只有在调用是带绝对路径的 `execve` 时才算通过；窗口只查 `prctl` 和 `setenv`；只去掉模型子进程里的 `prctl` 不算通过。

上一版审查意见无。不把 P2-4、P2-5、P2-6 拉进来。不改 TensorRT execution context，不改 `worker_nums`，不做动态 batch。不跑 calibrate，不跑 `http_infer_rps.py`，不跑 `mortredctl prepare`。

验收启动的模型是 catalog id `libface`（现有 `conf/server/object_detection/libface_det/libface_server_config.toml`，`worker_nums=1`，端口 9051）。不加载 pack，占用门闩不拦截这次拉起；`query_nvidia_smi()` 仍会在拉起前执行。gateway 沿现有 scrape 检查：本次验收不设置 metrics token，gateway 在 `fork` 之前失败，跟踪里可以没有 gateway 子进程。这不改变 token 规则，也不把 gateway 是否监听写进通过条件。

## 做法

### 推荐

子进程里父进程死亡信号仍在 `fork` 返回之后、`execve` 之前设置，信号仍是 `SIGINT`。调用改为 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)`，不再进入 glibc `prctl` 的取消点包装。跟踪里的库函数名是 `syscall`，不是 `prctl`。`getppid()` 与 `fork` 前记下的父 pid 不一致时仍 `_exit(0)`。`SIGINT` / `SIGQUIT` 恢复为 `SIG_DFL`、清空信号掩码、`chdir`、再 `execve` 现有绝对路径，这几步保留。Linux 在 `fork` 时清掉子进程的 `pdeath_signal`，所以这步留在子进程里。

两处 `nvidia-smi` 共用同一个父进程路径解析：在 `fork` 之前读取 `PATH`（未设置时用 `confstr(_CS_PATH)`），按 `execvp` 的顺序找第一个 `access(X_OK)` 成功的 `nvidia-smi`，相对目录先变成绝对路径。子进程只对这个绝对路径调用 `execve`，环境指针用当时的 `environ`，不在子进程里组环境。`argv` 的查询串保持现在的字节内容。找不到可执行文件时不 `fork`，`query_nvidia_smi` 仍返回 `available=false` 的空快照，采样器仍返回空串并打现在的 offline 日志。

代价：采样间隔仍是 `conf/mortred.toml` 的 `gpu_sample_interval_ms = 500`，每次探测在父进程多一次目录遍历。不新增常驻进程，不占显存，不生成 engine。`syscall` 与原来的 `prctl` 一样，每个被拉起的子进程一次。挂起的 `nvidia-smi` 仍由现有 poll、3 秒截止和 `SIGKILL` 收口。

### 否决

父进程在 `fork` 之前设置 `PDEATHSIG`，子进程里留下 `execvp`。内核在 `fork` 时把子进程这项标志清零，supervisor 死亡后模型进程收不到 `SIGINT`。只删 `prctl` 而留下 `execvp` 时，子进程仍在 `fork` 之后搜索 `PATH` 并可能分配内存；`execvp` 没有锁定句给绝对路径 `execve` 的那条例外，窗口判定失败。

改成 `posix_spawn` / `posix_spawnp`。POSIX 的 spawn 属性没有父进程死亡信号。要保住 `SIGINT`，子进程里仍要设置该信号，窗口还在。两处探测现在的 pipe、poll、超时 `SIGKILL` 和 `waitpid` 要整段换掉，挂起的 `nvidia-smi` 会拖住这次拉起。glibc 的 spawn 在子进程里仍会执行 `exec`。代价高于推荐，锁定句能看到的结果没有更好。

## 改动面

- `src/control/supervisor.cpp`：子进程里的 `::prctl(PR_SET_PDEATHSIG, SIGINT)` 改为 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)`。位置仍在 `fork` 返回之后、现有 `execve` 之前。`getppid` 竞态、`sigaction`、`sigprocmask`、`chdir`、`execve` 的参数保留。`fork` 之前组装环境块的 `override_env_entry` 保留。
- `src/control/occupancy_gate.h`：增加父进程绝对路径解析，供两处探测使用。`query_nvidia_smi` 的子进程改为 `execve` 该路径。查询串仍是 `--query-gpu=name,memory.total,memory.free` 和 `--format=csv,noheader,nounits`。超时、读上限、`SIGKILL`、`waitpid`、CSV 解析和 `available` 的赋值保留。
- `src/control/gpu_sampler.h`：`probe_nvidia_smi` 使用同一个解析函数，子进程改为 `execve`。查询串仍是现在的 name、utilization、memory、temperature、power、clocks、fan、pcie 那一长串，以及 `--format=csv,noheader,nounits`。3 秒截止和三句首次日志保留。
- `test/occupancy_gate_unittest.cc`：`query_nvidia_smi_times_out_on_hang` 增加耗时下限，见测试义务。假 `nvidia-smi` 仍放在 `PATH` 最前。

## 明确不改

- `conf/mortred.toml` 的 `autostart_default`、`gpu_sample_interval_ms`、`api_host`、`api_port`，以及 `[gateway]`、`[gateway.rate_limit]`。
- `conf/server/object_detection/libface_det/libface_server_config.toml` 的 `model`、`port`、`worker_nums`、`server_uri`、`server_exe`。
- `conf/model/object_detection/libfacedetection/libface.toml` 的 `type = "tensorrt"` 和 engine 路径。
- `src/control/supervisor/supervisor_app.cpp` 里 `fork` 之前的两处 `setenv`，以及 gpu sampler 的启动位置。
- `src/control/cli/cli_app.cpp` 的 `fork` / `execvp`。
- `src/control/catalog.h` 的 id、一个 id 一个进程、gateway 与模型分进程。
- 子进程里现有的 `dup2`、`close`、`open`、`getppid`、`sigemptyset`、`sigaction`、`sigprocmask`、`chdir`、`_exit`。关多余 fd 的循环保留。
- 环境块里已有的 `LD_LIBRARY_PATH`、`MORTRED_PROJECT_ROOT`、`MORTRED_LISTEN_HOST`、`MORTRED_AUTH_TOKEN`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_MODEL_CONFIG_FILE`。验收不设置 `MORTRED_WORKER_NUMS` 和 `MORTRED_DECODE_AUTO`，toml 里的 `worker_nums=1` 仍由现有逻辑在 `fork` 前写入环境块。
- TensorRT execution context、静态 batch=1、engine 文件、`docs/benchmark-h3/`。
- gateway 的 token 启动条件、限流、systemd 单元、停止信号顺序。
- 占用门闩在 pack 未激活时放行、以及 `nvidia-smi` 失败时 `available=false` 并跳过指纹比对的规则。

## 契约

管理 API 的路径和鉴权保持现状。`POST /api/v1/servers/libface/start` 成功仍是 HTTP 200 且 `ok` 为 true。`GET /api/v1/servers/libface` 的 `server.state` 为 `running` 时，现有就绪探测已经成功，`server.pid` 大于 0。`GET /api/v1/gpu` 的字段不增不改。本次验收不把该接口当作通过条件。

模型子进程的 argv 仍是可执行文件名、`--model`、`libface` 和现有 server toml 路径。映像替换仍是对绝对路径的 `execve`，环境块仍在 `fork` 之前写好。

两处 `nvidia-smi` 的查询参数字符串不变。`execve` 的文件参数改为父进程解析出的绝对路径，`argv` 的第一项用同一绝对路径。找不到可执行文件时不创建子进程。采样器首次结果仍是下面三句之一，原文保留：

- `mortred-supervisor: gpu sampler online (`
- `mortred-supervisor: gpu sampler offline (utilization N/A)`
- `mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`

前两句表示探测进程已经跑完并留下输出（利用率可读，或按现有规则把 `N/A` 标成 offline）。第三句是现有的失败标明，包括找不到可执行文件、超时和空输出。

父进程死亡信号仍是 `SIGINT`，仍在子进程里、`fork` 返回之后设置。

## 性能与资源

不生成 engine，不增加 execution context，不加 context 池，不改 `worker_nums`，不改 batch。常驻进程仍是这次 supervisor，加上它拉起的一个 `libface` 模型进程。gpu 采样仍是 supervisor 进程内的现有线程。`nvidia-smi` 仍是短命子进程，默认截止仍是 3000 毫秒。路径解析在父进程、每次探测一次，不另建常驻进程，不占显存。

验收不设置 `MORTRED_PACK`。`libface` 的产品 toml 保持 `worker_nums=1`。

## 验收标准

1. 打开仓库里的三个文件，同时满足：`conf/mortred.toml` 的 `[supervisor]` 含有 `autostart_default = false` 和 `gpu_sample_interval_ms = 500`；`conf/server/object_detection/libface_det/libface_server_config.toml` 含有 `model="libface"`、`port=9051`、`worker_nums=1`；`conf/model/object_detection/libfacedetection/libface.toml` 含有 `type = "tensorrt"`。本次验收不修改这三个文件，不设置 `MORTRED_PACK`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_PROFILE`、`MORTRED_METRICS_TOKEN`。

2. 同一次启动使用库函数跟踪。跟踪只记录 libc 的 `prctl`、`execvp`、`setenv`、`execve`。对每个 ppid 等于该 supervisor pid 的子进程：若存在第一条路径以 `/` 开头的 `execve`，则只检查该行之前的函数名；否则检查该子进程的全部跟踪行。这些函数名里没有 `prctl`、`execvp`、`setenv`。路径以 `/` 开头的那条 `execve` 不算失败。`GET /api/v1/servers/libface` 得到的 pid 必须在跟踪里作为这样的直接子进程出现。没有这行跟踪，不通过。

3. 同一次启动里，上述 `GET` 的 `server.state` 为 `running`，`server.pid` 大于 0，并且该 pid 当时仍存在。从发出 start 到读到这个状态，墙钟不超过 180 秒。

4. 同一次启动的 supervisor 标准错误出现第「契约」节三句之一。出现 `gpu sampler online (` 或 `gpu sampler offline (utilization N/A)` 时，跟踪里至少有一个直接子进程，其 `execve` 或被记下的路径以 `/nvidia-smi` 结尾，并且该子进程满足第 2 条。出现 `gpu sampler offline (no nvidia-smi / no GPU)` 时，不要求再有 nvidia-smi 子进程，也不要求采样数值。

父进程自己的 `setenv`、`exec` 之后同一 pid 的调用、以及 ppid 不是该 supervisor 的进程，不使第 2 条失败。

## 测试义务

QA 先跑白盒，再跑自测命令里的外部脚本。外部脚本覆盖验收标准 1 至 4。脚本必须真的启动 supervisor 并写入跟踪日志。日志里没有 `libface` 的 pid，不得判通过。不得用阅读 `supervisor.cpp`、`occupancy_gate.h`、`gpu_sampler.h` 代替这次跟踪。

白盒只动现有挂起用例：`query_nvidia_smi_times_out_on_hang` 仍把可执行假 `nvidia-smi` 放在 `PATH` 最前，`available` 仍为 false，耗时大于等于 250 毫秒且小于 2000 毫秒。立即返回的「没找到就不 fork」不能通过这个用例。其余占用门闩用例保持能通过。

不跑 `mortredctl prepare`、`mortredctl calibrate`、`http_infer_rps.py`。不拉起第二个模型。不把 gateway 是否监听、token 拒绝和停止顺序写进 P2-3 的结论。

跟踪器用 `LD_PRELOAD` 插入上述四个 libc 符号，这是锁定句里的等价工具。插入只在进入这些函数时用 `write` 记一行，符号在进程启动时解析好。不用 `strace`：`setenv` 没有单独的系统调用，而且 v2 的系统调用跟踪没有被选中。不用 `ltrace -f` 跟过 `exec`：ptrace 会留在模型进程上，这次启动里的模型可以因此到不了 `running`。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面每一条的期望退出码都是 0。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。不执行 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。

第一条编译 supervisor、模型进程和占用门闩白盒。

```bash
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target mortred-supervisor.out mortred-model-server.out occupancy_gate_unittest
```

第二条跑白盒。挂起用例必须出现在列表里，并且整个测试通过。

```bash
cd /mnt/g/Codex/mortred_model_server
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./_bin/occupancy_gate_unittest --gtest_list_tests > /tmp/p2-3-occ.txt
grep -F 'query_nvidia_smi_times_out_on_hang' /tmp/p2-3-occ.txt
./_bin/occupancy_gate_unittest
```

第三条覆盖验收标准 1 至 4。临时跟踪器写在 `/tmp`，结束时删除。supervisor 使用空闲管理端口。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import json, os, signal, socket, subprocess, threading, time, urllib.request
from pathlib import Path

ROOT = Path("/mnt/g/Codex/mortred_model_server")
LOG = Path("/tmp/p2-3-libtrace.log")
SO = Path("/tmp/p2-3-libtrace.so")
SRC = Path("/tmp/p2-3-libtrace.c")
BIN = ROOT / "_bin" / "mortred-supervisor.out"

def die(msg):
    raise SystemExit(msg)

text = (ROOT / "conf" / "mortred.toml").read_text(encoding="utf-8")
block = text.split("[supervisor]", 1)[1].split("\n[", 1)[0]
for line in ("autostart_default = false", "gpu_sample_interval_ms = 500"):
    if line not in block:
        die("mortred.toml missing " + line)
srv = (ROOT / "conf/server/object_detection/libface_det/libface_server_config.toml").read_text(encoding="utf-8")
for line in ('model="libface"', "port=9051", "worker_nums=1"):
    if line not in srv:
        die("libface server toml missing " + line)
model = (ROOT / "conf/model/object_detection/libfacedetection/libface.toml").read_text(encoding="utf-8")
if 'type = "tensorrt"' not in model:
    die("libface model toml missing tensorrt")

SRC.write_text(r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>
static int logfd = -1;
static int (*real_prctl)(int, unsigned long, unsigned long, unsigned long, unsigned long);
static int (*real_execvp)(const char *, char *const []);
static int (*real_execve)(const char *, char *const [], char *const []);
static int (*real_setenv)(const char *, const char *, int);
static void putu(char *b, int *n, unsigned v) {
    char t[16]; int k = 0;
    if (v == 0) t[k++] = '0';
    while (v) { t[k++] = '0' + (v % 10); v /= 10; }
    while (k) b[(*n)++] = t[--k];
}
static void wr(const char *func, const char *arg0) {
    char b[800]; int n = 0;
    if (logfd < 0) return;
    memcpy(b + n, "pid=", 4); n += 4; putu(b, &n, (unsigned)getpid());
    memcpy(b + n, " ppid=", 6); n += 6; putu(b, &n, (unsigned)getppid());
    memcpy(b + n, " func=", 6); n += 6;
    for (const char *p = func; *p && n < 760; ++p) b[n++] = *p;
    memcpy(b + n, " arg0=", 6); n += 6;
    if (!arg0) arg0 = "-";
    for (const char *p = arg0; *p && n < 790; ++p) b[n++] = *p;
    b[n++] = '\n';
    (void)write(logfd, b, (size_t)n);
}
__attribute__((constructor)) static void init(void) {
    real_prctl = dlsym(RTLD_NEXT, "prctl");
    real_execvp = dlsym(RTLD_NEXT, "execvp");
    real_execve = dlsym(RTLD_NEXT, "execve");
    real_setenv = dlsym(RTLD_NEXT, "setenv");
    logfd = open("/tmp/p2-3-libtrace.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
}
int prctl(int option, ...) {
    va_list ap; va_start(ap, option);
    unsigned long a2 = va_arg(ap, unsigned long);
    unsigned long a3 = va_arg(ap, unsigned long);
    unsigned long a4 = va_arg(ap, unsigned long);
    unsigned long a5 = va_arg(ap, unsigned long);
    va_end(ap);
    wr("prctl", "-");
    return real_prctl(option, a2, a3, a4, a5);
}
int execvp(const char *file, char *const argv[]) { wr("execvp", file); return real_execvp(file, argv); }
int execve(const char *path, char *const argv[], char *const envp[]) { wr("execve", path); return real_execve(path, argv, envp); }
int setenv(const char *name, const char *value, int overwrite) { wr("setenv", name); return real_setenv(name, value, overwrite); }
''', encoding="utf-8")
subprocess.check_call(["gcc", "-shared", "-fPIC", "-o", str(SO), str(SRC), "-ldl"])
if LOG.exists():
    LOG.unlink()

s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
env = os.environ.copy()
for key in (
    "MORTRED_PACK", "MORTRED_WORKER_NUMS", "MORTRED_DECODE_AUTO", "MORTRED_PROFILE",
    "MORTRED_METRICS_TOKEN", "MORTRED_GATEWAY_AUTH_TOKEN", "MORTRED_OCCUPANCY_ENFORCE",
    "MORTRED_MODEL_CONFIG_FILE", "MORTRED_RATE_LIMIT",
):
    env.pop(key, None)
env.update({
    "MORTRED_PROJECT_ROOT": str(ROOT),
    "MORTRED_API_TOKEN": "p2-3-admin",
    "MORTRED_API_HOST": "127.0.0.1",
    "MORTRED_API_PORT": str(port),
    "MORTRED_AUTOSTART": "false",
    "LD_PRELOAD": str(SO),
    "LD_LIBRARY_PATH": str(ROOT / "3rd_party" / "libs") + ":" + str(ROOT / "_lib")
        + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else ""),
})
proc = subprocess.Popen(
    [str(BIN)], cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    text=True, start_new_session=True)
err_buf = []
def drain():
    for line in proc.stderr:
        err_buf.append(line)
threading.Thread(target=drain, daemon=True).start()

def get(path, auth=True):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path))
    if auth:
        req.add_header("Authorization", "Bearer p2-3-admin")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")

try:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            status, _ = get("/api/v1/health", auth=False)
            if status == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        die("supervisor health timeout\n" + "".join(err_buf[-40:]))
    req = urllib.request.Request(
        "http://127.0.0.1:%d/api/v1/servers/libface/start" % port, data=b"", method="POST")
    req.add_header("Authorization", "Bearer p2-3-admin")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", "replace")
        if resp.status != 200 or json.loads(body).get("ok") is not True:
            die("start failed " + body)
    model_pid = None
    deadline = time.time() + 180
    while time.time() < deadline:
        status, body = get("/api/v1/servers/libface")
        server = json.loads(body).get("server", {})
        if server.get("state") == "running" and int(server.get("pid", -1)) > 0:
            model_pid = int(server["pid"])
            os.kill(model_pid, 0)
            break
        time.sleep(0.5)
    else:
        die("libface not running\n" + "".join(err_buf[-80:]))
    err = "".join(err_buf)
    marks = (
        "mortred-supervisor: gpu sampler online (",
        "mortred-supervisor: gpu sampler offline (utilization N/A)",
        "mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)",
    )
    hit = next((m for m in marks if m in err), None)
    if hit is None:
        die("nvidia-smi probe did not finish\n" + err[-4000:])
    rows = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        item = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                item[k] = v
        if "pid" in item and "ppid" in item and "func" in item:
            item["pid"] = int(item["pid"])
            item["ppid"] = int(item["ppid"])
            rows.append(item)
    kids = {}
    for row in rows:
        if row["ppid"] == proc.pid:
            kids.setdefault(row["pid"], []).append(row)
    if model_pid not in kids:
        die("trace missed libface pid %s" % model_pid)
    forbidden = {"prctl", "execvp", "setenv"}
    for pid, lines in kids.items():
        cut = None
        for i, row in enumerate(lines):
            if row["func"] == "execve" and row.get("arg0", "").startswith("/"):
                cut = i
                break
        window = lines if cut is None else lines[:cut]
        bad = [row["func"] for row in window if row["func"] in forbidden]
        if bad:
            die("pid %s window has %s" % (pid, ",".join(bad)))
    if hit != marks[2]:
        smi = [
            row for lines in kids.values() for row in lines
            if row.get("arg0", "").endswith("/nvidia-smi")
        ]
        if not smi:
            die("probe had output but trace has no nvidia-smi child")
finally:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    for path in (LOG, SO, SRC):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
PY
```

## RD 不得自行决定

- 父进程死亡信号仍是 `SIGINT`，仍在子进程里、`fork` 返回之后、`execve` 之前设置。不得移到 `fork` 之前，不得改信号号，不得删掉 `getppid` 与父 pid 不一致时的 `_exit(0)`。
- 子进程不得调用 libc 函数 `prctl`、`execvp`、`execlp`、`setenv`、`putenv`、`clearenv`。不得把 `syscall(SYS_prctl, ...)` 再包成一个名为 `prctl` 的调用。不得改用 `posix_spawn`。
- 两处 `nvidia-smi` 必须调用同一个父进程路径解析。查找顺序与现在的 `execvp` 一致，不得改成写死 `/usr/bin/nvidia-smi`。子进程替换映像只用 `execve`，文件参数是该绝对路径。环境用 `fork` 时已经可见的 `environ`，不得在子进程里组装。
- 两处查询参数字符串、3 秒截止、超时 `SIGKILL`、`waitpid`，以及采样器三句首次日志的原文，保持不变。找不到可执行文件时不 `fork`，失败结果仍是现在的空快照和 offline 日志。
- 模型子进程的 argv、`fork` 前的环境块字段、`worker_nums`、execution context、动态 batch、gateway token 规则、限流和 systemd 单元，不在本条里改。
- 自测通过只认上面的跟踪日志和同一次启动的状态。不得用对照源码作为通过。
