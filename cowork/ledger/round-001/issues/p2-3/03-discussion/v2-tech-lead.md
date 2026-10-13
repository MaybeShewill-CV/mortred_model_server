# 技术方案

- 问题: P2-3
- 版本: v2
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-3

## 边界

锁定句照录。场景是一次启动 supervisor，拉起一个模型子进程，并让这次启动中的 `nvidia-smi` 探测跑完。现在拉起子进程时，`fork` 之后、`execve` 之前调用 `prctl(PR_SET_PDEATHSIG)`。探测 `nvidia-smi` 的子进程在 `fork` 之后调用 `execvp`。`prctl` 与 `execvp` 都不在 POSIX async-signal-safe 列表中。这段路径没有调用 `setenv`。通过条件是：对这一次启动里 fork 出的子进程做一次库函数跟踪（ltrace 或等价工具）。`fork` 返回之后、随后的 `exec` 之前，跟踪里出现的库函数名没有 `prctl`、`execvp`、`setenv`。带绝对路径的 `execve` 可以作为随后那次 `exec` 本身出现，不算失败。只对照源码不算通过。同一次启动后模型子进程处于运行，`nvidia-smi` 探测有输出，或按现有规则标明失败。

本版不改写这段通过条件。附录里下面两句仍然不是通过条件：只对照源码确认三处没有 `prctl` 和 `execvp`；把「nvidia-smi 改为带绝对路径的 `execve`」本身写成通过条件。列表阶段未选中的那一套也不写入通过条件：跟踪必须写出随后那次 `exec` 的函数名；探测只有在调用是带绝对路径的 `execve` 时才算通过；窗口只查 `prctl` 和 `setenv`；只去掉模型子进程里的 `prctl` 不算通过。

v1 的产品路径与锁定句一致，自测没有把锁定句落到同一次启动里。模型子进程在 `fork` 之后先执行 `for (int fd = 3; fd < max_fd; ++fd) close(fd)`，然后才是 `prctl` 和 `execve`。v1 的跟踪器只在 constructor 里打开一次日志 fd，这两次调用发生时该 fd 已经关掉，`write` 失败，日志里没有 libface 的 pid。新映像若在 `exec` 之后再调用被跟踪的函数，脚本又会因为缺少那条绝对路径 `execve` 而把这些调用整段算进窗口。验收标准 4 还要求探测路径以 `/nvidia-smi` 结尾。

下面两条写入本版正文，用来判定锁定句，不另立通过条件：

- 同一次启动里，对 libface 这个直接子进程的跟踪要覆盖从 `fork` 返回到随后那次 `exec` 之前，该 pid 出现在这段记录中，库函数名没有 `prctl`、`execvp`、`setenv`，带绝对路径的 `execve` 只作为这段终点；只读到 `exec` 之后同一 pid 的调用不算通过。本次 `nvidia-smi` 探测有输出时，那个直接子进程用同一段判定，路径不必以 `/nvidia-smi` 结尾。
- 模型子进程仍先关掉 fd 3 及以上，随后同一次启动的跟踪日志新增一行：pid 为该 libface 子进程，ppid 为 supervisor，`func=execve`，arg0 以 `/` 开头。

窗口按这次外部验收判定：

- 只看这次 supervisor 的直接子进程，跟踪行的 ppid 等于该 supervisor 的 pid。模型进程在 `exec` 之后再 fork 出的进程不在窗口里。
- libface 这个 pid 必须出现在它自己的终点行上。终点行是该 pid 第一条 `func=execve` 且 arg0 以 `/` 开头的记录。这行只充当从 `fork` 返回到随后那次 `exec` 的终点，本身不算失败。这行之前的函数名里没有 `prctl`、`execvp`、`setenv`。
- 没有这行终点时，同一 pid 上其余调用整段都不通过。其中只有 `exec` 之后才出现的调用，同样不通过。
- `execvp` 没有终点资格。它出现在终点之前，这一段失败。
- 跟踪挂钩在 `fork` 之前已经装进 supervisor，子进程在 `exec` 之前仍是这块地址空间。`prctl`、`execvp`、`setenv` 若在关掉 fd 3 及以上之后、`execve` 之前被调用，仍会先于终点行写入。终点之前没有这三项，表示这段被跟踪到了。
- 父进程自己的 `setenv` 不在子进程窗口里。终点行之后同一 pid 的调用不在这段里。
- 标准错误出现 `gpu sampler online (` 或 `gpu sampler offline (utilization N/A)` 时，探测已经留下输出。那个直接子进程用上面同一段判定，arg0 只要以 `/` 开头。出现 `gpu sampler offline (no nvidia-smi / no GPU)` 时，这是现有的失败标明，不要求再出现一个探测子进程。

本方案只处理这三处调用：`src/control/supervisor.cpp` 里模型以及同函数拉起的 gateway 子进程在 `fork` 返回之后的 `prctl`，以及 `src/control/occupancy_gate.h`、`src/control/gpu_sampler.h` 两处 `nvidia-smi` 子进程里的 `execvp`。父进程死亡时子进程仍收到 `SIGINT`。两处查询参数保持原样。`query_nvidia_smi` 的 3 秒截止、超时 `SIGKILL` 和失败后的空快照保持原样。采样器的 3 秒读截止、截止后的阻塞 `waitpid` 和三句首次日志保持原样。

不把 P2-4、P2-5、P2-6 拉进来。不改 TensorRT execution context，不改 `worker_nums`，不做动态 batch。不跑 calibrate，不跑 `http_infer_rps.py`，不跑 `mortredctl prepare`。

验收启动的模型是 catalog id `libface`（现有 `conf/server/object_detection/libface_det/libface_server_config.toml`，`worker_nums=1`，端口 9051）。不加载 pack。`query_nvidia_smi()` 仍在拉起前执行；pack 未激活时占用门闩不拦截这次拉起。gateway 沿现有 scrape 检查：本次验收不设置 metrics token，gateway 在 `fork` 之前失败，跟踪里可以没有 gateway 子进程。这不改变 token 规则，也不把 gateway 是否监听写进通过条件。

## 做法

### 推荐

子进程里，父进程死亡信号仍在关掉 fd 3 及以上之后、`execve` 之前设置，信号仍是 `SIGINT`。调用改为 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)`。跟踪里因此不会出现 libc 函数名 `prctl`。`getppid()` 与 `fork` 前记下的父 pid 不一致时仍 `_exit(0)`。`SIGINT` 与 `SIGQUIT` 恢复为 `SIG_DFL`、清空信号掩码、`chdir`、再 `execve` 现有绝对路径，这几步保留。Linux 在 `fork` 时清掉子进程的 `pdeath_signal`，所以这步留在子进程里。

两处 `nvidia-smi` 共用同一个父进程路径解析，并且在 `fork` 之前做完。读取 `PATH`。`PATH` 未设置时，默认目录与本机 glibc `execvp` 相同。查找顺序与该 `execvp` 相同：空的 `PATH` 字段表示当前目录，某个候选 `access(X_OK)` 失败时继续下一项，取第一个成功项。相对路径在 `fork` 之前变成以 `/` 开头的绝对路径。结果放在调用方自己的字符串里，采样线程和拉起线程可以同时解析。子进程只对这个绝对路径调用 `execve`，环境指针用当时的 `environ`，不在子进程里组环境。`argv` 的查询串保持现在的字节内容，`argv` 第一项用同一绝对路径。找不到可执行文件时不 `fork`。`query_nvidia_smi` 仍返回 `available=false` 的空快照。采样器仍返回空串并打现在的 offline 日志。

代价：采样间隔仍是 `conf/mortred.toml` 的 `gpu_sample_interval_ms = 500`，每次探测在父进程多一次目录遍历。不新增常驻进程，不占显存，不生成 engine。`syscall` 每个被拉起的子进程一次。`query_nvidia_smi` 挂起时仍由现有 poll、3 秒截止和 `SIGKILL` 收口。采样器挂起时仍停在现有的读截止和阻塞 `waitpid` 上。

模型子进程里现有的关 fd 循环留在原处，所以跟踪用的启动 fd 会在 `execve` 之前被关掉。自测跟踪器每次记录都重新打开日志文件，因此这行 `execve` 仍能在关 fd 之后写入。

### 否决

父进程在 `fork` 之前设置 `PDEATHSIG`，子进程里留下 `execvp`。内核在 `fork` 时把子进程这项标志清零，supervisor 死亡后模型进程收不到 `SIGINT`。留下 `execvp` 时，子进程仍在 `fork` 之后搜索 `PATH`。`execvp` 这个函数名落在窗口里，这一段失败。

改成 `posix_spawn` 或 `posix_spawnp`。POSIX 的 spawn 属性没有父进程死亡信号。要保住 `SIGINT`，子进程里仍要设置该信号，窗口还在。两处探测现在的 pipe、poll、超时处理和 `waitpid` 要整段换掉。glibc 的 spawn 在子进程里仍会执行 `exec`。代价高于推荐，锁定句能看到的结果没有更好。

## 改动面

- `src/control/supervisor.cpp`：子进程里的 `::prctl(PR_SET_PDEATHSIG, SIGINT)` 改为 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)`。位置仍在关 fd 循环之后、现有 `execve` 之前。`getppid` 竞态、`sigaction`、`sigprocmask`、`chdir`、`execve` 的参数保留。关 fd 的循环保留在 `dup2` 和管道 fd 关闭之后。`fork` 之前组装环境块的 `override_env_entry` 保留。
- `src/control/occupancy_gate.h`：增加父进程绝对路径解析，供两处探测使用。`query_nvidia_smi` 的子进程改为 `execve` 该路径。查询串仍是 `--query-gpu=name,memory.total,memory.free` 和 `--format=csv,noheader,nounits`。超时、读上限、`SIGKILL`、`waitpid`、CSV 解析和 `available` 的赋值保留。
- `src/control/gpu_sampler.h`：`probe_nvidia_smi` 使用同一个解析函数，子进程改为 `execve`。查询串仍是现在的 name、utilization、memory、temperature、power、clocks、fan、pcie 那一长串，以及 `--format=csv,noheader,nounits`。3 秒读截止、阻塞 `waitpid` 和三句首次日志保留。
- `test/occupancy_gate_unittest.cc`：`query_nvidia_smi_times_out_on_hang` 增加耗时下限，见测试义务。假 `nvidia-smi` 仍放在 `PATH` 最前。

## 明确不改

- 模型子进程里 `for (int fd = 3; fd < max_fd; ++fd) close(fd)` 的位置和范围。它仍在标准输出、标准错误 `dup2` 以及管道 fd 关闭之后，仍在父进程死亡信号和 `execve` 之前。
- 关 fd 之后、`execve` 之前现有的 `syscall`、`getppid`、`sigaction`、`sigprocmask`、`chdir`。这段不再 `open` 新的 fd。
- `conf/mortred.toml` 的 `autostart_default`、`gpu_sample_interval_ms`、`api_host`、`api_port`，以及 `[gateway]`、`[gateway.rate_limit]`。
- `conf/server/object_detection/libface_det/libface_server_config.toml` 的 `model`、`port`、`worker_nums`、`server_uri`、`server_exe`。
- `conf/model/object_detection/libfacedetection/libface.toml` 的 `type = "tensorrt"` 和 engine 路径。
- `src/control/supervisor/supervisor_app.cpp` 里 `fork` 之前的两处 `setenv`，以及 gpu sampler 的启动位置。
- `src/control/cli/cli_app.cpp` 的 `fork` 与 `execvp`。
- `src/control/catalog.h` 的 id、一个 id 一个进程、gateway 与模型分进程。
- 子进程里现有的 `dup2`、`close`、`open`、`getppid`、`sigemptyset`、`sigaction`、`sigprocmask`、`chdir`、`_exit`。
- 环境块里已有的 `LD_LIBRARY_PATH`、`MORTRED_PROJECT_ROOT`、`MORTRED_LISTEN_HOST`、`MORTRED_AUTH_TOKEN`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_MODEL_CONFIG_FILE`。验收不设置 `MORTRED_WORKER_NUMS` 和 `MORTRED_DECODE_AUTO`，toml 里的 `worker_nums=1` 仍由现有逻辑在 `fork` 前写入环境块。
- `query_nvidia_smi` 的 `SIGKILL`。`probe_nvidia_smi` 在读截止之后仍是现在的阻塞 `waitpid`，本条不加 `SIGKILL`。
- TensorRT execution context、静态 batch=1、engine 文件、`docs/benchmark-h3/`。
- gateway 的 token 启动条件、限流、systemd 单元、停止信号顺序。
- 占用门闩在 pack 未激活时放行，以及 `nvidia-smi` 失败时 `available=false` 并跳过指纹比对的规则。

## 契约

管理 API 的路径和鉴权保持现状。`POST /api/v1/servers/libface/start` 成功仍是 HTTP 200 且 `ok` 为 true。`GET /api/v1/servers/libface` 的 `server.state` 为 `running` 时，现有就绪探测已经成功，`server.pid` 大于 0。`GET /api/v1/gpu` 的字段不增不改。本次验收不把该接口当作通过条件。

模型子进程的 argv 仍是可执行文件名、`--model`、`libface` 和现有 server toml 路径。映像替换仍是对绝对路径的 `execve`。跟踪行里的 arg0 指这次 `execve` 的文件参数，就是这个绝对路径，所以以 `/` 开头。argv 第 0 项仍是 `mortred-model-server.out`。环境块仍在 `fork` 之前写好。

两处 `nvidia-smi` 的查询参数字符串不变。`execve` 的文件参数改为父进程解析出的绝对路径，`argv` 的第一项用同一绝对路径。找不到可执行文件时不创建子进程。采样器首次结果仍是下面三句之一，原文保留：

- `mortred-supervisor: gpu sampler online (`
- `mortred-supervisor: gpu sampler offline (utilization N/A)`
- `mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`

前两句表示探测进程已经跑完并留下输出（利用率可读，或按现有规则把 `N/A` 标成 offline）。第三句是现有的失败标明，包括找不到可执行文件、超时和空输出。

父进程死亡信号仍是 `SIGINT`，仍在子进程里、关 fd 循环之后、`execve` 之前设置。

## 性能与资源

不生成 engine，不增加 execution context，不加 context 池，不改 `worker_nums`，不改 batch。常驻进程仍是这次 supervisor，加上它拉起的一个 `libface` 模型进程。gpu 采样仍是 supervisor 进程内的现有线程。`nvidia-smi` 仍是短命子进程。`query_nvidia_smi` 的默认截止仍是 3000 毫秒，超时仍 `SIGKILL`。采样器的读截止仍是 3000 毫秒。路径解析在父进程、每次探测一次，不另建常驻进程，不占显存。

验收不设置 `MORTRED_PACK`。`libface` 的产品 toml 保持 `worker_nums=1`。

## 验收标准

1. 打开仓库里的三个文件，同时满足：`conf/mortred.toml` 的 `[supervisor]` 含有 `autostart_default = false` 和 `gpu_sample_interval_ms = 500`；`conf/server/object_detection/libface_det/libface_server_config.toml` 含有 `model="libface"`、`port=9051`、`worker_nums=1`；`conf/model/object_detection/libfacedetection/libface.toml` 含有 `type = "tensorrt"`。本次验收不修改这三个文件，不设置 `MORTRED_PACK`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_PROFILE`、`MORTRED_METRICS_TOKEN`。

2. 同一次启动使用库函数跟踪，并且跟踪真的跑了。日志里有一行 `pid` 等于该 supervisor、`func=init`、`watchfd` 大于等于 3。没有这行，不通过。挂钩的 libc 符号只有 `prctl`、`execvp`、`setenv`、`execve`。每次记录都重新打开日志文件；`execve` 钩子先检查启动时打开、并被子进程继承的那个 fd，再写终点行，然后才调用原来的 `execve`。对每个 ppid 等于该 supervisor pid 的子进程：必须有一条 `func=execve` 且 arg0 以 `/` 开头的终点行；这行之前的函数名里没有 `prctl`、`execvp`、`setenv`；没有这行终点时，该 pid 的其余行整段不通过。`GET /api/v1/servers/libface` 得到的 pid 必须作为这样的直接子进程出现，并另有一行同时满足：`pid` 为该值，`ppid` 为该 supervisor，`func=execve`，arg0 以 `/` 开头，`inh=closed`。`inh=closed` 表示进入这次 `execve` 时，上面那个编号不小于 3 的继承 fd 已经是 `EBADF`，也就是该模型子进程已经先关掉 fd 3 及以上，而这行是关 fd 之后才追加的。这个 pid 若只出现在 `exec` 之后的调用里，不通过。

3. 同一次启动里，上述 `GET` 的 `server.state` 为 `running`，`server.pid` 大于 0，并且该 pid 当时仍存在。从发出 start 到读到这个状态，墙钟不超过 180 秒。

4. 同一次启动的 supervisor 标准错误，在读到 `running` 之后 15 秒内出现第「契约」节三句之一。出现 `gpu sampler online (` 或 `gpu sampler offline (utilization N/A)` 时，日志里至少还有一个直接子进程，pid 不是 libface，并且它已经按第 2 条的同一段规则通过：有绝对路径 `execve` 终点，终点之前没有 `prctl`、`execvp`、`setenv`。这条路径不必以 `/nvidia-smi` 结尾，验收不读取这个后缀。出现 `gpu sampler offline (no nvidia-smi / no GPU)` 时，不要求再有探测子进程，也不要求采样数值；已经出现在日志里的直接子进程仍按第 2 条判定。

父进程自己的 `setenv`、终点行之后同一 pid 的调用，以及 ppid 不是该 supervisor 的进程，不进入第 2 条的那段。

## 测试义务

QA 先跑白盒，再跑自测命令里的外部脚本。外部脚本覆盖验收标准 1 至 4。脚本必须真的启动 supervisor，库函数跟踪必须真的跑：日志里要有 supervisor 自己的 `func=init` 行。日志里没有 libface 的终点行，或该行的 `inh` 不是 `closed`，不得判通过。不得用阅读 `supervisor.cpp`、`occupancy_gate.h`、`gpu_sampler.h` 代替这次跟踪。

白盒只动现有挂起用例：`query_nvidia_smi_times_out_on_hang` 仍把可执行假 `nvidia-smi` 放在 `PATH` 最前。这个假文件仍是睡眠 60 秒的 shell 脚本。`available` 仍为 false。耗时大于等于 250 毫秒且小于 2000 毫秒。立即返回的「没找到就不 fork」不能通过这个用例。路径解析必须能执行这个脚本，并让现有的 300 毫秒截止和 `SIGKILL` 生效。其余占用门闩用例保持能通过。

不跑 `mortredctl prepare`、`mortredctl calibrate`、`http_infer_rps.py`。不拉起第二个模型。不把 gateway 是否监听、token 拒绝和停止顺序写进 P2-3 的结论。

跟踪器用 `LD_PRELOAD` 插入上述四个 libc 符号，这是锁定句里的等价工具。符号在进程启动时用 `dlsym` 解析好。记录行在调用点用 `open`、`write`、`close` 写下，不依赖 constructor 里留下的那个 fd。`execve` 钩子在调用原来的 `execve` 之前写下终点行，所以模型子进程关掉 fd 3 及以上之后，这行仍然进入 `/tmp/p2-3-libtrace.log`。不用 `strace`：`setenv` 没有单独的系统调用。不用 `ltrace -f` 跟过 `exec`：ptrace 会留在模型进程上，这次启动里的模型可以因此到不了 `running`。`syscall` 这个名字不在三个禁止名里。日志里没有 `syscall` 不影响通过，日志里出现 `syscall` 也不作为失败。

## 自测命令

在本机 WSL Ubuntu 执行，仓库为 `/mnt/g/Codex/mortred_model_server`。下面每一条的期望退出码都是 0。不设置 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。不执行 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。

第一条编译 supervisor、模型进程和占用门闩白盒。

```bash
set -euo pipefail
cd /mnt/g/Codex/mortred_model_server
cmake --build build/full --target mortred-supervisor.out mortred-model-server.out occupancy_gate_unittest
```

第二条跑白盒。挂起用例必须出现在列表里，并且整个测试通过。

```bash
set -euo pipefail
cd /mnt/g/Codex/mortred_model_server
export LD_LIBRARY_PATH="/mnt/g/Codex/mortred_model_server/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./_bin/occupancy_gate_unittest --gtest_list_tests > /tmp/p2-3-occ.txt
grep -F 'query_nvidia_smi_times_out_on_hang' /tmp/p2-3-occ.txt
./_bin/occupancy_gate_unittest
```

第三条覆盖验收标准 1 至 4。临时跟踪器写在 `/tmp`，结束时删除。supervisor 使用空闲管理端口。库函数跟踪在这条命令里真的编译并装进同一次启动。

```bash
set -euo pipefail
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
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
static int logfd = -1;
static int (*real_prctl)(int, unsigned long, unsigned long, unsigned long, unsigned long);
static int (*real_execvp)(const char *, char *const []);
static int (*real_execve)(const char *, char *const [], char *const []);
static int (*real_setenv)(const char *, const char *, int);
static void putu(char *b, int *n, unsigned v) {
    char t[16]; int k = 0;
    if (v == 0) t[k++] = '0';
    while (v) { t[k++] = (char)('0' + (v % 10)); v /= 10; }
    while (k) b[(*n)++] = t[--k];
}
static void wr(const char *func, const char *arg0, const char *inh) {
    char b[900]; int n = 0;
    int fd = open("/tmp/p2-3-libtrace.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    memcpy(b + n, "pid=", 4); n += 4; putu(b, &n, (unsigned)getpid());
    memcpy(b + n, " ppid=", 6); n += 6; putu(b, &n, (unsigned)getppid());
    memcpy(b + n, " func=", 6); n += 6;
    for (const char *p = func; *p && n < 700; ++p) b[n++] = *p;
    memcpy(b + n, " arg0=", 6); n += 6;
    if (!arg0) arg0 = "-";
    for (const char *p = arg0; *p && n < 820; ++p) b[n++] = *p;
    memcpy(b + n, " inh=", 5); n += 5;
    if (!inh) inh = "na";
    for (const char *p = inh; *p && n < 850; ++p) b[n++] = *p;
    memcpy(b + n, " watchfd=", 9); n += 9;
    if (logfd < 0) { memcpy(b + n, "-1", 2); n += 2; }
    else putu(b, &n, (unsigned)logfd);
    b[n++] = '\n';
    (void)write(fd, b, (size_t)n);
    (void)close(fd);
}
__attribute__((constructor)) static void init(void) {
    real_prctl = dlsym(RTLD_NEXT, "prctl");
    real_execvp = dlsym(RTLD_NEXT, "execvp");
    real_execve = dlsym(RTLD_NEXT, "execve");
    real_setenv = dlsym(RTLD_NEXT, "setenv");
    logfd = open("/tmp/p2-3-libtrace.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    wr("init", "-", "open");
}
int prctl(int option, ...) {
    va_list ap; va_start(ap, option);
    unsigned long a2 = va_arg(ap, unsigned long);
    unsigned long a3 = va_arg(ap, unsigned long);
    unsigned long a4 = va_arg(ap, unsigned long);
    unsigned long a5 = va_arg(ap, unsigned long);
    va_end(ap);
    wr("prctl", "-", "na");
    return real_prctl(option, a2, a3, a4, a5);
}
int execvp(const char *file, char *const argv[]) {
    wr("execvp", file, "na");
    return real_execvp(file, argv);
}
int execve(const char *path, char *const argv[], char *const envp[]) {
    const char *inh = "open";
    if (logfd >= 3 && fcntl(logfd, F_GETFD) < 0 && errno == EBADF) inh = "closed";
    wr("execve", path, inh);
    return real_execve(path, argv, envp);
}
int setenv(const char *name, const char *value, int overwrite) {
    wr("setenv", name, "na");
    return real_setenv(name, value, overwrite);
}
''', encoding="utf-8")

proc = None
try:
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
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/api/v1/servers/libface/start" % port, data=b"", method="POST")
    req.add_header("Authorization", "Bearer p2-3-admin")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", "replace")
        if resp.status != 200 or json.loads(body).get("ok") is not True:
            die("start failed " + body)
    model_pid = None
    deadline = t0 + 180
    while time.time() < deadline:
        status, body = get("/api/v1/servers/libface")
        server = json.loads(body).get("server", {})
        if server.get("state") == "running" and int(server.get("pid", -1)) > 0:
            model_pid = int(server["pid"])
            try:
                os.kill(model_pid, 0)
            except OSError:
                time.sleep(0.5)
                continue
            break
        time.sleep(0.5)
    else:
        die("libface not running within 180s\n" + "".join(err_buf[-80:]))
    marks = (
        "mortred-supervisor: gpu sampler online (",
        "mortred-supervisor: gpu sampler offline (utilization N/A)",
        "mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)",
    )
    hit = None
    mark_deadline = time.time() + 15
    while time.time() < mark_deadline:
        err = "".join(err_buf)
        hit = next((m for m in marks if m in err), None)
        if hit is not None:
            break
        time.sleep(0.2)
    if hit is None:
        die("nvidia-smi probe did not finish\n" + "".join(err_buf)[-4000:])
    if not LOG.exists():
        die("trace log missing; library tracer did not run")
    rows = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        item = {}
        for part in line.split():
            if "=" in part:
                k, v = part.split("=", 1)
                item[k] = v
        if "pid" in item and "ppid" in item and "func" in item:
            item["pid"] = int(item["pid"])
            item["ppid"] = int(item["ppid"])
            rows.append(item)
    inits = [r for r in rows if r["pid"] == proc.pid and r["func"] == "init"]
    if not inits or "watchfd" not in inits[0]:
        die("library tracer did not run in supervisor")
    if int(inits[0]["watchfd"]) < 3:
        die("tracer watchfd is below 3")
    kids = {}
    for row in rows:
        if row["ppid"] == proc.pid:
            kids.setdefault(row["pid"], []).append(row)
    if model_pid not in kids:
        die("trace missed libface pid %s; post-exec calls alone do not pass" % model_pid)
    forbidden = {"prctl", "execvp", "setenv"}
    cuts = {}
    for pid, lines in kids.items():
        cut = None
        for i, row in enumerate(lines):
            if row["func"] == "execve" and row.get("arg0", "").startswith("/"):
                cut = i
                break
        if cut is None:
            die("pid %s has no absolute execve endpoint; calls after exec do not pass" % pid)
        bad = [row["func"] for row in lines[:cut] if row["func"] in forbidden]
        if bad:
            die("pid %s window has %s" % (pid, ",".join(bad)))
        cuts[pid] = cut
    end = kids[model_pid][cuts[model_pid]]
    if end.get("ppid") != proc.pid or end.get("func") != "execve":
        die("libface endpoint row mismatch")
    if not str(end.get("arg0", "")).startswith("/"):
        die("libface execve arg0 is not absolute")
    if end.get("inh") != "closed":
        die("libface execve logged while fd 3+ was still open")
    if hit != marks[2]:
        others = [pid for pid in kids if pid != model_pid]
        if not others:
            die("probe had output but its direct child is absent")
finally:
    if proc is not None:
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

- 父进程死亡信号仍是 `SIGINT`，仍在子进程里、关 fd 循环之后、`execve` 之前设置。不得移到 `fork` 之前，不得改信号号，不得删掉 `getppid` 与父 pid 不一致时的 `_exit(0)`。
- 模型子进程仍先关掉 fd 3 及以上，循环的起点、上界和位置保持现在这样。不得删掉，不得移到 `execve` 之后，也不得移到父进程死亡信号之后。关 fd 之后、`execve` 之前不得再 `open`。
- 子进程不得调用 libc 函数 `prctl`、`execvp`、`execlp`、`setenv`、`putenv`、`clearenv`。不得把 `syscall(SYS_prctl, ...)` 再包成一个名为 `prctl` 的调用。不得改用 `posix_spawn`。
- 两处 `nvidia-smi` 必须调用同一个父进程路径解析。查找顺序与本机 glibc 的 `execvp` 一致，不得改成写死 `/usr/bin/nvidia-smi`。子进程替换映像只用 `execve`，文件参数是该绝对路径。环境用 `fork` 时已经可见的 `environ`，不得在子进程里组装。
- 两处查询参数字符串保持不变。`query_nvidia_smi` 的 3 秒截止、超时 `SIGKILL`、`waitpid` 保持不变。采样器的 3 秒读截止、阻塞 `waitpid` 和三句首次日志的原文保持不变。找不到可执行文件时不 `fork`，失败结果仍是现在的空快照和 offline 日志。
- 模型子进程的 argv 第 0 项仍是可执行文件名。跟踪行 arg0 来自 `execve` 的文件参数，不靠改 argv 第 0 项来满足「以 `/` 开头」。`fork` 前的环境块字段、`worker_nums`、execution context、动态 batch、gateway token 规则、限流和 systemd 单元，不在本条里改。
- 自测通过只认上面同一次启动的跟踪日志和状态。libface 的终点行必须是关 fd 之后追加的，`inh=closed`。探测有输出时，那个直接子进程只按同一段窗口判定，不得把路径是否以 `/nvidia-smi` 结尾写成通过或失败。不得用对照源码作为通过。只读到 `exec` 之后同一 pid 的调用，不得判通过。
