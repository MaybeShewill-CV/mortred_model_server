# 技术方案

- 问题: P2-6
- 版本: 信号顺序
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-6

## 边界

锁定句照录。systemd 单元没有把停止信号限制在 supervisor 主进程（OP5）。场景是对已安装的 `mortred-supervisor.service` 执行停止。本机没有 systemd 时，向 supervisor 主进程发送停止信号，按同一进程顺序验收。现在单元设置了 `KillSignal=SIGTERM` 和 `TimeoutStopSec=120`，没有 `KillMode=`。supervisor 进程自己收到 SIGINT 或 SIGTERM 后，`perform_ordered_shutdown()` 会先向模型进程发 SIGINT 并等待，再停 gateway。单元没有把 systemd 的停止信号收成只发给主进程。改完后能观察到的变化：执行停止后，驻留集模型进程先退出，gateway 随后退出，gateway 退出后这些模型进程不再存在。

用户选定的可观察行为：一次外部跟踪里，supervisor 先向 yolov8s 和 libface 发 SIGINT 并等到它们退出，再向 gateway 发 SIGINT。gateway 退出后这两个模型进程不再存在。不靠拖住 gateway，也不要求采样落进短于 1 毫秒的停留。

这次通过不靠 50 毫秒或 1 毫秒去采「两个模型 pid 已不在、gateway 仍在」那一拍。那一拍的停留可以短于采样间隔，没有这一拍不算失败。通过看的是 supervisor 自己发出的系统调用顺序：两个模型的 SIGINT，两个模型的等待返回，然后才是 gateway 的 SIGINT。gateway 退出之后，这两个模型 pid 不再存在。

systemd 未写 `KillMode=` 时，停止信号按 control-group 打进该单元 cgroup 里的每个进程。`KillMode=process` 使 `KillSignal` 只打到 `ExecStart` 的主进程，也就是 `mortred-supervisor.out`。本机验收不安装、不 enable、不 `systemctl start`、不 `systemctl stop`。仓库里的单元文件证明已安装单元在下一次 `mortredctl install-host` 拷贝后会按这个范围停止。进程顺序由本次脚本自己拉起的 supervisor 证明：只向该主进程发 SIGTERM，并用 strace 跟踪这个主进程。本机没有 systemd，或没有由 systemd 管理的 `mortred-supervisor.service`，不记失败，也不记阻塞。8787、8080、9051、9056 已被占用时，实跑退出码不是 0，也不去停那个占用者。

这次实跑的驻留集就是这份临时 pack 拉起的 `yolov8s` 与 `libface` 两个模型进程，加上 `__gateway`。`servers` 里其余 catalog id 不参与退出顺序。两个模型之间不要求谁先谁后。

不把 P2-7、P2-9、P2-10、P2-11 拉进来。不改 TensorRT execution context 的归属，不改动态 batch，不改 token 启动判定。

## 做法

### 推荐

在 `deploy/mortred-supervisor.service` 的 `[Service]` 里增加一行 `KillMode=process`，与现有的 `KillSignal=SIGTERM`、`TimeoutStopSec=120` 放在一起。`Type=simple` 的主进程仍是 `ExecStart` 的 `mortred-supervisor.out`。不增加 `ExecStop=`。`install_host` 拷贝单元时对路径和 cpu profile 的现有替换不碰这一行。

有序退出仍用现有的 `perform_ordered_shutdown()`：先对每个已有 pid 的模型子进程发 SIGINT 并等待，再对 gateway 发 SIGINT 并等待。主进程收到 SIGTERM 后走这条已有路径。验收不改这段等待，也不在脚本里把 gateway 的 `stop()` 拖长。

另增 `scripts/stop_order_record.py`。`--check` 不拉起进程，不调用 strace。交付后的单元文件满足契约，并且脚本内的固定样例跟踪能通过同一套顺序判定时，`--check` 的标准输出恰好是验收标准 1 的七行，退出码为 0。`--check` 不创建 `logs/stop_order_yolov8s_libface.json`。

实跑用仓库外临时 pack，只含 `[pack.yolov8s]` 与 `[pack.libface]`，每节 `worker_nums = 1`、`gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"`。不写回 `conf/packs/`，不设置 `decode_auto`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。二进制先找 `_bin/mortred-supervisor.out`，再找 `bin/mortred-supervisor.out`。`LD_LIBRARY_PATH` 按 `scripts/mortredctl_next.sh` 带上 `_lib` 与 `3rd_party/libs`。`MORTRED_PROJECT_ROOT` 为仓库根。三个 token 只从已有环境或 `conf/local/trust.env` 读入，须都非空且两两不同，不打印、不写入记录。supervisor 由本机 `/usr/bin/strace` 拉起：`strace -f -tt -e trace=kill,tgkill,tkill,rt_sigqueueinfo,wait4,waitpid`，输出写入临时文件。strace 的子进程才是 supervisor。supervisor 以新会话启动。两者都 `ready` 且 `state` 为 `running` 之后，只对这次的 supervisor 主进程 pid 发 SIGTERM，不对进程组发信号，也不向 gateway 或模型 pid 发信号，也不对 `127.0.0.1:8080` 建立连接。这个 pid 同时写入 `signal_pid` 与 `supervisor_pid`。

跟踪里只保留发出者是这个 supervisor pid 的 `kill`、`tgkill`、`tkill`、`rt_sigqueueinfo`，以及 `wait4`、`waitpid` 的返回 pid 属于这次三个子进程的行。每行有 strace 的时刻，记成从当天 0 点起的微秒整数 `ts_us`，同一份跟踪里可以比较大小。信号号 2 记为 `SIGINT`。模型之间不要求谁先谁后。两个模型都要有一条 `op` 为 `kill`、`signal` 为 `SIGINT` 的事件，以及一条更晚的 `op` 为 `wait` 的事件。gateway 的 `kill`/`SIGINT` 的 `ts_us` 必须大于两个模型 `wait` 的 `ts_us`。没有这组顺序就是失败。不把「两个模型 pid 与 gateway pid 在同一次 /proc 查看里一起消失」写成失败，也不写成 `models_gone`。

gateway 的 `wait` 出现之后，再看一次这三个 pid。写入 `after_gateway`：`gateway_alive`、`yolov8s_alive`、`libface_alive` 都是 false。这一看发生在 gateway 已经被等待返回之后，不要求它落在短于 1 毫秒的窗口里。

发出 SIGTERM 的那一刻读取 `/proc/net/tcp`，数本机 `127.0.0.1:8080` 上状态为已建立的连接数，写入整数 `established_to_8080`。通过值是 0。脚本自己不连接 8080。

判定通过才写入 `logs/stop_order_yolov8s_libface.json`。

代价：`KillMode=process` 下，主进程退出后 systemd 不会再向 cgroup 里剩下的进程发 SIGKILL。子进程仍靠现有的有序退出收掉。实跑会拉起 `yolov8s` 与 `libface`，停止前占用这两份 static batch=1 引擎的显存。strace 只在这次验收进程上打开，不留在产品二进制里。

### 否决

用 50 毫秒或 1 毫秒轮询 `/proc/<pid>`，把「两个模型已不在、gateway 仍在」当成必须出现的一拍。现有退出在模型被 `waitpid` 回收之后马上对 gateway 发 SIGINT，没有连接时 gateway 的 `stop()` 不会等客户端。这段停留可以短于 1 毫秒。轮询跨过它就得不到这一拍，把没有这一拍判失败会把已经按顺序发出的 SIGINT 记成失败。

对 `127.0.0.1:8080` 保持一条不发请求的 TCP 连接，让 `stop()` 等到连接关闭再返回。这样能把 gateway 拖过一轮采样，但 gateway 仍在是因为连接还开着。

把 `KillMode` 写成 `mixed`，或只改 `mortredctl down` 的 `pkill`。`mixed` 会在主进程退出后立刻向剩余进程发 SIGKILL。`pkill` 不是锁定场景里的那一次单元停止。

## 改动面

- `deploy/mortred-supervisor.service`：`[Service]` 增加一行 `KillMode=process`。`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 保持原文。
- `scripts/stop_order_record.py`：单元判定、临时 pack、strace、顺序判定、记录、`--check`。
- `test/CMakeLists.txt`：按 `resident_pair_record_rules` 的注册方式增加 ctest `stop_order_record_rules`，命令是 `python3 scripts/stop_order_record.py --check`，`WORKING_DIRECTORY` 为仓库根。不打 `valgrind` 标签。实跑不注册。

## 明确不改

- `perform_ordered_shutdown()` 的顺序、对子进程的 SIGINT、5000 毫秒宽限和随后的 SIGKILL。
- `fork` 之后、`execve` 之前的路径，包括 `PR_SET_PDEATHSIG` 为 SIGINT。
- 每个 worker 各自的 TensorRT execution context。不增加共享 context，也不增加跨 worker 的 context 池。
- 动态 batch、产品 toml 的 `type`、`conf/server` 的 `worker_nums`、engine 的 static shape 与 batch=1。
- token 启动判定，以及三个 token 的拒绝规则。
- `src/control/cli/cli_app.cpp` 里 `down` 的 `systemctl stop` 与 `pkill`。
- `scripts/mortredctl_install-host.sh`、`scripts/install_host_common.sh`、`scripts/docker_entrypoint.sh`、`docker-compose.yml`、`deploy/nginx/mortred-edge.service`。
- `conf/packs/` 里已有的四份 pack。临时 pack 不写回这里。
- 推理请求、`logs/resident_pair_yolov8s_libface.json`、calibrate、`http_infer_rps.py`。

## 契约

单元 `[Service]` 在 `[Install]` 之前有且只有一行 `KillMode=process`，等号两侧没有空格。同一文件不再出现第二个 `KillMode=`。`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 两行保持原文。不增加 `ExecStop=`。

HTTP 路径、状态 JSON 的字段名、tensor 形状都不改。`GET /api/v1/status` 仍把 gateway 放在键 `gateway` 下，`id` 为 `__gateway`；模型在 `servers` 数组里按 `id` 区分。

这次发给 supervisor 的信号是 SIGTERM。跟踪里看到的、打到模型与 gateway 的信号是 supervisor 自己发的 SIGINT。不是测试脚本发的 SIGINT，不是进程组，不是 `systemctl`。

记录文件是一个 JSON 对象，键为 `signal`、`signal_pid`、`supervisor_pid`、`before`、`pack`、`pids`、`ppid`、`established_to_8080`、`events`、`after_gateway`。`signal` 为 `SIGTERM`。`signal_pid` 与 `supervisor_pid` 是同一个正整数。`pids` 与 `ppid` 的键为 `yolov8s`、`libface`、`gateway`。`ppid` 的三个值都等于这个整数。`established_to_8080` 为 0。全文不出现 `systemctl`。

`events` 是数组。每个对象有 `pid_key`、`op`、`ts_us`。`pid_key` 是 `yolov8s`、`libface`、`gateway` 之一。`op` 是 `kill` 或 `wait`。`kill` 另有 `signal`，这里是 `SIGINT`。`ts_us` 是整数。验收只比较这些整数的大小，不把它们换成「pid 是否仍在」的采样拍。

`after_gateway` 有 `gateway_alive`、`yolov8s_alive`、`libface_alive` 三个布尔值，没有 pid 键。三个值都是 false。

## 性能与资源

不生成 engine，不改 execution context 的个数，不做 calibrate。临时 pack 的 1800 MiB 只用于这次拉起通过现有占用门闩。

实跑等待 ready 的上限是现有 autostart 预算：gateway 15 秒，模型 60 秒，合计 75 秒，不延长。发出 SIGTERM 之后等待 supervisor 主进程退出的上限是 30 秒。不改 `TimeoutStopSec`。

## 验收标准

同一次外部验收先确认 `logs/stop_order_yolov8s_libface.json` 不存在，再跑「自测命令」第 1 条，看标准输出和退出码，打开 `deploy/mortred-supervisor.service`，并再次确认该路径不存在。然后跑第 2 条，只打开这一份 json。下面 7 条都要满足。只满足第 1 条不通过。无 GPU 的 CI 跑过 `stop_order_record_rules` 不得写成 OP5 已满足。

1. 同一次只跑交付后的 `python3 scripts/stop_order_record.py --check`，退出码为 0。标准输出恰好是下面七行，每行以换行结束，第七行之后没有空行，也没有其他标准输出：

```
KillMode=process 为通过
KillSignal=SIGTERM 为通过
TimeoutStopSec=120 为通过
模型 SIGINT 并等待之后才向 gateway 发 SIGINT 为通过
gateway 的 SIGINT 早于任一模型退出为不通过
未等到两个模型退出就向 gateway 发 SIGINT 为不通过
gateway 退出后模型 pid 再次出现为不通过
```

跑该命令之前 `logs/stop_order_yolov8s_libface.json` 不存在，命令结束后该路径仍不存在。同一次打开 `deploy/mortred-supervisor.service`：`[Service]` 在 `[Install]` 之前有且只有一行 `KillMode=process`，等号两侧没有空格；`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 两行保持原文；没有 `ExecStop=`。这次验收不改单元文件再跑。后四行是这次标准输出里的四句固定文字。`--check` 退出码 0 单独不表示驻留进程已经按顺序退出。

2. `python3 scripts/stop_order_record.py` 退出码为 0，并写出仓库根下的 `logs/stop_order_yolov8s_libface.json`。该文件的文本里不出现这次环境中的管理 token、推理 token、metrics token，也不出现 `systemctl`。

3. `before` 是同一次 `GET /api/v1/status` 的响应正文。`yolov8s` 与 `libface` 只取 `servers` 里 `id` 为这两个值的对象，各恰好一个。gateway 只取键 `gateway`，其 `id` 为 `__gateway`。这三处的 `ready` 都是 true，`state` 都是 `running`，`pid` 都是正整数，且与 `pids` 里的同名值相同。`servers` 里其余 id 不要求 `ready`。

4. `pack` 全文恰好两个 `[pack.` 节，名字是 `yolov8s` 与 `libface`。全文没有 `model_config`，也没有 `decode_auto`。两节都有 `worker_nums = 1`、`gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"`。

5. 只看这份 json：`signal` 为 `SIGTERM`；`signal_pid` 与 `supervisor_pid` 是同一个正整数；`ppid` 的 `yolov8s`、`libface`、`gateway` 都等于这个整数；`established_to_8080` 是 0。验收不再另取进程号，也不另查连接。

6. 只看 `events`。`yolov8s` 与 `libface` 各有一条 `op` 为 `kill`、`signal` 为 `SIGINT` 的事件，以及一条 `ts_us` 更大的 `op` 为 `wait` 的事件。`gateway` 有一条 `op` 为 `kill`、`signal` 为 `SIGINT` 的事件，它的 `ts_us` 大于两个模型 `wait` 事件的 `ts_us`。缺少其中一条，或 gateway 的这条 `kill` 不晚于某个模型的 `wait`，标准 6 不通过。这里不要求一份「两个模型 pid 已不在、gateway pid 仍在」的采样。

7. 只看 `after_gateway`。`gateway_alive`、`yolov8s_alive`、`libface_alive` 都是 false。对象里不写 pid。这三个布尔值表示标准 3 记下的那三个 pid 在 gateway 的 `wait` 之后是否仍在。

## 测试义务

白盒只有 `stop_order_record_rules`：默认 `check` 会跑到它，它必须退出码 0，标准输出是验收标准 1 的七行。它不拉起 supervisor、gateway 或模型，不写 json，也不调用 `systemctl` 或 strace。实跑不进入默认 `check`。`--check` 退出码 0 不能单独当作 OP5 通过。

实跑是黑盒：打开第 2 条写出的那一份 json，按验收标准 2 至 7 判定。不发推理请求。不读 `logs/resident_pair_yolov8s_libface.json`。

缺 systemd 时，实跑仍只向自己拉起的主进程发 SIGTERM，退出码仍按第 2 条为 0，不写成阻塞。端口被占用、二进制不存在、三个 token 缺失或两两有相同、75 秒内两者没有同时 ready、30 秒内 supervisor 没有退出、strace 没有写出标准 6 的顺序，退出码都不是 0，该次不得改记为 0，也不得改记为阻塞。

不把 `ctest -R stop_order_record_rules` 的通过写成驻留进程已经按顺序退出。

## 自测命令

工作目录是 `/mnt/g/Codex/mortred_model_server`。两条都在本机 WSL Ubuntu 执行。实现时改在问题 worktree 里执行，这是执行位置。

1.

```bash
python3 scripts/stop_order_record.py --check
```

期望退出码：0。同时满足验收标准 1 的七行标准输出，以及 `logs/stop_order_yolov8s_libface.json` 在命令前后都不存在。同一次打开 `deploy/mortred-supervisor.service` 并按验收标准 1 看单元正文。

2.

```bash
python3 scripts/stop_order_record.py
```

期望退出码：0。完成后仓库根下有 `logs/stop_order_yolov8s_libface.json`，并且该文件满足验收标准 2 至 7。命令不调用 `systemctl`。本机没有 systemd 时，这条命令向这次拉起的 supervisor 主进程发 SIGTERM，期望退出码仍是 0，不记阻塞。

## RD 不得自行决定

- `KillMode` 的值。它是 `process`。不写成 `mixed`、`control-group` 或 `none`。
- 不改 `KillSignal` 与 `TimeoutStopSec`。不增加 `ExecStop=`。
- 实跑发给 supervisor 的信号是 SIGTERM，目标只是这次拉起的 supervisor 主进程 pid。不向进程组、gateway pid、模型 pid 发信号。
- 不把缺 systemd 写成阻塞。不在实跑里 `systemctl stop`。
- 跟踪工具是 `/usr/bin/strace`，参数包含 `-f -tt`，系统调用集合包含 `kill`、`tgkill`、`tkill`、`rt_sigqueueinfo`、`wait4`、`waitpid`。不改成 1 毫秒或 50 毫秒轮询 `/proc` 来证明顺序。
- 不要求「两个模型已不在、gateway 仍在」这一拍。没有这一拍不是失败。
- 不对 `127.0.0.1:8080` 保持连接，也不用别的客户端等待把 `stop()` 拖长。`established_to_8080` 必须是 0。
- 驻留集就是 `yolov8s` 与 `libface`。
- 不改 `perform_ordered_shutdown()`，不改 token 启动判定，不改每个 worker 的 TensorRT execution context，不改动态 batch。
- `--check` 不得创建 `logs/stop_order_yolov8s_libface.json`。实跑不进默认 `check`。`--check` 的退出码 0 不能单独当作 OP5 通过。

## 确认

| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 不同意 | 见 v3 确认 | v3 | cowork/ledger/round-001/issues/p2-6/03-discussion/v3-sign-pm.md |
| RD | 不同意 | 见 v3 确认 | v3 | cowork/ledger/round-001/issues/p2-6/03-discussion/v3-sign-rd.md |
| 技术负责人 | 同意 | 无 | v3 | cowork/ledger/round-001/issues/p2-6/03-discussion/v3-sign-tech-lead.md |
| 用户 | 选定 | 无 | 信号顺序 | 本文件「用户决定」 |

## 用户决定

- 日期: 2026-10-13
- 选用版本: 信号顺序
- 决定原文: 一次外部跟踪里，supervisor 先向 yolov8s 和 libface 发 SIGINT 并等到它们退出，再向 gateway 发 SIGINT。gateway 退出后这两个模型进程不再存在。不靠拖住 gateway，也不要求采样落进短于 1 毫秒的停留。
- 未选用:

v2 的可观察测量：50 毫秒采样必须看到两个模型已经退出、当时 gateway 与 supervisor 仍在，然后 gateway 退出。

v3 的可观察测量：1 毫秒采样必须看到两个模型已经退出、当时 gateway 与 supervisor 仍在，然后 gateway 退出。不保持到 8080 的空连接。

PM 对 v3 的条件原文：标准 6 只看这份 json。顶层有整数 `sample_interval_ms`，值是 1；没有这个键或值不是 1，标准 6 不通过。`models_gone` 与 `gateway_gone` 各有整数 `elapsed_ms`，从发出 SIGTERM 到该次采样的毫秒。`gateway_gone.elapsed_ms` 减去 `models_gone.elapsed_ms` 的差大于等于 0，并且小于两处 `sample_index` 之差的 50 倍。没有 `elapsed_ms`、差为负、或差大于等于序号差的 50 倍，标准 6 不通过。`models_gone` 上 `yolov8s_alive` 与 `libface_alive` 仍是 false，`gateway_alive` 与 `supervisor_alive` 仍是 true；差大于等于 0 不代替这一拍，没有这一拍标准 6 不通过。`models_gone` 另有整数 `established_to_8080`，值是 0，表示这一拍本机 `127.0.0.1:8080` 上已建立的 TCP 连接数是 0；大于 0 标准 6 不通过。这次验收不另取进程号，也不另查连接。实跑不对 `127.0.0.1:8080` 保持一条不发请求的 TCP 连接，也不用别的客户端等待把 gateway 的退出拖长。标准 7 不因 `elapsed_ms` 或 `established_to_8080` 失败；这两个对象仍然不写 pid；json 仍然不另含这两次采样之间的采样。

停止本轮未被选中。
