# 技术方案

- 问题: P2-6
- 版本: v3
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 锁定队列 v3 的 P2-6

## 边界

锁定句照录。systemd 单元没有把停止信号限制在 supervisor 主进程（OP5）。场景是对已安装的 `mortred-supervisor.service` 执行停止。本机没有 systemd 时，向 supervisor 主进程发送停止信号，按同一进程顺序验收。现在单元设置了 `KillSignal=SIGTERM` 和 `TimeoutStopSec=120`，没有 `KillMode=`。supervisor 进程自己收到 SIGINT 或 SIGTERM 后，`perform_ordered_shutdown()` 会先向模型进程发 SIGINT 并等待，再停 gateway。单元没有把 systemd 的停止信号收成只发给主进程。改完后能观察到的变化：执行停止后，驻留集模型进程先退出，gateway 随后退出，gateway 退出后这些模型进程不再存在。

上一版审查意见：PM 不同意，条件已写入本版正文。条件原文：标准 1 同一次只跑交付后的 `python3 scripts/stop_order_record.py --check`，退出码为 0。标准输出恰好是这七行，每行以换行结束，第七行之后没有空行，也没有其他标准输出：`KillMode=process 为通过`、`KillSignal=SIGTERM 为通过`、`TimeoutStopSec=120 为通过`、`模型先退出且当时 gateway 与 supervisor 仍在为通过`、`gateway 先于任一模型消失为不通过`、`模型消失时 supervisor 已退出为不通过`、`gateway 退出后模型 pid 再次出现为不通过`。跑该命令之前 `logs/stop_order_yolov8s_libface.json` 不存在，命令结束后该路径仍不存在。同一次打开 `deploy/mortred-supervisor.service`：`[Service]` 在 `[Install]` 之前有且只有一行 `KillMode=process`，等号两侧没有空格；`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 两行保持原文；没有 `ExecStop=`。这次验收不改单元文件再跑。后四行是这次标准输出里的四句固定文字。`--check` 退出码 0 单独不表示驻留进程已经按顺序退出。标准 5 只看这份 json：`signal` 为 `SIGTERM`；`signal_pid` 与 `supervisor_pid` 是同一个正整数；`ppid` 的 `yolov8s`、`libface`、`gateway` 都等于这个整数；全文不出现 `systemctl`。验收不再另取发出信号时的进程号。标准 7 只看 `gateway_gone` 与 `after_supervisor` 这两个对象。`gateway_gone` 的 `yolov8s_alive` 与 `libface_alive` 都是 false。`after_supervisor` 的 `supervisor_alive`、`gateway_alive`、`yolov8s_alive`、`libface_alive` 都是 false，且 `sample_index` 大于 `gateway_gone.sample_index`。这四个布尔值表示标准 3 记下的那四个 `pid` 在该次采样是否仍在，对象里不重复写 pid。json 不另含这两次采样之间的采样。

这段原文里的标准 1、标准 5、标准 7 就是下面验收标准的第 1、5、7 条。v1 标准 1 里五种未在这次命令中出现的单元文件，以及「前三行来自仓库文件」「后四行来自脚本内的固定样例，实跑不读这些样例」，不写入本版。v1 标准 5 里「等于发出信号时那个主进程的 pid」不写入本版。v1 标准 7 里「gateway 已消失之后直到 supervisor 消失，两个模型 pid 都不再变成 true」不写入本版。

systemd 未写 `KillMode=` 时，停止信号按 control-group 打进该单元 cgroup 里的每个进程。`KillMode=process` 使 `KillSignal` 只打到 `ExecStart` 的主进程，也就是 `mortred-supervisor.out`。本机验收不安装、不 enable、不 `systemctl start`、不 `systemctl stop`。仓库里的单元文件证明已安装单元在下一次 `mortredctl install-host` 拷贝后会按这个范围停止。进程顺序由本次脚本自己拉起的 supervisor 证明：只向该主进程发 SIGTERM。本机没有 systemd，或没有由 systemd 管理的 `mortred-supervisor.service`，不记失败，也不记阻塞。8787、8080、9051、9056 已被占用时，实跑退出码不是 0，也不去停那个占用者。

这次实跑的驻留集就是这份临时 pack 拉起的 `yolov8s` 与 `libface` 两个模型进程，加上 `__gateway`。`servers` 里其余 catalog id 不参与退出顺序。两个模型之间不要求谁先谁后。

不把 P2-7、P2-9、P2-10、P2-11 拉进来。不改 TensorRT execution context 的归属，不改动态 batch，不改 token 启动判定。

## 做法

### 推荐

在 `deploy/mortred-supervisor.service` 的 `[Service]` 里增加一行 `KillMode=process`，与现有的 `KillSignal=SIGTERM`、`TimeoutStopSec=120` 放在一起。`Type=simple` 的主进程仍是 `ExecStart` 的 `mortred-supervisor.out`。不增加 `ExecStop=`。`install_host` 拷贝单元时对路径和 cpu profile 的现有替换不碰这一行。

有序退出仍用现有的 `perform_ordered_shutdown()`：先对每个已有 pid 的模型子进程发 SIGINT 并等待，再对 gateway 发 SIGINT 并等待。主进程收到 SIGTERM 后走这条已有路径。验收不改这段等待。

另增 `scripts/stop_order_record.py`。`--check` 与实跑共用同一个单元判定和同一套退出顺序判定。`--check` 不拉起进程。交付后的 `deploy/mortred-supervisor.service` 满足契约时，`--check` 的标准输出恰好是验收标准 1 的七行，退出码为 0。后四行是脚本写出的四句固定文字。`--check` 不创建 `logs/stop_order_yolov8s_libface.json`。这次验收打开的就是这份单元文件，不改它再跑，也不另备单元文件。

实跑用仓库外临时 pack，只含 `[pack.yolov8s]` 与 `[pack.libface]`，每节 `worker_nums = 1`、`gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"`。不写回 `conf/packs/`，不设置 `decode_auto`、`MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`。二进制先找 `_bin/mortred-supervisor.out`，再找 `bin/mortred-supervisor.out`。`LD_LIBRARY_PATH` 按 `scripts/mortredctl_next.sh` 带上 `_lib` 与 `3rd_party/libs`。`MORTRED_PROJECT_ROOT` 为仓库根。三个 token 只从已有环境或 `conf/local/trust.env` 读入，须都非空且两两不同，不打印、不写入记录。supervisor 以新会话启动。两者都 `ready` 且 `state` 为 `running` 之后，只对这次的 supervisor 主进程 `pid` 发 SIGTERM，不对进程组发信号，也不向 gateway 或模型 pid 发信号。这个 pid 同时写入 `signal_pid` 与 `supervisor_pid`。采样间隔固定为 1 毫秒，指相邻两次查看这四个 pid 的间隔。不持连接时，两个模型在停止信号后 0.1314 秒消失，gateway 在 0.1336 秒消失，窗口约 2.2 毫秒；50 毫秒一拍会从四个 pid 都在跨到 supervisor 仍在、gateway 与两个模型都已不在，中间没有 `models_gone`。1 毫秒小于该窗口，因此 `models_gone` 仍是这一间隔上的一拍：两个模型已不在，gateway 与 supervisor 仍在。同一拍里 gateway 与两个模型都已经不在，不能写成 gateway 仍在。没有这一拍就是失败，不得写成通过，也不改成两个时间戳小于等于。不对 `127.0.0.1:8080` 保持一条不发请求的 TCP 连接，也不用别的客户端等待把 `stop()` 拖长。判定通过才写入 `logs/stop_order_yolov8s_libface.json`。

写入的采样对象只有 `models_gone`、`gateway_gone`、`after_supervisor`。每个对象只有整数 `sample_index`，以及 `supervisor_alive`、`gateway_alive`、`yolov8s_alive`、`libface_alive` 四个布尔值。四个布尔值表示标准 3 记下的那四个 pid 在该次采样是否仍在。对象里不写 pid。`sample_index` 是 1 毫秒采样的序号。`gateway_gone` 与 `after_supervisor` 之间的采样不写入 json。同一次采样里 gateway 与 supervisor 都已不在时，仍再取下一次采样写入 `after_supervisor`，使后者的 `sample_index` 更大。

代价：`KillMode=process` 下，主进程退出后 systemd 不会再向 cgroup 里剩下的进程发 SIGKILL。子进程仍靠现有的有序退出收掉：SIGINT、等待、宽限后再 SIGKILL。宽限仍是现有的 5000 毫秒加随后 2000 毫秒，两份模型再加 gateway，落在现有 `TimeoutStopSec=120` 里面。已经装到 `/etc/systemd/system/mortred-supervisor.service` 的旧文件要等下一次安装拷贝才带上这一行；这次验收不重装。实跑会按 `worker_nums = 1` 拉起 `yolov8s` 与 `libface`，停止前占用这两份 static batch=1 引擎的显存。每个模型进程仍是一个 OS 进程，worker 仍是该进程里的线程，各持有自己的 execution context。

### 否决

把 `KillMode` 写成 `mixed`。第一次 SIGTERM 可以只打到主进程，120 秒内有序退出也能走完。systemd 的规则是：`mixed` 在主进程已经退出而 cgroup 里还有进程时，立刻向这些剩余进程发 SIGKILL，不等到 `TimeoutStopSec`。supervisor 必须活到两个模型都退出之后才向 gateway 发 SIGINT。主进程若在 gateway 仍在时退出，gateway 和尚未消失的模型会一起收到 SIGKILL。这不是「停止信号只发给主进程」。

不改单元，只改 `mortredctl down` 在 `systemctl stop` 失败后的 `pkill -TERM`，或让 gateway 与模型忽略 SIGTERM、只认 supervisor 发来的 SIGINT。已安装单元在默认 control-group 下仍会把 SIGTERM 打进 cgroup 里的每个进程。gateway 现有的停止闩把 SIGTERM 当成退出，gateway 可以在模型退出之前消失。让子进程忽略 SIGTERM 会改到 `src/common/process_stop.h`，单进程直接收到 SIGTERM 时不再退出。`down` 的 `pkill` 也不是锁定场景里的那一次单元停止。

## 改动面

- `deploy/mortred-supervisor.service`：`[Service]` 增加一行 `KillMode=process`。`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 保持原文。
- `scripts/stop_order_record.py`：单元判定、临时 pack、实跑、1 毫秒采样、三份采样对象、记录、`--check`。
- `test/CMakeLists.txt`：按 `resident_pair_record_rules` 的注册方式增加 ctest `stop_order_record_rules`，命令是 `python3 scripts/stop_order_record.py --check`，`WORKING_DIRECTORY` 为仓库根。不打 `valgrind` 标签，因此进入默认 `check`（`ctest -LE valgrind`）。实跑不注册。

## 明确不改

- `perform_ordered_shutdown()` 的顺序、对子进程的 SIGINT、5000 毫秒宽限和随后的 SIGKILL。
- `fork` 之后、`execve` 之前的路径，包括 `PR_SET_PDEATHSIG` 为 SIGINT。
- 每个 worker 各自的 TensorRT execution context。不增加共享 context，也不增加跨 worker 的 context 池。
- 动态 batch、产品 toml 的 `type`、`conf/server` 的 `worker_nums`、engine 的 static shape 与 batch=1。
- token 启动判定，以及 `MORTRED_API_TOKEN`、`MORTRED_GATEWAY_AUTH_TOKEN`、`MORTRED_METRICS_TOKEN` 的拒绝规则。
- `src/control/cli/cli_app.cpp` 里 `down` 的 `systemctl stop` 与 `pkill`。单 id 的 `mortredctl stop` 仍只打该 id 的停止接口。
- `scripts/mortredctl_install-host.sh`、`scripts/install_host_common.sh`、`scripts/docker_entrypoint.sh`、`docker-compose.yml`、`deploy/nginx/mortred-edge.service`。
- `conf/packs/` 里已有的四份 pack。临时 pack 不写回这里。
- 推理请求、`logs/resident_pair_yolov8s_libface.json`、calibrate、`http_infer_rps.py`。

## 契约

单元 `[Service]` 在 `[Install]` 之前有且只有一行 `KillMode=process`，等号两侧没有空格。同一文件不再出现第二个 `KillMode=`。`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 两行保持原文，等号两侧没有空格。不增加 `ExecStop=`、`SendSIGKILL=`、`FinalKillSignal=`。

HTTP 路径、状态 JSON 的字段名、tensor 形状都不改。`GET /api/v1/status` 仍把 gateway 放在键 `gateway` 下，`id` 为 `__gateway`；模型在 `servers` 数组里按 `id` 区分。

这次停止信号是 SIGTERM，目标 pid 是这次拉起的 supervisor 主进程。不是 SIGINT，不是进程组，不是 `systemctl`。supervisor 现有逻辑对 SIGINT 与 SIGTERM 都进入同一套有序退出；验收信号与单元的 `KillSignal` 一致，用 SIGTERM。本机没有 systemd 时，实跑仍向这次拉起的主进程发 SIGTERM。

记录文件是一个 JSON 对象，键为 `signal`、`signal_pid`、`supervisor_pid`、`before`、`pack`、`pids`、`ppid`、`models_gone`、`gateway_gone`、`after_supervisor`。`signal` 为 `SIGTERM`。`signal_pid` 与 `supervisor_pid` 是同一个正整数。`pids` 与 `ppid` 的键为 `yolov8s`、`libface`、`gateway`。`ppid` 的三个值都等于这个整数。全文不出现 `systemctl`。验收不在这份 json 之外再取进程号。

`models_gone`、`gateway_gone`、`after_supervisor` 是仅有的三份采样对象。各有整数 `sample_index`，以及 `supervisor_alive`、`gateway_alive`、`yolov8s_alive`、`libface_alive` 四个布尔值，没有其他键，不写 pid。四个布尔值表示标准 3 记下的那四个 pid 在该次采样是否仍在。`gateway_gone.sample_index` 大于 `models_gone.sample_index`。`after_supervisor.sample_index` 大于 `gateway_gone.sample_index`。json 不另含这两次采样之间的采样。

## 性能与资源

不生成 engine，不改 execution context 的个数，不做 calibrate。临时 pack 的 1800 MiB 只用于这次拉起通过现有占用门闩，不写成这台卡的测量值。

实跑等待 ready 的上限是现有 autostart 预算：gateway 15 秒，模型 60 秒，合计 75 秒，不延长。发出 SIGTERM 之后等待 supervisor 主进程退出的上限是 30 秒。两个模型按现有宽限依次退出，再停 gateway，这段时间落在这 30 秒里，也落在单元的 120 秒里。不改 `TimeoutStopSec`。

`KillMode=process` 不增加常驻进程。停止完成后，这次的 supervisor、gateway、`yolov8s`、`libface` 四个 pid 都不再存在。worker 线程随各自的模型进程一起退出。

## 验收标准

同一次外部验收先确认 `logs/stop_order_yolov8s_libface.json` 不存在，再跑「自测命令」第 1 条，看标准输出和退出码，打开 `deploy/mortred-supervisor.service`，并再次确认该路径不存在。这次不改单元文件，也不再跑第 1 条。然后跑第 2 条，只打开这一份 json。下面 7 条都要满足。只满足第 1 条不通过。无 GPU 的 CI 跑过 `stop_order_record_rules` 不得写成 OP5 已满足。

1. 同一次只跑交付后的 `python3 scripts/stop_order_record.py --check`，退出码为 0。标准输出恰好是下面七行，每行以换行结束，第七行之后没有空行，也没有其他标准输出：

```
KillMode=process 为通过
KillSignal=SIGTERM 为通过
TimeoutStopSec=120 为通过
模型先退出且当时 gateway 与 supervisor 仍在为通过
gateway 先于任一模型消失为不通过
模型消失时 supervisor 已退出为不通过
gateway 退出后模型 pid 再次出现为不通过
```

跑该命令之前 `logs/stop_order_yolov8s_libface.json` 不存在，命令结束后该路径仍不存在。同一次打开 `deploy/mortred-supervisor.service`：`[Service]` 在 `[Install]` 之前有且只有一行 `KillMode=process`，等号两侧没有空格；`KillSignal=SIGTERM` 与 `TimeoutStopSec=120` 两行保持原文；没有 `ExecStop=`。这次验收不改单元文件再跑。后四行是这次标准输出里的四句固定文字。`--check` 退出码 0 单独不表示驻留进程已经按顺序退出。

2. `python3 scripts/stop_order_record.py` 退出码为 0，并写出仓库根下的 `logs/stop_order_yolov8s_libface.json`。该文件的文本里不出现这次环境中的管理 token、推理 token、metrics token。

3. `before` 是同一次 `GET /api/v1/status` 的响应正文。`yolov8s` 与 `libface` 只取 `servers` 里 `id` 为这两个值的对象，各恰好一个。gateway 只取键 `gateway`，其 `id` 为 `__gateway`。这三处的 `ready` 都是 true，`state` 都是 `running`，`pid` 都是正整数，且与 `pids` 里的同名值相同。`servers` 里其余 id 不要求 `ready`。

4. `pack` 全文恰好两个 `[pack.` 节，名字是 `yolov8s` 与 `libface`。全文没有 `model_config`，也没有 `decode_auto`。两节都有 `worker_nums = 1`、`gpu_mem_mib = 1800`、`gpu_mem_at_workers = 1`、`gpu_mem_source = "seed"`。

5. 只看这份 json：`signal` 为 `SIGTERM`；`signal_pid` 与 `supervisor_pid` 是同一个正整数；`ppid` 的 `yolov8s`、`libface`、`gateway` 都等于这个整数；全文不出现 `systemctl`。验收不再另取发出信号时的进程号。

6. `models_gone` 的 `yolov8s_alive` 与 `libface_alive` 都是 false，`gateway_alive` 与 `supervisor_alive` 都是 true。`gateway_gone` 的 `gateway_alive`、`yolov8s_alive`、`libface_alive` 都是 false，且 `sample_index` 大于 `models_gone.sample_index`。

7. 只看 `gateway_gone` 与 `after_supervisor` 这两个对象。`gateway_gone` 的 `yolov8s_alive` 与 `libface_alive` 都是 false。`after_supervisor` 的 `supervisor_alive`、`gateway_alive`、`yolov8s_alive`、`libface_alive` 都是 false，且 `sample_index` 大于 `gateway_gone.sample_index`。这四个布尔值表示标准 3 记下的那四个 `pid` 在该次采样是否仍在，对象里不重复写 pid。json 不另含这两次采样之间的采样。没有中间采样，并且这两个对象没有 pid 键，按本条通过。

## 测试义务

白盒只有 `stop_order_record_rules`：默认 `check` 会跑到它，它必须退出码 0，标准输出是验收标准 1 的七行。它不拉起 supervisor、gateway 或模型，不写 `logs/stop_order_yolov8s_libface.json`，也不调用 `systemctl`。同一次外部验收另外打开 `deploy/mortred-supervisor.service`，不改文件再跑。实跑不进入默认 `check`。`--check` 退出码 0 不能单独当作 OP5 通过。

实跑是黑盒：打开第 2 条写出的那一份 json，按验收标准 2 至 7 判定。标准 5 不另取进程号。标准 7 以这两个对象上的布尔值、`sample_index`，以及这两次之间没有另一份采样为通过。不发推理请求。不读 `logs/resident_pair_yolov8s_libface.json`。

缺 systemd 或没有 active 的 `mortred-supervisor.service` 时，实跑仍只向自己拉起的主进程发 SIGTERM。这种情况退出码仍按第 2 条要求为 0，不写成阻塞。端口被占用、二进制不存在、三个 token 缺失或两两有相同、75 秒内两者没有同时 ready、30 秒内 supervisor 没有退出，退出码都不是 0，该次不得改记为 0，也不得改记为阻塞。

不把 `ctest -R stop_order_record_rules` 的通过写成驻留进程已经按顺序退出。

## 自测命令

工作目录是 `/mnt/g/Codex/mortred_model_server`。两条都在本机 WSL Ubuntu 执行。

1.

```bash
python3 scripts/stop_order_record.py --check
```

期望退出码：0。同时满足验收标准 1 的七行标准输出，以及 `logs/stop_order_yolov8s_libface.json` 在命令前后都不存在。同一次打开 `deploy/mortred-supervisor.service` 并按验收标准 1 看单元正文。不改单元文件，不再跑这条命令。

2.

```bash
python3 scripts/stop_order_record.py
```

期望退出码：0。完成后仓库根下有 `logs/stop_order_yolov8s_libface.json`，并且该文件满足验收标准 2 至 7。命令不调用 `systemctl`。本机没有 systemd 时，这条命令向这次拉起的 supervisor 主进程发 SIGTERM，期望退出码仍是 0，不记阻塞。端口被占用、二进制不存在、三个 token 缺失或两两有相同、75 秒内两者没有同时 ready、30 秒内 supervisor 没有退出时，期望退出码不是 0。

## RD 不得自行决定

- `KillMode` 的值。它是 `process`。不写成 `mixed`、`control-group` 或 `none`，也不靠注释代替这一行。
- 不改 `KillSignal` 与 `TimeoutStopSec`。不增加 `ExecStop=`。
- 实跑的停止信号是 SIGTERM，目标只是这次拉起的 supervisor 主进程 pid。不向进程组、gateway pid、模型 pid 发信号，也不改用 SIGINT。`signal_pid` 与 `supervisor_pid` 写入同一个正整数。验收不另取发出信号时的进程号。
- 不把缺 systemd、`systemctl` 不存在、或单元不是 active 写成阻塞或写成退出码 0 的例外。不在实跑里 `systemctl stop` 已有单元。没有 systemd 时仍向这次拉起的主进程发 SIGTERM，退出码按验收标准 2 为 0。
- 这次验收不改 `deploy/mortred-supervisor.service` 再跑，也不另备单元文件。
- 驻留集就是 `yolov8s` 与 `libface`。不减成一个 id，也不扩成九个覆盖集 id。
- 采样间隔是 1 毫秒。它小于不持连接时两个模型消失到 gateway 消失的约 2.2 毫秒窗口，不得改成 50 毫秒或其他毫秒数。写入的采样对象只有 `models_gone`、`gateway_gone`、`after_supervisor`。json 不另含 `gateway_gone` 与 `after_supervisor` 之间的采样。这三个对象不写 pid。`models_gone` 是这一间隔上的一拍：两个模型已不在，gateway 与 supervisor 仍在。没有这一拍就是失败。同一拍里 gateway 与两个模型都已经不在，不得写成 gateway 仍在，也不得写成通过。不改成两个时间戳小于等于。不对 `127.0.0.1:8080` 保持一条不发请求的 TCP 连接，也不用别的客户端等待把 `stop()` 拖长。同一次采样里 gateway 与 supervisor 都已不在时，仍再取下一次采样作为 `after_supervisor`。
- 不改 `perform_ordered_shutdown()`，不改 token 启动判定，不改每个 worker 的 TensorRT execution context，不改动态 batch。
- `--check` 不得创建 `logs/stop_order_yolov8s_libface.json`。实跑不进默认 `check`。`--check` 的退出码 0 不能单独当作 OP5 通过。
