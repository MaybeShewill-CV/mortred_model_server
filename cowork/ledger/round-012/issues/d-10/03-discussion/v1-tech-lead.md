# 技术方案

- 问题: D-10
- 版本: v1
- 对应列表: `cowork/ledger/round-012/02-p0-p1-final.md` 的 D-10

## 边界

只做已经冻结的 D-10。冻结句不改写：`nvidia-smi` 读数为 N/A 时，监督侧仍把该采样当成有效 GPU 利用率。

改完后能观察到的变化只这一句：同一次探测的 `available` 为 false，或日志/状态写出 offline，CLI 不再把 -1 印成利用率百分比。

本条没有上一版方案，没有待并入的审查条件。

现在的成功探测走 `GpuSampler::loop`。`probe_nvidia_smi` 用 `execvp("nvidia-smi")` 取 CSV。标准输出非空时，`parse_i32_or` 把 utilization 列的 `N/A` 和 `[N/A]` 收成样本里的整数 `-1`，`available_` 仍设为 true，样本进入 ring，stderr 打印 `mortred-supervisor: gpu sampler online (...)`。`GET /api/v1/gpu` 把 `available` 和样本 `util` 原样写出。`mortredctl ps` 在 `samples` 非空时用最后一条的 `util` 打印 `util %lld%%`，`-1` 因此变成 `util -1%`。

`nvidia-smi` 整次失败且输出为空时，现有分支已经把 `available_` 设为 false，并打印一次 `mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`，不把样本推进 ring。那条路径不在本条里，不重做。

本方案把「读数为 N/A」收成两列，字段按现有 CSV 裁剪之后整列相等：`N/A`，或 `[N/A]`。推荐做法同时做出三件可观察结果：这次 `available` 为 false，stderr 有一行 utilization N/A 的 offline，`mortredctl ps` 把这条 `-1` 印成 `util n/a`。

不覆盖 P0-2、P1-8、D-8、D-11、D-12。不重开已关闭原句：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、P1-12、D-1 至 D-6、D-9。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`centerface` 的 engine 输入名 `input.1` 不重开。

## 做法

### 推荐

只改监督侧这一次采样的在线判定，以及 `mortredctl ps` 对 `-1` 的印法。

`GpuSampler` 在非空 CSV 已经解析出至少 11 列之后、写入 `available_ = true` 之前，看 utilization 这一列。现有裁剪之后整列等于 `N/A` 或 `[N/A]` 时：

- `available_` 设为 false，不设为 true。
- 样本仍推进 ring。`util` 仍是现有整数 `-1`。显存等其余列仍按现有解析写入。
- 先完成上述 `available_` 与 ring 更新，再向 stderr 写一行，全文是 `mortred-supervisor: gpu sampler offline (utilization N/A)`。同一进程这行最多一行。这次探测不写 `gpu sampler online`。
- 这次探测之后按配置的 `gpu_sample_interval_ms` 休眠，不走空输出分支的 backoff 翻倍。

utilization 列是整数时，现有成功分支保持不变：`available_` 为 true，样本入 ring，本进程还没打过采样日志时写 `mortred-supervisor: gpu sampler online (<name>)`。

`mortredctl ps` 只改利用率那一格。最后一条样本的 `util` 读出来是 `-1` 时，该格印 `util n/a`，后面仍是两个空格再接现有的 `vram`。不印 `util -1%`，这一格不加百分号。`util` 为 `0` 或其他非负整数时，仍印 `util <数字>%`。

代价：utilization 为 N/A 的卡，监督侧这次探测不再显示为在线利用率。`GET /api/v1/gpu` 的 `available` 为 false，样本里仍有整数 `util: -1`。`mortredctl ps` 仍打 GPU 行，利用率是 `util n/a`，显存格仍按现有整数算法。控制台在 `available` 为 false 时仍走现有的 “GPU offline” 面板。不增加进程，不占显存，不生成 engine。采样仍是 supervisor 里的一个线程，探活deadline仍是 3000 毫秒。

### 否决

把 N/A 列收成空输出，复用 `gpu sampler offline (no nvidia-smi / no GPU)`。空输出且整次失败已经标 offline，不在本条。成功返回的 CSV 会被写成没有 nvidia-smi，定位是错的。ring 里没有这条样本时，`mortredctl ps` 不会打出 `util n/a`。这条否决。

utilization 一旦解析成 `-1` 就把整次探测标成 offline，包括空列、列数不足和 `stod` 失败。冻结句只覆盖读数 `N/A` 与 `[N/A]`。空输出仍走现有分支。这条否决。

只改 CLI，`available` 仍为 true，stderr 仍是 `gpu sampler online`。`mortredctl ps` 即使不再印 `util -1%`，监督侧仍把这次采样当成有效利用率。这条否决。

## 改动面

- `src/control/gpu_sampler.h`：`GpuSampler::loop`。非空 CSV 的 utilization 列在现有裁剪后整列等于 `N/A` 或 `[N/A]` 时，按推荐做法更新 `available_`、ring 和 stderr。`parse_i32_or` 对其余列仍把 `N/A` 与 `[N/A]` 收成 `-1`。
- `src/control/cli/cli_app.cpp`：`cmd_ps` 里 GPU 行的利用率格。`util` 为 `-1` 时印 `util n/a`。

## 明确不改

- 空输出分支：`available_` 为 false，不入 ring，stderr 仍是一行 `mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`，backoff 仍按现有方式翻倍，上限仍是 30 秒。
- utilization 列裁剪后为空、列数不足 11、或不是 `N/A` / `[N/A]` 的非空 CSV。这些仍走现在的非空分支。
- 温度、功耗、风扇、时钟、PCIe 列上的 `N/A`。它们仍收成 `-1` 或 `-1.0`，不单独把 `available_` 设为 false，也不写本条的 offline 行。
- `parse_i32_or` / `parse_f64_or` 的大小写。小写 `n/a` 不归本条。
- 多卡多行、GPU 名称里含逗号时的列位移。
- `probe_nvidia_smi` 不看子进程退出码。探活 deadline 仍是 3000 毫秒。ring 容量仍是 300。已有的历史样本不因这次 N/A 被清掉。
- `src/control/occupancy_gate.h` 的 `nvidia-smi` 查询。那次查询的是显存，不是利用率。
- `src/control/supervisor/ui/`。`available` 为 false 时，现有 `pollGpu` 已经把面板写成 `GPU offline`。
- `GET /api/v1/gpu` 的键名。不新增 `status` 或其他键。`util` 仍是 JSON 整数，不是字符串，也不是 null。
- `mortredctl ps` 的绿色圆点、显存格式、温度键 `temp_c`、功率的整数读取。温度因此仍是 `temp -1°C`，功率因此仍是 `pwr -1W`。
- `conf/mortred.toml` 的 `gpu_sample_interval_ms = 500`，以及产品端口 gateway `:8080`、supervisor `:8787`。
- 产品 toml 的 `type=mnn` / `type=tensorrt`。HTTP 推理契约。张量契约。一个 catalog id 一个 OS 进程。engine 仍在本机本卡用 `mortredctl prepare` 生成。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT，也仍启动现有 sampler。
- P0-2、P1-8、D-8、D-11、D-12，以及已关闭原句。

## 契约

`GET /api/v1/gpu` 的键名不变。utilization 列是 `N/A` 或 `[N/A]` 的这一次探测，`available` 为 false。该样本的 `util` 仍是整数 `-1`。不新增 JSON 键。

`mortredctl ps` 在标准输出不是终端、颜色码为空时，这一次的 GPU 行利用率格是 `util n/a`，没有百分号。整数利用率仍是 `util <数字>%`。

空输出探测的 stderr 仍是：

`mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`

整数利用率第一次打出的 stderr 仍是：

`mortred-supervisor: gpu sampler online (<gpu name>)`

无张量契约变化。产品端口仍是 gateway `:8080`、supervisor `:8787`。

## 性能与资源

产品进程不变：sampler 仍是 supervisor 进程里的一个线程，不新起进程，不占显存，不生成 engine。探活 deadline 仍是 3000 毫秒。N/A 探测之后按 `gpu_sample_interval_ms` 休眠。产品 toml 的间隔仍是 500 毫秒。

验收只拉起一个临时 supervisor，项目根是临时目录，`conf/server` 为空，因此 catalog 为空，不会去拉 gateway 或模型。临时 toml 把间隔写成 60000 毫秒，使这次验收在结束前只有一条样本。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

四条都在 Linux 上、仓库根目录分开执行，不要并行。前置：已有 `./_bin/mortred-supervisor.out` 和 `./_bin/mortredctl.out`。四条都不修改仓库里的 `conf/mortred.toml`。临时目录在命令结束时删除。标准错误为空。脚本退出码是 0。

`mortredctl ps` 自身退出码是 1，因为验收不拉起网关，`18080` 的 `healthz` 没有响应。脚本把这个 1 收进判断，不把它当成脚本失败，也不把 `mortredctl` 的标准错误打到脚本的标准错误。`ps` 的标准输出经命令替换取得，因此不是终端，颜色码为空。

验收 1。utilization 列为 `[N/A]`。退出码 0。命令：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
sup_pid=
cleanup() {
  if [[ -n "${sup_pid:-}" ]]; then
    kill "$sup_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$sup_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    kill -9 "$sup_pid" 2>/dev/null || true
    wait "$sup_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/conf/server"
cat >"$tmp/bin/nvidia-smi" <<'SH'
#!/bin/sh
printf '%s\n' 'FakeGPU, [N/A], 2048, 8192, 40, 20.5, 1000, 5000, 30, 4, 16'
SH
chmod +x "$tmp/bin/nvidia-smi"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[supervisor]
api_host = "127.0.0.1"
api_port = 18787
autostart_default = false
gpu_sample_interval_ms = 60000
log_dir = "logs"

[gateway]
host = "127.0.0.1"
port = 18080
TOML
log_line='mortred-supervisor: gpu sampler offline (utilization N/A)'
env -u MORTRED_CONTROL_CONFIG -u MORTRED_PACK -u MORTRED_PROFILE -u MORTRED_API_PORT -u MORTRED_API_HOST -u MORTRED_AUTOSTART -u MORTRED_EXPOSE -u MORTRED_UI_DIR -u APP_BIN_DIR -u APP_LIB_DIR -u APP_LIBS_DIR -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST \
  PATH="$tmp/bin:$PATH" \
  MORTRED_PROJECT_ROOT="$tmp" \
  MORTRED_API_TOKEN=d10-token \
  MORTRED_METRICS_TOKEN=d10-metrics \
  MORTRED_GATEWAY_AUTH_TOKEN=d10-infer \
  ./_bin/mortred-supervisor.out >"$tmp/sup.err" 2>&1 &
sup_pid=$!
ok=0
for _ in $(seq 1 200); do
  if grep -F -q -x "$log_line" "$tmp/sup.err" && grep -F -q 'autostart blocked:' "$tmp/sup.err"; then
    code=$(curl -s -m 2 -o "$tmp/gpu.json" -w '%{http_code}' -H 'Authorization: Bearer d10-token' http://127.0.0.1:18787/api/v1/gpu || true)
    if [[ "$code" == "200" ]] && grep -F -q '"available":false' "$tmp/gpu.json" && grep -F -q '"util":-1,' "$tmp/gpu.json"; then
      n=$(grep -o '"util"' "$tmp/gpu.json" | wc -l)
      n=${n// /}
      if [[ "$n" == "1" ]]; then
        ok=1
        break
      fi
    fi
  fi
  sleep 0.05
done
[[ "$ok" == "1" ]] || exit 1
if grep -F -q 'gpu sampler online' "$tmp/sup.err"; then exit 1; fi
if grep -F -q 'gpu sampler offline (no nvidia-smi / no GPU)' "$tmp/sup.err"; then exit 1; fi
log_n=$(grep -F -c -x "$log_line" "$tmp/sup.err" || true)
[[ "$log_n" == "1" ]] || exit 1
ps_ec=0
ps_out=$(./_bin/mortredctl.out --addr http://127.0.0.1:18787 --gateway http://127.0.0.1:18080 --token d10-token ps 2>"$tmp/ps.err") || ps_ec=$?
[[ "$ps_ec" == "1" ]] || exit 1
[[ ! -s "$tmp/ps.err" ]] || exit 1
[[ "$ps_out" != *'util -1%'* ]] || exit 1
gpu_line=$(printf '%s\n' "$ps_out" | grep -F '  GPU         ● FakeGPU  util ' || true)
[[ "$gpu_line" == '  GPU         ● FakeGPU  util n/a  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W' ]] || exit 1
printf '%s\n' 'available=false'
printf '%s\n' 'sample_count=1'
printf '%s\n' 'util=-1'
printf '%s\n' "$log_line"
printf '%s\n' "$gpu_line"
EOF
```

标准输出原文：

```
available=false
sample_count=1
util=-1
mortred-supervisor: gpu sampler offline (utilization N/A)
  GPU         ● FakeGPU  util n/a  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W
```

验收 2。utilization 列为 `N/A`。退出码 0。命令与验收 1 相同，只把假 `nvidia-smi` 的那一行改成：

`printf '%s\n' 'FakeGPU, N/A, 2048, 8192, 40, 20.5, 1000, 5000, 30, 4, 16'`

标准输出原文与验收 1 相同。

验收 3。整数利用率仍算在线。退出码 0。命令与验收 1 相同，只改这三处：假 `nvidia-smi` 打出 `FakeGPU, 37, 2048, 8192, 40, 20.5, 1000, 5000, 30, 4, 16`；`log_line` 改为 `mortred-supervisor: gpu sampler online (FakeGPU)`；等待条件改为 HTTP 200 且正文同时含 `"available":true` 与 `"util":37,`，并且 `"util"` 出现一次。通过前还要确认 stderr 没有 `gpu sampler offline (utilization N/A)`，也没有 `gpu sampler offline (no nvidia-smi / no GPU)`，且该 online 行恰好一行。`ps` 全文不含 `util -1%`，也不含 `util n/a`。GPU 行全文改为下面标准输出里的那一行。

标准输出原文：

```
available=true
sample_count=1
util=37
mortred-supervisor: gpu sampler online (FakeGPU)
  GPU         ● FakeGPU  util 37%  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W
```

验收 4。空输出仍走原来的 offline。退出码 0。命令与验收 1 相同，只改这三处：假 `nvidia-smi` 改为 `#!/bin/sh` 然后 `exit 0`，不写标准输出；`log_line` 改为 `mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`；等待条件改为 HTTP 200、正文含 `"available":false` 与 `"samples":[]`、且不含 `"util"`。通过前还要确认 stderr 没有 `gpu sampler online`，也没有 `gpu sampler offline (utilization N/A)`，且该空输出 offline 行恰好一行。`ps` 全文不含 `util -1%`，不含 `util n/a`，也不含 `  GPU         `。脚本在 GPU 行为空时打印 `gpu_line=absent`。

标准输出原文：

```
available=false
sample_count=0
mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)
gpu_line=absent
```

验收 3 与验收 4 的完整命令见自测命令。通过条件仍是各自的退出码、标准输出原文和空的标准错误。

## 测试义务

QA 跑验收 1、验收 2、验收 3、验收 4。通过与否只看脚本退出码、标准输出原文和空的标准错误。不把读源码、打开控制台页面、拉起产品 gateway `:8080`、拉起产品 supervisor `:8787`、拉起模型、生成 engine、跑 GPU 数值比较写成通过条件。

验收 3 钉住整数利用率仍是 `available: true`、online 日志和 `util 37%`。验收 4 钉住空输出仍是原来的 offline 行，并且 CLI 没有利用率百分比。这两条不增加冻结句的完成范围。

## 自测命令

与验收标准里的四条命令逐字相同。在 Linux 上、仓库根目录分开执行。期望退出码都是 0。标准错误都为空。标准输出原文与对应验收相同。

验收 1 的命令与标准输出见上一节。

验收 2 的完整命令：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
sup_pid=
cleanup() {
  if [[ -n "${sup_pid:-}" ]]; then
    kill "$sup_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$sup_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    kill -9 "$sup_pid" 2>/dev/null || true
    wait "$sup_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/conf/server"
cat >"$tmp/bin/nvidia-smi" <<'SH'
#!/bin/sh
printf '%s\n' 'FakeGPU, N/A, 2048, 8192, 40, 20.5, 1000, 5000, 30, 4, 16'
SH
chmod +x "$tmp/bin/nvidia-smi"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[supervisor]
api_host = "127.0.0.1"
api_port = 18787
autostart_default = false
gpu_sample_interval_ms = 60000
log_dir = "logs"

[gateway]
host = "127.0.0.1"
port = 18080
TOML
log_line='mortred-supervisor: gpu sampler offline (utilization N/A)'
env -u MORTRED_CONTROL_CONFIG -u MORTRED_PACK -u MORTRED_PROFILE -u MORTRED_API_PORT -u MORTRED_API_HOST -u MORTRED_AUTOSTART -u MORTRED_EXPOSE -u MORTRED_UI_DIR -u APP_BIN_DIR -u APP_LIB_DIR -u APP_LIBS_DIR -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST \
  PATH="$tmp/bin:$PATH" \
  MORTRED_PROJECT_ROOT="$tmp" \
  MORTRED_API_TOKEN=d10-token \
  MORTRED_METRICS_TOKEN=d10-metrics \
  MORTRED_GATEWAY_AUTH_TOKEN=d10-infer \
  ./_bin/mortred-supervisor.out >"$tmp/sup.err" 2>&1 &
sup_pid=$!
ok=0
for _ in $(seq 1 200); do
  if grep -F -q -x "$log_line" "$tmp/sup.err" && grep -F -q 'autostart blocked:' "$tmp/sup.err"; then
    code=$(curl -s -m 2 -o "$tmp/gpu.json" -w '%{http_code}' -H 'Authorization: Bearer d10-token' http://127.0.0.1:18787/api/v1/gpu || true)
    if [[ "$code" == "200" ]] && grep -F -q '"available":false' "$tmp/gpu.json" && grep -F -q '"util":-1,' "$tmp/gpu.json"; then
      n=$(grep -o '"util"' "$tmp/gpu.json" | wc -l)
      n=${n// /}
      if [[ "$n" == "1" ]]; then
        ok=1
        break
      fi
    fi
  fi
  sleep 0.05
done
[[ "$ok" == "1" ]] || exit 1
if grep -F -q 'gpu sampler online' "$tmp/sup.err"; then exit 1; fi
if grep -F -q 'gpu sampler offline (no nvidia-smi / no GPU)' "$tmp/sup.err"; then exit 1; fi
log_n=$(grep -F -c -x "$log_line" "$tmp/sup.err" || true)
[[ "$log_n" == "1" ]] || exit 1
ps_ec=0
ps_out=$(./_bin/mortredctl.out --addr http://127.0.0.1:18787 --gateway http://127.0.0.1:18080 --token d10-token ps 2>"$tmp/ps.err") || ps_ec=$?
[[ "$ps_ec" == "1" ]] || exit 1
[[ ! -s "$tmp/ps.err" ]] || exit 1
[[ "$ps_out" != *'util -1%'* ]] || exit 1
gpu_line=$(printf '%s\n' "$ps_out" | grep -F '  GPU         ● FakeGPU  util ' || true)
[[ "$gpu_line" == '  GPU         ● FakeGPU  util n/a  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W' ]] || exit 1
printf '%s\n' 'available=false'
printf '%s\n' 'sample_count=1'
printf '%s\n' 'util=-1'
printf '%s\n' "$log_line"
printf '%s\n' "$gpu_line"
EOF
```

期望退出码 0。标准输出原文与验收 1 相同。

验收 3 的完整命令：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
sup_pid=
cleanup() {
  if [[ -n "${sup_pid:-}" ]]; then
    kill "$sup_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$sup_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    kill -9 "$sup_pid" 2>/dev/null || true
    wait "$sup_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/conf/server"
cat >"$tmp/bin/nvidia-smi" <<'SH'
#!/bin/sh
printf '%s\n' 'FakeGPU, 37, 2048, 8192, 40, 20.5, 1000, 5000, 30, 4, 16'
SH
chmod +x "$tmp/bin/nvidia-smi"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[supervisor]
api_host = "127.0.0.1"
api_port = 18787
autostart_default = false
gpu_sample_interval_ms = 60000
log_dir = "logs"

[gateway]
host = "127.0.0.1"
port = 18080
TOML
log_line='mortred-supervisor: gpu sampler online (FakeGPU)'
env -u MORTRED_CONTROL_CONFIG -u MORTRED_PACK -u MORTRED_PROFILE -u MORTRED_API_PORT -u MORTRED_API_HOST -u MORTRED_AUTOSTART -u MORTRED_EXPOSE -u MORTRED_UI_DIR -u APP_BIN_DIR -u APP_LIB_DIR -u APP_LIBS_DIR -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST \
  PATH="$tmp/bin:$PATH" \
  MORTRED_PROJECT_ROOT="$tmp" \
  MORTRED_API_TOKEN=d10-token \
  MORTRED_METRICS_TOKEN=d10-metrics \
  MORTRED_GATEWAY_AUTH_TOKEN=d10-infer \
  ./_bin/mortred-supervisor.out >"$tmp/sup.err" 2>&1 &
sup_pid=$!
ok=0
for _ in $(seq 1 200); do
  if grep -F -q -x "$log_line" "$tmp/sup.err" && grep -F -q 'autostart blocked:' "$tmp/sup.err"; then
    code=$(curl -s -m 2 -o "$tmp/gpu.json" -w '%{http_code}' -H 'Authorization: Bearer d10-token' http://127.0.0.1:18787/api/v1/gpu || true)
    if [[ "$code" == "200" ]] && grep -F -q '"available":true' "$tmp/gpu.json" && grep -F -q '"util":37,' "$tmp/gpu.json"; then
      n=$(grep -o '"util"' "$tmp/gpu.json" | wc -l)
      n=${n// /}
      if [[ "$n" == "1" ]]; then
        ok=1
        break
      fi
    fi
  fi
  sleep 0.05
done
[[ "$ok" == "1" ]] || exit 1
if grep -F -q 'gpu sampler offline (utilization N/A)' "$tmp/sup.err"; then exit 1; fi
if grep -F -q 'gpu sampler offline (no nvidia-smi / no GPU)' "$tmp/sup.err"; then exit 1; fi
log_n=$(grep -F -c -x "$log_line" "$tmp/sup.err" || true)
[[ "$log_n" == "1" ]] || exit 1
ps_ec=0
ps_out=$(./_bin/mortredctl.out --addr http://127.0.0.1:18787 --gateway http://127.0.0.1:18080 --token d10-token ps 2>"$tmp/ps.err") || ps_ec=$?
[[ "$ps_ec" == "1" ]] || exit 1
[[ ! -s "$tmp/ps.err" ]] || exit 1
[[ "$ps_out" != *'util -1%'* ]] || exit 1
[[ "$ps_out" != *'util n/a'* ]] || exit 1
gpu_line=$(printf '%s\n' "$ps_out" | grep -F '  GPU         ● FakeGPU  util ' || true)
[[ "$gpu_line" == '  GPU         ● FakeGPU  util 37%  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W' ]] || exit 1
printf '%s\n' 'available=true'
printf '%s\n' 'sample_count=1'
printf '%s\n' 'util=37'
printf '%s\n' "$log_line"
printf '%s\n' "$gpu_line"
EOF
```

期望退出码 0。标准输出原文见验收 3。

验收 4 的完整命令：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
sup_pid=
cleanup() {
  if [[ -n "${sup_pid:-}" ]]; then
    kill "$sup_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$sup_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    kill -9 "$sup_pid" 2>/dev/null || true
    wait "$sup_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/conf/server"
cat >"$tmp/bin/nvidia-smi" <<'SH'
#!/bin/sh
exit 0
SH
chmod +x "$tmp/bin/nvidia-smi"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[supervisor]
api_host = "127.0.0.1"
api_port = 18787
autostart_default = false
gpu_sample_interval_ms = 60000
log_dir = "logs"

[gateway]
host = "127.0.0.1"
port = 18080
TOML
log_line='mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)'
env -u MORTRED_CONTROL_CONFIG -u MORTRED_PACK -u MORTRED_PROFILE -u MORTRED_API_PORT -u MORTRED_API_HOST -u MORTRED_AUTOSTART -u MORTRED_EXPOSE -u MORTRED_UI_DIR -u APP_BIN_DIR -u APP_LIB_DIR -u APP_LIBS_DIR -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST \
  PATH="$tmp/bin:$PATH" \
  MORTRED_PROJECT_ROOT="$tmp" \
  MORTRED_API_TOKEN=d10-token \
  MORTRED_METRICS_TOKEN=d10-metrics \
  MORTRED_GATEWAY_AUTH_TOKEN=d10-infer \
  ./_bin/mortred-supervisor.out >"$tmp/sup.err" 2>&1 &
sup_pid=$!
ok=0
for _ in $(seq 1 200); do
  if grep -F -q -x "$log_line" "$tmp/sup.err" && grep -F -q 'autostart blocked:' "$tmp/sup.err"; then
    code=$(curl -s -m 2 -o "$tmp/gpu.json" -w '%{http_code}' -H 'Authorization: Bearer d10-token' http://127.0.0.1:18787/api/v1/gpu || true)
    if [[ "$code" == "200" ]] && grep -F -q '"available":false' "$tmp/gpu.json" && grep -F -q '"samples":[]' "$tmp/gpu.json" && ! grep -F -q '"util"' "$tmp/gpu.json"; then
      ok=1
      break
    fi
  fi
  sleep 0.05
done
[[ "$ok" == "1" ]] || exit 1
if grep -F -q 'gpu sampler online' "$tmp/sup.err"; then exit 1; fi
if grep -F -q 'gpu sampler offline (utilization N/A)' "$tmp/sup.err"; then exit 1; fi
log_n=$(grep -F -c -x "$log_line" "$tmp/sup.err" || true)
[[ "$log_n" == "1" ]] || exit 1
ps_ec=0
ps_out=$(./_bin/mortredctl.out --addr http://127.0.0.1:18787 --gateway http://127.0.0.1:18080 --token d10-token ps 2>"$tmp/ps.err") || ps_ec=$?
[[ "$ps_ec" == "1" ]] || exit 1
[[ ! -s "$tmp/ps.err" ]] || exit 1
[[ "$ps_out" != *'util -1%'* ]] || exit 1
[[ "$ps_out" != *'util n/a'* ]] || exit 1
[[ "$ps_out" != *'  GPU         '* ]] || exit 1
printf '%s\n' 'available=false'
printf '%s\n' 'sample_count=0'
printf '%s\n' "$log_line"
printf '%s\n' 'gpu_line=absent'
EOF
```

期望退出码 0。标准输出原文见验收 4。

## RD 不得自行决定

- 不得把空输出分支改成写 `gpu sampler offline (utilization N/A)`，也不得让 N/A 列去写 `gpu sampler offline (no nvidia-smi / no GPU)`。
- 不得用子串包含判断 N/A。裁剪后的 utilization 列必须整列等于 `N/A` 或 `[N/A]`。小写 `n/a`、空列、列数不足 11 都不走本条。
- 不得在这次 N/A 探测里打印 `gpu sampler online`，也不得每个采样间隔都再打印一行 offline。
- 不得丢掉这条 N/A 样本。验收要看到整数 `"util":-1` 和 CLI 的 `util n/a`。
- 不得把 `util` 改成字符串、null，或新增 JSON 键。
- 不得把 `util` 为 `0` 的探测印成 `util n/a` 或标成 offline。
- 不得改 `mortredctl ps` 的圆点、显存格式、`temp_c` 和功率的整数读取。
- 不得改 `conf/mortred.toml` 的 500 毫秒间隔，不得把验收端口改成产品的 `:8787` 或 `:8080`，不得在验收里拉起 gateway 或模型。
- 不得改 `occupancy_gate` 的 nvidia-smi 查询，不得改控制台页面。
- 不得按 cpu / gpu profile 关掉 sampler。
- 不得做 P0-2、P1-8、D-8、D-11、D-12，或重开已关闭原句。
