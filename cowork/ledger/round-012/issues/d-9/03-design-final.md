# 技术方案

- 问题: D-9
- 版本: v1
- 对应列表: `cowork/ledger/round-012/02-p0-p1-final.md` 的 D-9

## 边界

只做已经冻结的 D-9。冻结句不改写：推理请求在鉴权失败时仍消耗限流配额。

改完后能观察到的变化只这三句：上述 401 不再占配额，紧接着的合法推理进入转发；`enabled = false` 时这道门不计数也不回 429；无 inference 权限的 key 连续两次都是 401。

本条没有上一版方案，没有待并入的审查条件。

现在 401 占配额的位置有两处。

- 网关按来源的门在鉴权之前计数。`GatewayApp::process` 先调用 `check_ip_rate_limit`，再解析路由，再做 Bearer 鉴权。`check_ip_rate_limit` 在 `ip_limiter_` 非空时调用 `admit`，不读 `rate_limit_enabled_`。`init` 总是用 toml 里的 `rate_per_sec` 和 `burst` 构造 `ShardedIpLimiter`。`GcraPolicy::enabled()` 只看这两个数是否大于 0。`[gateway.rate_limit]` 的 `enabled` 和 `shadow` 只参与启动日志：`rate_limit_enabled_ = enabled || shadow`。仓库 `conf/mortred.toml` 是 `enabled = false`、`shadow = false`、`rate_per_sec = 100000`、`burst = 200000`，策略仍然计数。把这两个数改成 1 且保持 `enabled = false` 时，同一来源先对推理路由发无 Bearer 请求：`admit` 吃掉 burst 1，随后 401；同一秒内再带正确 token，`admit` 拒绝，响应是 429，到不了 `forward_to_model`。
- 已识别且 key 自身 `enabled = true`、但 `key_may_infer` 为假的 key，在 scope 判定之前扣每 key 窗口。`ApiKeyManager::authenticate` 找到该 key 后调用 `allow_rate_limit`，窗口加一。`process` 在这之后才调用 `key_may_infer`（`inference`、`admin`、`all` 可以推理）。`rate_limit_qps = 1` 且 scope 为 `none` 时，第一次推理是 401，第二次在窗口内因 `AuthReason::RATE_LIMITED` 返回 429，文案是 `api key rate limit exceeded`。无 Bearer、未知 key、key 自身 `enabled = false` 的 key 在 `allow_rate_limit` 之前就返回，不走这条窗口。

模型进程已经在 `allow()` 之前返回 401。本条不改那一侧。

「进入转发」的黑盒原文是：上游端口没有进程，网关已调用 `forward_to_model`，对端连接被拒绝，HTTP 503，正文是现有的 `service not ready` 信封。验收不要求推理 200。

不覆盖 P0-2、P1-8、D-8、D-10、D-11、D-12。不重开已关闭原句：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、D-1 至 D-6。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`centerface` 的 engine 输入名 `input.1` 不重开。

## 做法

### 推荐

两处一起改，缺一不可。

按来源的门：`GatewayApp::init` 在读完 `[gateway.rate_limit]`、并应用 `MORTRED_RATE_LIMIT`（`off` / `shadow` / `enforce`）之后，只有 `enabled` 或 `shadow` 为真才计数。两者都为假时，`check_ip_rate_limit` 直接放行，不调用 `admit`，不写 429，不为这次请求留下按来源的额度。`rate_per_sec` 和 `burst` 仍留在配置里，关闭时不生效。`MORTRED_RATE_LIMIT=off` 把两个标志都写成假，走这条关闭路径。`enforce` 仍按现有 `rate_per_sec` 与 `burst` 计数并可以回 429。`shadow = true` 仍计数，拒绝时仍放行且不回 429。

每 key 的窗口：`allow_rate_limit` 只在 `key_may_infer` 已经为真之后增加 `rate_window_count`。scope 为 `none` 的 key 连续两次推理都走现有 401，窗口不加。`inference`、`admin`、`all` 仍可以推理；这类 key 在 `rate_limit_qps = 1` 时，同一秒内第二次仍是现有 429，文案仍是 `api key rate limit exceeded`。

改动模块是 `src/control/gateway/gateway_app.cpp` 与 `src/control/api_key_manager.h`。模型进程、GCRA 公式、产品 toml 数值不改。HTTP 上会变的只有冻结句里的那两种 401：它们不再变成下一次的 429；`enabled = false` 且 `shadow = false` 时按来源的门不回 429。401、503、有权限 key 的 429 正文保持现有原文。常驻进程仍是一个 gateway；本条不占显存，不生成 engine。QA 只跑下面的黑盒命令。RD 的自测命令与验收命令相同。RD 不能自己改 `enabled = true` 的匿名顺序、`shadow` 的计数方式，也不能把无权限改成 403。

代价：`enabled = false` 且 `shadow = false` 时，这道按来源的门不再把匿名推理收成 429，匿名请求得到 401 或路由失败。无 inference 权限的 key 可以连续得到 401，不再被自己的 `rate_limit_qps` 收成 429。有推理权限的 key，以及 `enabled = true` 的按来源门，仍会计数。不增加常驻进程，不增加显存，不生成 engine。

### 否决

不论 `enabled` 真假，都把 `admit` 挪到鉴权成功之后。`enabled = true` 时，匿名请求不再在路由前被这道门丢掉，冻结的三条观察里没有这条。若只挪顺序、仍无视 `enabled = false`，则 `burst = 1` 时同一秒内第二次合法推理仍是 429，冻结句「这道门不计数也不回 429」达不到。这条否决。

401 之后把已经 `admit` 的额度退回，或只改每 key 窗口、不改按来源的门。退回仍先计数，同一来源上并发的合法请求会看到被占掉的 burst；`enabled = false` 时第二次合法推理仍会 429。只改 key 窗口时，无 Bearer 的 401 仍占 `burst = 1`，紧接着的合法推理仍是 429。这条否决。

## 改动面

- `src/control/gateway/gateway_app.cpp`：`init` 与 `check_ip_rate_limit`。`enabled` 与 `shadow` 都为假时不 `admit`、不回 429。`process` 里对已识别 key 的窗口消耗放在 `key_may_infer` 为真之后。
- `src/control/api_key_manager.h`：`allow_rate_limit` 不再在 scope 被接受之前增加 `rate_window_count`。

## 明确不改

- `src/control/rate_limit/gcra.h` 的间隔、burst 容量和拒绝计算。`enabled = true` 时仍用配置里的 `rate_per_sec` 与 `burst`。
- `src/server/base_server_impl.h` 的鉴权与 `allow()` 顺序。模型端口的 401 保持在 IP QPS 之前。
- `conf/mortred.toml` 的现有数值：`enabled = false`、`shadow = false`、`rate_per_sec = 100000`、`burst = 200000`。验收用临时 toml，不改这个产品文件。
- `conf/server/object_detection/yolov8/yolov8s_server_config.toml` 的 `port=9056`。验收只在临时目录里把副本端口改成命令中的端口。
- `key_may_infer` 的范围：`inference`、`admin`、`all` 可以推理。
- `/healthz`、`/metrics`、`OPTIONS` 相对这道门的位置。`/healthz` 仍在这道门之前返回。`GET /metrics` 的 401 仍在这道门之前返回。
- `enabled = true` 时匿名请求先经过这道门的顺序。
- `shadow = true` 时的行为：计数，不回 429。
- 401 正文、503 转发失败正文、有推理权限的 key 触发的 429 正文。
- 产品 toml 的 `type=mnn` / `type=tensorrt`，HTTP 成功信封，张量契约，gateway `:8080` 与 supervisor `:8787` 的产品端口。一个 catalog id 仍是一个 OS 进程。
- engine 仍在本机本卡用 `mortredctl prepare` 生成。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。
- P0-2、P1-8、D-8、D-10、D-11、D-12，以及已关闭原句。

## 契约

产品 toml 的键名不变。`[gateway.rate_limit]` 的 `enabled = false` 且 `shadow = false` 时，这道按来源的门不计数，也不产生 HTTP 429。`enabled = true` 时，按来源的 429 文案仍是 `per-source rate limit exceeded`。`shadow = true` 时仍不回 429。

无 Bearer 的推理路由，以及 scope 为 `none` 的 key，HTTP 状态仍是 401。正文仍是下面这一行，网关本身不另加换行：

`{"status":401,"status_str":"unauthorized","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"unauthorized"}]}`

上游没有进程时，已经转发的推理 HTTP 状态仍是 503。正文仍是下面这一行：

`{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}`

有推理权限且 `rate_limit_qps = 1` 的 key，同一秒内第二次推理 HTTP 状态仍是 429。正文仍是下面这一行：

`{"status":429,"status_str":"too many requests","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"api key rate limit exceeded"}]}`

无推理权限的 key 的 HTTP 状态保持 401，正文保持上面的 `unauthorized` 行。无张量契约变化。gateway 产品端口仍是 `:8080`，supervisor 仍是 `:8787`。

## 性能与资源

验收只拉起一个临时 gateway 进程，不拉起 supervisor，不拉起模型进程，不占显存，不生成 engine。`enabled = false` 且 `shadow = false` 时，这道门不为验收里的请求保存按来源额度。`enabled = true` 时，现有 GCRA 与 `max_tracked` 上限保持原样。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

三条都在 Linux 上、仓库根目录执行。前置：已有 `_bin/mortred-gateway.out`。三条分开跑。每条的标准错误为空。标准输出原文如下，每行以换行结束。两条 curl 之间不 sleep。临时目录在命令结束时删除。临时上游端口上不能有进程。命令不修改仓库里的 `conf/mortred.toml`、`conf/api_keys.toml` 和 `conf/server` 下的产品文件。

验收 1。退出码 0。命令：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
gw_pid=
cleanup() {
 if [[ -n "${gw_pid:-}" ]]; then
 kill "$gw_pid" 2>/dev/null || true
 wait "$gw_pid" 2>/dev/null || true
 fi
 rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/conf/server/object_detection/yolov8"
cp conf/server/object_detection/yolov8/yolov8s_server_config.toml "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
sed -i 's/^port=9056$/port=19056/' "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[gateway]
host = "127.0.0.1"
port = 18080

[gateway.rate_limit]
enabled = false
shadow = false
rate_per_sec = 1
burst = 1
TOML
env -u MORTRED_API_TOKEN -u MORTRED_RATE_LIMIT -u MORTRED_CONTROL_CONFIG -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST -u MORTRED_PROFILE -u MORTRED_INTERNAL_TOKEN -u MORTRED_EXPOSE -u MORTRED_GATEWAY_CORS_ORIGINS \
 MORTRED_PROJECT_ROOT="$tmp" \
 MORTRED_GATEWAY_AUTH_TOKEN=d9-infer \
 MORTRED_METRICS_TOKEN=d9-metrics \
 ./_bin/mortred-gateway.out >"$tmp/gw.err" 2>&1 &
gw_pid=$!
ok=0
for _ in $(seq 1 100); do
 if curl -sS -m 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18080/healthz 2>/dev/null | grep -qx 200; then
 ok=1
 break
 fi
 sleep 0.05
done
[[ "$ok" == 1 ]] || exit 1
b1=$(mktemp)
b2=$(mktemp)
c1=$(curl -sS -m 5 -o "$b1" -w '%{http_code}' -X POST http://127.0.0.1:18080/v1/models/yolov8s/infer -H 'Content-Type: application/json' -d '{}')
c2=$(curl -sS -m 5 -o "$b2" -w '%{http_code}' -X POST http://127.0.0.1:18080/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-infer' -d '{}')
body1=$(cat "$b1")
body2=$(cat "$b2")
printf '%s\n' "$c1"
printf '%s\n' "$body1"
printf '%s\n' "$c2"
printf '%s\n' "$body2"
rm -f "$b1" "$b2"
unauth='{"status":401,"status_str":"unauthorized","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"unauthorized"}]}'
fwd='{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}'
[[ "$c1" == "401" && "$body1" == "$unauth" && "$c2" == "503" && "$body2" == "$fwd" ]]
EOF
```

标准输出原文：

```
401
{"status":401,"status_str":"unauthorized","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"unauthorized"}]}
503
{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}
```

验收 2。退出码 0。命令与验收 1 相同，只改这三处：gateway 端口 `18081`，副本 `port=19057`，两次请求都带 `Authorization: Bearer d9-infer`。判定行改为两次状态都是 `503` 且两次正文都等于验收 1 里的 `fwd`。标准输出原文：

```
503
{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}
503
{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}
```

验收 3。退出码 0。在验收 1 的临时根上再写 `conf/api_keys.toml`：明文 `d9-no-infer` 的 SHA-256 写入 `hash`，`scope = "none"`，`rate_limit_qps = 1`，`enabled = true`。gateway 端口 `18082`，副本 `port=19058`。两次请求都带 `Authorization: Bearer d9-no-infer`。判定行改为两次状态都是 `401` 且两次正文都等于验收 1 里的 `unauth`。标准输出原文：

```
401
{"status":401,"status_str":"unauthorized","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"unauthorized"}]}
401
{"status":401,"status_str":"unauthorized","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"unauthorized"}]}
```

验收 2 与验收 3 的完整命令见自测命令。通过条件仍是各自的退出码、标准输出原文和空的标准错误。

## 测试义务

QA 跑验收 1、验收 2、验收 3，以及下面这条回归。通过与否只看退出码、标准输出原文和空的标准错误。不把读源码、拉起 supervisor、拉起模型、生成 engine、跑 GPU 数值比较写成通过条件。

回归命令与验收 3 的启动方式相同，换成：gateway 端口 `18083`，副本 `port=19059`，明文 `d9-may-infer`，`scope = "inference"`，`rate_limit_qps = 1`，`enabled = true`。两次请求都带 `Authorization: Bearer d9-may-infer`。退出码 0。标准输出原文：

```
503
{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}
429
{"status":429,"status_str":"too many requests","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"api key rate limit exceeded"}]}
```

这条回归钉住有 inference 权限的 key 仍按 `rate_limit_qps` 回 429。它不增加冻结句里的完成范围。

## 自测命令

在 Linux 上、仓库根目录执行。四条分开跑。期望退出码都是 0。标准错误都为空。

验收 1 的命令与标准输出见上一节。期望退出码 0。

验收 2：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
gw_pid=
cleanup() {
 if [[ -n "${gw_pid:-}" ]]; then
 kill "$gw_pid" 2>/dev/null || true
 wait "$gw_pid" 2>/dev/null || true
 fi
 rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/conf/server/object_detection/yolov8"
cp conf/server/object_detection/yolov8/yolov8s_server_config.toml "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
sed -i 's/^port=9056$/port=19057/' "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[gateway]
host = "127.0.0.1"
port = 18081

[gateway.rate_limit]
enabled = false
shadow = false
rate_per_sec = 1
burst = 1
TOML
env -u MORTRED_API_TOKEN -u MORTRED_RATE_LIMIT -u MORTRED_CONTROL_CONFIG -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST -u MORTRED_PROFILE -u MORTRED_INTERNAL_TOKEN -u MORTRED_EXPOSE -u MORTRED_GATEWAY_CORS_ORIGINS \
 MORTRED_PROJECT_ROOT="$tmp" \
 MORTRED_GATEWAY_AUTH_TOKEN=d9-infer \
 MORTRED_METRICS_TOKEN=d9-metrics \
 ./_bin/mortred-gateway.out >"$tmp/gw.err" 2>&1 &
gw_pid=$!
ok=0
for _ in $(seq 1 100); do
 if curl -sS -m 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18081/healthz 2>/dev/null | grep -qx 200; then
 ok=1
 break
 fi
 sleep 0.05
done
[[ "$ok" == 1 ]] || exit 1
b1=$(mktemp)
b2=$(mktemp)
c1=$(curl -sS -m 5 -o "$b1" -w '%{http_code}' -X POST http://127.0.0.1:18081/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-infer' -d '{}')
c2=$(curl -sS -m 5 -o "$b2" -w '%{http_code}' -X POST http://127.0.0.1:18081/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-infer' -d '{}')
body1=$(cat "$b1")
body2=$(cat "$b2")
printf '%s\n' "$c1"
printf '%s\n' "$body1"
printf '%s\n' "$c2"
printf '%s\n' "$body2"
rm -f "$b1" "$b2"
fwd='{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}'
[[ "$c1" == "503" && "$body1" == "$fwd" && "$c2" == "503" && "$body2" == "$fwd" ]]
EOF
```

期望退出码 0。标准输出原文与验收 2 相同。

验收 3：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
gw_pid=
cleanup() {
 if [[ -n "${gw_pid:-}" ]]; then
 kill "$gw_pid" 2>/dev/null || true
 wait "$gw_pid" 2>/dev/null || true
 fi
 rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/conf/server/object_detection/yolov8"
cp conf/server/object_detection/yolov8/yolov8s_server_config.toml "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
sed -i 's/^port=9056$/port=19058/' "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[gateway]
host = "127.0.0.1"
port = 18082

[gateway.rate_limit]
enabled = false
shadow = false
rate_per_sec = 1
burst = 1
TOML
hash=$(printf '%s' 'd9-no-infer' | sha256sum | awk '{print $1}')
cat >"$tmp/conf/api_keys.toml" <<TOML
[keys.no-infer]
hash = "${hash}"
scope = "none"
rate_limit_qps = 1
enabled = true
TOML
env -u MORTRED_API_TOKEN -u MORTRED_RATE_LIMIT -u MORTRED_CONTROL_CONFIG -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST -u MORTRED_PROFILE -u MORTRED_INTERNAL_TOKEN -u MORTRED_EXPOSE -u MORTRED_GATEWAY_CORS_ORIGINS \
 MORTRED_PROJECT_ROOT="$tmp" \
 MORTRED_GATEWAY_AUTH_TOKEN=d9-infer \
 MORTRED_METRICS_TOKEN=d9-metrics \
 ./_bin/mortred-gateway.out >"$tmp/gw.err" 2>&1 &
gw_pid=$!
ok=0
for _ in $(seq 1 100); do
 if curl -sS -m 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18082/healthz 2>/dev/null | grep -qx 200; then
 ok=1
 break
 fi
 sleep 0.05
done
[[ "$ok" == 1 ]] || exit 1
b1=$(mktemp)
b2=$(mktemp)
c1=$(curl -sS -m 5 -o "$b1" -w '%{http_code}' -X POST http://127.0.0.1:18082/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-no-infer' -d '{}')
c2=$(curl -sS -m 5 -o "$b2" -w '%{http_code}' -X POST http://127.0.0.1:18082/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-no-infer' -d '{}')
body1=$(cat "$b1")
body2=$(cat "$b2")
printf '%s\n' "$c1"
printf '%s\n' "$body1"
printf '%s\n' "$c2"
printf '%s\n' "$body2"
rm -f "$b1" "$b2"
unauth='{"status":401,"status_str":"unauthorized","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"unauthorized"}]}'
[[ "$c1" == "401" && "$body1" == "$unauth" && "$c2" == "401" && "$body2" == "$unauth" ]]
EOF
```

期望退出码 0。标准输出原文与验收 3 相同。

回归：

```bash
bash <<'EOF'
tmp=$(mktemp -d)
gw_pid=
cleanup() {
 if [[ -n "${gw_pid:-}" ]]; then
 kill "$gw_pid" 2>/dev/null || true
 wait "$gw_pid" 2>/dev/null || true
 fi
 rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/conf/server/object_detection/yolov8"
cp conf/server/object_detection/yolov8/yolov8s_server_config.toml "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
sed -i 's/^port=9056$/port=19059/' "$tmp/conf/server/object_detection/yolov8/yolov8s_server_config.toml"
cat >"$tmp/conf/mortred.toml" <<'TOML'
[gateway]
host = "127.0.0.1"
port = 18083

[gateway.rate_limit]
enabled = false
shadow = false
rate_per_sec = 1
burst = 1
TOML
hash=$(printf '%s' 'd9-may-infer' | sha256sum | awk '{print $1}')
cat >"$tmp/conf/api_keys.toml" <<TOML
[keys.may-infer]
hash = "${hash}"
scope = "inference"
rate_limit_qps = 1
enabled = true
TOML
env -u MORTRED_API_TOKEN -u MORTRED_RATE_LIMIT -u MORTRED_CONTROL_CONFIG -u MORTRED_GATEWAY_PORT -u MORTRED_GATEWAY_HOST -u MORTRED_PROFILE -u MORTRED_INTERNAL_TOKEN -u MORTRED_EXPOSE -u MORTRED_GATEWAY_CORS_ORIGINS \
 MORTRED_PROJECT_ROOT="$tmp" \
 MORTRED_GATEWAY_AUTH_TOKEN=d9-infer \
 MORTRED_METRICS_TOKEN=d9-metrics \
 ./_bin/mortred-gateway.out >"$tmp/gw.err" 2>&1 &
gw_pid=$!
ok=0
for _ in $(seq 1 100); do
 if curl -sS -m 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:18083/healthz 2>/dev/null | grep -qx 200; then
 ok=1
 break
 fi
 sleep 0.05
done
[[ "$ok" == 1 ]] || exit 1
b1=$(mktemp)
b2=$(mktemp)
c1=$(curl -sS -m 5 -o "$b1" -w '%{http_code}' -X POST http://127.0.0.1:18083/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-may-infer' -d '{}')
c2=$(curl -sS -m 5 -o "$b2" -w '%{http_code}' -X POST http://127.0.0.1:18083/v1/models/yolov8s/infer -H 'Content-Type: application/json' -H 'Authorization: Bearer d9-may-infer' -d '{}')
body1=$(cat "$b1")
body2=$(cat "$b2")
printf '%s\n' "$c1"
printf '%s\n' "$body1"
printf '%s\n' "$c2"
printf '%s\n' "$body2"
rm -f "$b1" "$b2"
fwd='{"status":65,"status_str":"service not ready","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"model server not running or still loading; TensorRT ids need mortredctl prepare"}]}'
limited='{"status":429,"status_str":"too many requests","task_id":"","model":{"name":"","version":""},"results":[],"server_time_ms":0.0,"partial":false,"errors":[{"pointer":"","message":"api key rate limit exceeded"}]}'
[[ "$c1" == "503" && "$body1" == "$fwd" && "$c2" == "429" && "$body2" == "$limited" ]]
EOF
```

期望退出码 0。标准输出原文与测试义务里的回归相同。

## RD 不得自行决定

- 改 `enabled = true` 时匿名请求先经过按来源门的顺序，或把 `shadow = true` 改成不计数，或让 `shadow = true` 回 429。
- 用把 `rate_per_sec` 的最小值改成 0、或改写 `conf/mortred.toml` 里的 `100000` / `200000`，代替 `enabled = false` 这道开关。
- 把无 inference 权限的 key 改成 403，或改成 `api key rate limit exceeded` 的 429。
- 让 `admin` 或 `all` 不能推理，或让 scope `none` 进入转发。
- 删掉有 inference 权限且 `rate_limit_qps = 1` 的 key 在同一秒内第二次的 429。
- 改 401、503、有权限 key 的 429 的 JSON 原文，包括字段顺序、`server_time_ms` 的 `0.0` 和错误文案。
- 为了让验收得到 200 去拉起模型、生成 engine，或改 `yolov8s` 产品 toml 的 `port=9056`。
- 把 `conf/api_keys.toml` 写进仓库里的 `conf/`。验收 key 只存在于临时 `MORTRED_PROJECT_ROOT`。
- 做 P0-2、P1-8、D-8、D-10、D-11、D-12，或重开已关闭原句。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 产品 | 同意 | 无 | v1 | `03-discussion/v1-sign-pm.md` |
| 研发 | 同意 | 无 | v1 | `03-discussion/v1-sign-rd.md` |
| 技术负责人 | 正文 | 无 | v1 | `03-discussion/v1-tech-lead.md` |
