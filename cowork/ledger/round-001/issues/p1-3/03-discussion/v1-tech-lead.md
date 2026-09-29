# 技术方案

- 问题: P1-3
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md`

## 边界

只收口本机安装：安装树、`mortred` 用户、`/etc/mortred`、supervisor/edge 单元的安装与 enable。验收是一条命令。复用 `scripts/tarball_install.sh`，不平行再写一套安装器。不改 P0-1、P0-2，不做 P1-4。不默认 `enable --now`。不在本命令里拉权重或 prepare。

## 做法

### 推荐

抽出共享主机安装核。`tarball_install.sh` 与新入口 `mortredctl install-host` 调用同一核。可选 `--with-edge` 同时 enable edge。默认只 enable，不 start。

### 否决

只改注释、继续手搓。另写一套与 `tarball_install.sh` 分叉的安装器。

## 改动面

- 共享函数，供 `scripts/tarball_install.sh` 与 `scripts/mortredctl_install-host.sh` 使用。
- CLI 增加 `install-host`，与现有 `init` 一样分发到 `scripts/mortredctl_install-host.sh`。
- `deploy/mortred-supervisor.service` 头部注释改为指向这一条命令。

## 明确不改

P0-1 门禁、P0-2 起停语义、compose 轨、端口、产品 type、默认 start。

## 契约

`mortredctl install-host` 需要 root。最小参数：`--prefix`（默认 `/opt/mortred`）、来源 `--from-tarball-root` 或 `--from-build`、`--with-edge`。解压后 `sudo ./install.sh` 仍等价于不带 edge 的同一核。

## 性能与资源

一次性安装。不改变常驻进程和显存。

## 验收标准

1. **AC1**：单条 `sudo mortredctl install-host …` 或 `sudo bash scripts/mortredctl_install-host.sh …` 退出码 0，且 `systemctl is-enabled mortred-supervisor` 为 enabled。
2. **AC2**：用户 `mortred` 存在；prefix 下有 supervisor 二进制；`/etc/mortred/supervisor.env` mode 为 600。
3. **AC3**：`--with-edge` 后 `mortred-edge` 为 enabled；没有 nginx 时该开关退出码非 0。
4. **AC4**：重复执行同一条命令仍退出码 0。
5. **AC5**：安装成功不以 health 200 为条件，不改空 token 拒启。
6. **AC6**：验收记录写明可复述的完整命令。

无 systemd 的环境把 AC1–AC4 记 blocked，不把「脚本 `--help` 能跑」写成安装成功。

## 测试义务

黑盒记录命令和 `systemctl is-enabled`。非 root 调用必须非 0。

## 自测命令

| 命令 | 期望退出码 |
| --- | --- |
| `bash scripts/mortredctl_install-host.sh --help` | `0` |
| 非 root 调用安装 | `≠0` |
| `sudo bash scripts/mortredctl_install-host.sh --from-tarball-root <unpack>` | `0` |
| `systemctl is-enabled mortred-supervisor` | `0` |
| `id mortred` | `0` |
| `test "$(stat -c %a /etc/mortred/supervisor.env)" = 600` | `0` |

## RD 不得自行决定

不得默认 start。不得另起第二套安装器。不得改 P0 行为或做 P1-4。不得在本命令里强制 fetch_weights 或 prepare。
