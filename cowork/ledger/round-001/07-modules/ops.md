# 模块评分

- id: ops
- 权重: 15
- 分数: 8.5
- 读过的路径:
  - `scripts/mortredctl_init.sh`
  - `scripts/mortredctl_prepare.sh`
  - `scripts/mortredctl_doctor.sh`
  - `docker-compose.yml`
  - `deploy/`（`deploy/mortred-supervisor.service`、`deploy/nginx/mortred-edge.service`）
  - `scripts/mortredctl_install-host.sh`、`scripts/install_host_common.sh`、`scripts/tarball_install.sh`、`scripts/test_p0_1_engine_fail_closed.sh`

## 优势
- init 对交付关键门禁 fail-closed：权重拉取失败或 GPU pack 缺 TRT engine 时置 FAIL 并 `exit 1`，不再打印 `init done`；CPU 不要求 engine；GPU 明确指向 `mortredctl prepare`（`scripts/mortredctl_init.sh`）。
- doctor 默认 fail-closed 且可定位：occupancy / pack TRT 失败即退出并点名路径与 `next:`；supervisor 可达时解析 autostart blocked / gateway failed 并 FAIL（`scripts/mortredctl_doctor.sh`）。
- 主机安装收成单一命令：`mortredctl install-host` 建用户、拷树、写 `supervisor.env`、装单元并 `systemctl enable`，不 start。
- 容器与边缘入口仍完整：compose gpu/cpu 强制三 token、只绑 loopback、healthcheck。

## 劣势
- prepare 运维入口仍过薄：`mortredctl_prepare.sh` 仅 `exec prepare_pack.sh`，失败语义与步骤全在下游。
- 主机装完仍有手工收尾：install-host 只 enable 不 start，且要求事后填 `/etc/mortred/supervisor.env` 再 `systemctl start`。

## 依据
- 分数 8.5（相对上轮 7.3）：init/doctor fail-open 与主机纯注释装机已落地为 fail-closed + `install-host`。距 9+ 仍差在 prepare 入口薄、装机后 token/start 手递。

## 未读
无
