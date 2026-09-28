# 模块评分

- id: ops
- 权重: 15
- 分数: 7.3
- 读过的路径:
  - `scripts/mortredctl_init.sh`
  - `scripts/mortredctl_prepare.sh`
  - `scripts/mortredctl_doctor.sh`
  - `docker-compose.yml`
  - `deploy/`（列出并读入口：`deploy/mortred-supervisor.service`、`deploy/nginx/mortred-edge.service`、`deploy/nginx/nginx.conf.skeleton`、`deploy/docker-compose.monitoring.yml`；目录另含 nginx snippets、prometheus/grafana 配置与 `alert-rules.yml`）

## 优势

- 首小时入口清晰：`mortredctl init` 自动判 cpu/gpu、写 `PROFILE`、按 profile 拉权重并跑 `--basic` 校验（`scripts/mortredctl_init.sh` L26–72），小团队有单一上机动作。
- 验收可定位：`mortredctl doctor` 查 ASan 链、pack occupancy、pack TRT，supervisor 可达走 `--live`、不可达回退 `--basic` 并打印 `next:`（`scripts/mortredctl_doctor.sh` L29–90），出错时有可执行下一步。
- 容器交付分档完整：`docker compose --profile gpu|cpu`，强制三 token、只绑 `127.0.0.1:8787/8080`、healthcheck + restart（`docker-compose.yml` L18–79），默认不把明文推理口暴露到 LAN。
- 主机与边缘有落盘单元：supervisor systemd、`init-trust`/`init-edge` 注释链路，以及 Nginx host-network edge（`deploy/mortred-supervisor.service`、`deploy/nginx/mortred-edge.service`、`deploy/nginx/nginx.conf.skeleton`）。
- GPU engine 有 pack 级入口：`mortredctl prepare` 明确转到 pack-scoped TRT convert + `/ready`（`scripts/mortredctl_prepare.sh`；doctor 缺 engine 时指向 `mortredctl prepare`，`scripts/mortredctl_doctor.sh` L62–69）。

## 劣势

- init 对交付关键失败偏软：权重拉取失败只 WARN 仍打印 `init done`；GPU 段只 list/提示 `convert_trt_engines.sh`，不强制本机本卡准备 engine（`scripts/mortredctl_init.sh` L45–64、L74–78）。现场经验：小团队易以为「init 完就能上客户 TRT 包」。
- doctor 默认 fail-open：缺 occupancy stamp / TRT engine 仅 WARN，只有 `--strict` 才因门禁失败退出（`scripts/mortredctl_doctor.sh` L49–77）；非严格验收可把「未就绪」当成「可交付」。
- 主机安装仍是注释驱动手工：useradd、`cmake --install`、拷 `supervisor.env`/edge、enable 单元等未收进单一命令（`deploy/mortred-supervisor.service` L3–14），换机重复成本高。
- prepare 运维入口过薄：`mortredctl_prepare.sh` 仅 `exec prepare_pack.sh`，失败语义与步骤全在下游脚本，入口本身无可读验收逻辑（`scripts/mortredctl_prepare.sh`）。

## 依据

- 分数 7.3：有完整 mortredctl 主链路与 docker/systemd 入口，小团队跟 doctor/`next` 能交付并排查；但 init/doctor 对权重与 GPU engine 默认不 fail-closed，主机路径仍多手工，距「独立交付且出错必拦」的 9+ 差一截（出处见上列脚本与 `docker-compose.yml` / `deploy/*`）。

## 未读

无
